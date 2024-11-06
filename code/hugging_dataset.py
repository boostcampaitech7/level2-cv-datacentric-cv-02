import os.path as osp
import math
import json
from PIL import Image
from PIL import ImageDraw, ImageFont, ImageEnhance

import torch
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset
from shapely.geometry import Polygon
from numba import njit

@njit
def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

@njit
def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices

@njit
def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1,y1,x2,y2), cal_distance(x1,y1,x4,y4))
    r2 = min(cal_distance(x2,y2,x1,y1), cal_distance(x2,y2,x3,y3))
    r3 = min(cal_distance(x3,y3,x2,y2), cal_distance(x3,y3,x4,y4))
    r4 = min(cal_distance(x4,y4,x1,y1), cal_distance(x4,y4,x3,y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4) > \
       cal_distance(x2,y2,x3,y3) + cal_distance(x1,y1,x4,y4):
        offset = 0 # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1 # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v

@njit
def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4,2)).T
    if anchor is None:
        anchor = v[:,:1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)

@njit
def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max

@njit
def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err

@njit
def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
    '''check if the crop image crosses text regions
    Input:
        start_loc: left-top position
        length   : length of crop image
        vertices : vertices of text regions <numpy.ndarray, (n,8)>
    Output:
        True if crop image crosses text region
    '''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h, start_w + length, start_h + length,
                  start_w, start_h + length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True
    return False


def crop_img(img, vertices, labels, length):
    '''crop img patches to obtain batch and augment
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        length      : length of cropped image region
    Output:
        region      : cropped image region
        new_vertices: new vertices in cropped region
    '''
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert(ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h

    # find random position
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels==1,:])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:,[0,2,4,6]] -= start_w
    new_vertices[:,[1,3,5,7]] -= start_h
    return region, new_vertices

@njit
def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                                                   np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def resize_img(img, vertices, size):
    h, w = img.height, img.width
    ratio = size / max(h, w)
    if w > h:
        img = img.resize((size, int(h * ratio)), Image.BILINEAR)
    else:
        img = img.resize((int(w * ratio), size), Image.BILINEAR)
    new_vertices = vertices * ratio
    return img, new_vertices


def adjust_height(img, vertices, ratio=0.2):
    '''adjust height of image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
    return img, new_vertices


def rotate_img(img, vertices, angle_range=90):
    '''rotate image [-10, 10] degree to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range : rotate range
    Output:
        img         : rotated PIL Image
        new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
    return img, new_vertices


def generate_roi_mask(image, vertices, labels):
    mask = np.ones(image.shape[:2], dtype=np.float32)
    ignored_polys = []
    for vertice, label in zip(vertices, labels):
        if label == 0:
            ignored_polys.append(np.around(vertice.reshape((4, 2))).astype(np.int32))
    cv2.fillPoly(mask, ignored_polys, 0)
    return mask


def filter_vertices(vertices, labels, ignore_under=0, drop_under=0):
    if drop_under == 0 and ignore_under == 0:
        return vertices, labels

    new_vertices, new_labels = vertices.copy(), labels.copy()

    areas = np.array([Polygon(v.reshape((4, 2))).convex_hull.area for v in vertices])
    labels[areas < ignore_under] = 0

    if drop_under > 0:
        passed = areas >= drop_under
        new_vertices, new_labels = new_vertices[passed], new_labels[passed]

    return new_vertices, new_labels

# 추가한 augmentations
def scale_image(image, vertices, scale_factor):
    """
    Scale the image by a scale_factor and adjust vertices accordingly.
    """
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    img = image.resize(new_size, Image.BILINEAR)
    new_vertices = vertices * scale_factor
    return img, new_vertices

def translate_image(image, vertices, x_offset=None, y_offset=None):
    """
    Translate the image and vertices by random or specified x_offset and y_offset.
    If offsets are not provided, they are generated randomly within 5% ~ 10% of image dimensions.
    """
    if x_offset is None:
        x_offset = np.random.randint(-int(image.width * 0.1), int(image.width * 0.1))
    if y_offset is None:
        y_offset = np.random.randint(-int(image.height * 0.1), int(image.height * 0.1))
    
    img = Image.new("RGB", (image.width + abs(x_offset), image.height + abs(y_offset)))
    img.paste(image, (x_offset, y_offset))
    
    new_vertices = vertices.copy()
    new_vertices[:, [0, 2, 4, 6]] += x_offset
    new_vertices[:, [1, 3, 5, 7]] += y_offset
    
    return img, new_vertices

def perspective_transform(image, vertices):
    """
    Apply a perspective transformation to the image and adjust vertices accordingly.
    """
    width, height = image.size
    pts1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    shift = np.random.randint(-width // 10, width // 10, size=(4, 2))
    pts2 = (pts1 + shift).astype(np.float32)  # Ensure pts2 is float32

    # Calculate perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    # Transform image
    img = image.transform((width, height), Image.PERSPECTIVE, M.flatten()[:8], Image.BICUBIC)
    
    # Transform vertices
    new_vertices = np.zeros_like(vertices)
    for i, vertice in enumerate(vertices):
        reshaped_vertice = vertice.reshape(-1, 2).astype(np.float32)
        transformed_vertice = cv2.perspectiveTransform(np.array([reshaped_vertice]), M)
        new_vertices[i] = transformed_vertice.reshape(-1)
        
    return img, new_vertices

def adjust_brightness_contrast_saturation(image, brightness=1.0, contrast=1.0, saturation=1.0):
    """
    Adjust brightness, contrast, and saturation of the image.
    """
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(saturation)
    return image

def add_gaussian_noise(image, mean=0, std_range=(0.01, 0.05)):
    """
    Add Gaussian noise to the image with a random standard deviation within a given range.
    """
    std = np.random.uniform(*std_range)
    np_image = np.array(image) / 255.0
    noise = np.random.normal(mean, std, np_image.shape)
    noisy_image = np.clip(np_image + noise, 0, 1) * 255
    
    return Image.fromarray(noisy_image.astype(np.uint8))

def overlay_text(image, text="Sample Text", position=None, font_size=15, color=(0, 0, 0)):
    """
    Overlay synthetic text onto the image.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    draw = ImageDraw.Draw(image)

    if position is None:
        position = (np.random.randint(0, image.width - 100), np.random.randint(0, image.height - 50))
    draw.text(position, text, fill=color)

    return image

class (Dataset):
    def __init__(self, root_dir,  # root_dir을 hugging/data 경로로 지정
                 split='train',
                 image_size=2048,
                 crop_size=1024,
                 ignore_under_threshold=10,
                 drop_under_threshold=1,
                 color_jitter=True,
                 normalize=True):
        self.root_dir = root_dir
        self.split = split
        
        # ufo_cord_data.json 파일 로드
        with open(osp.join(self.root_dir, 'ufo_cord_data.json'), 'r', encoding='utf-8') as f:
            self.anno = json.load(f)
        
        self.image_fnames = sorted(self.anno['images'].keys())
        self.image_size, self.crop_size = image_size, crop_size
        self.color_jitter, self.normalize = color_jitter, normalize
        self.drop_under_threshold = drop_under_threshold
        self.ignore_under_threshold = ignore_under_threshold

    def __len__(self):
        return len(self.image_fnames)
        
    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_info = self.anno['images'][image_fname]
        
        # image_info['img_path']가 절대 경로가 아닌 상대 경로라고 가정하여 수정
        image_fpath = osp.join(self.root_dir, '..', image_info['img_path'])

        vertices, labels = [], []
        for word_info in image_info['words']:
            bbox = word_info['bbox']
            vertices.append(np.array([bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]))
            labels.append(1)
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

        # 필터링 처리
        vertices, labels = filter_vertices(
            vertices,
            labels,
            ignore_under=self.ignore_under_threshold,
            drop_under=self.drop_under_threshold
        )

        # 이미지 로드 및 크기 조정
        image = Image.open(image_fpath)
        image, vertices = resize_img(image, vertices, self.image_size)
        image, vertices = adjust_height(image, vertices)
        image, vertices = rotate_img(image, vertices)
        image, vertices = crop_img(image, vertices, labels, self.crop_size)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        # Albumentations color jitter와 normalize 적용
        funcs = []
        if self.color_jitter:
            funcs.append(A.ColorJitter())
        if self.normalize:
            funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = A.Compose(funcs)

        image = transform(image=image)['image']
        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        # 이미지와 ROI 마스크 크기 조정
        image = cv2.resize(image, (self.crop_size, self.crop_size))
        roi_mask = cv2.resize(roi_mask, (self.crop_size, self.crop_size))

        return image, word_bboxes, roi_mask

# class SceneTextDataset(Dataset):
#     def __init__(self, root_dir, 
#                  split='train', 
#                  image_size=2048, 
#                  crop_size=1024,
#                  ignore_under_threshold=10, 
#                  drop_under_threshold=1, 
#                  color_jitter=True,
#                  normalize=True, 
#                  apply_augments=True):
#         self.root_dir = root_dir
#         self.split = split
#         self.apply_augments = apply_augments
#         self.color_jitter = color_jitter
#         self.normalize = normalize
#         self.image_size = image_size
#         self.crop_size = crop_size
#         self.ignore_under_threshold = ignore_under_threshold
#         self.drop_under_threshold = drop_under_threshold
        
#         # Load annotations
#         with open(osp.join(root_dir, 'ufo_cord_data.json'), 'r', encoding='utf-8') as f:
#             self.anno = json.load(f)
#         self.image_fnames = sorted(self.anno['images'].keys())

#     def __len__(self):
#         return len(self.image_fnames)

#     def __getitem__(self, idx):
#         image_fname = self.image_fnames[idx]
#         image_info = self.anno['images'][image_fname]
#         image_fpath = osp.join(self.root_dir, '..', image_info['img_path'])
#         image = Image.open(image_fpath)
        
#         vertices, labels = [], []
#         for word_info in image_info['words']:
#             bbox = word_info['bbox']
#             vertices.append(np.array([bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]))
#             labels.append(1)
#         vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

#         # Resize and Crop
#         image, vertices = resize_img(image, vertices, self.image_size)
#         image, vertices = adjust_height(image, vertices)
#         image, vertices = rotate_img(image, vertices)
#         image, vertices = crop_img(image, vertices, labels, self.crop_size)

#         if self.apply_augments:
#             if np.random.rand() > 0.5:
#                 scale_factor = np.random.uniform(0.9, 1.1)
#                 image, vertices = scale_image(Image.fromarray(image), vertices, scale_factor)
#                 image = np.array(image)

#             if np.random.rand() > 0.5:
#                 x_offset, y_offset = np.random.randint(-10, 10), np.random.randint(-10, 10)
#                 image, vertices = translate_image(Image.fromarray(image), vertices, x_offset, y_offset)
#                 image = np.array(image)

#             if np.random.rand() > 0.5:
#                 image, vertices = perspective_transform(Image.fromarray(image), vertices)
#                 image = np.array(image)

#             if np.random.rand() > 0.5:
#                 brightness = np.random.uniform(0.8, 1.2)
#                 contrast = np.random.uniform(0.8, 1.2)
#                 saturation = np.random.uniform(0.8, 1.2)
#                 image = adjust_brightness_contrast_saturation(Image.fromarray(image), brightness, contrast, saturation)
#                 image = np.array(image)

#             if np.random.rand() > 0.5:
#                 image = add_gaussian_noise(image, std_range=(0.01, 0.05))
#                 image = np.array(image)

#             if np.random.rand() > 0.5:
#                 image = overlay_text(image)

#         # Color jitter and normalization
#         funcs = []
#         if self.color_jitter:
#             funcs.append(A.ColorJitter())
#         if self.normalize:
#             funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
#         transform = A.Compose(funcs)

#         if not isinstance(image, np.ndarray):
#             image = np.array(image)

#         image = transform(image=image)['image']
#         word_bboxes = np.reshape(vertices, (-1, 4, 2))
#         roi_mask = generate_roi_mask(image, vertices, labels)

#         image = cv2.resize(image, (self.crop_size, self.crop_size))
#         roi_mask = cv2.resize(roi_mask, (self.crop_size, self.crop_size))

#         return image, word_bboxes, roi_mask