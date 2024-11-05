import streamlit as st
import os
import json
import re
from PIL import Image, ImageDraw, ImageFont, ExifTags


def open_image_correct_orientation(image_path):
    # 이미지 열기
    image = Image.open(image_path)

    # EXIF 데이터에서 Orientation 태그 확인
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None and orientation in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # EXIF 데이터가 없거나 처리할 수 없는 경우
        pass

    return image

# 다국어 지원을 위한 폰트 파일 경로 설정
font_paths = {
    'Chinese': "../Streamlit/font/NotoSansTC-VariableFont_wght.ttf",
    'Japanese': "../Streamlit/font/NotoSansJP-VariableFont_wght.ttf",
    'Thai': "../Streamlit/font/NotoSansThai-VariableFont_wdth,wght.ttf",
    'Vietnamese': "../Streamlit/font/NotoSans_Condensed-Regular.ttf"
}

# 기본 경로 설정
data_path = 'data/'
languages = {
    'Chinese': 'chinese_receipt',
    'Japanese': 'japanese_receipt',
    'Thai': 'thai_receipt',
    'Vietnamese': 'vietnamese_receipt'
}



# 사이드바에서 언어 선택
st.sidebar.title("언어 선택")
selected_language = st.sidebar.selectbox("언어를 선택하세요", list(languages.keys()))

# JSON 주석 파일 로드 (항상 Train 데이터만 로드)
ufo_path = os.path.join(data_path, languages[selected_language], 'ufo')
train_json_path = os.path.join(ufo_path, 'train.json')

with open(train_json_path, 'r', encoding='utf-8') as f:
    train_annotations = json.load(f)

# Train 데이터만 사용하도록 설정
annotations = train_annotations

# 필터 옵션 설정
st.sidebar.title("필터 옵션")
min_chars = st.sidebar.number_input("최소 글자 수", min_value=0, value=0, step=1)

# 체크박스로 필터 유형 설정
show_all = st.sidebar.checkbox("전체표시")
show_special_char = st.sidebar.checkbox("특수문자")
show_text = st.sidebar.checkbox("문자")
show_numbers = st.sidebar.checkbox("숫자")
show_null_empty = st.sidebar.checkbox("null 또는 빈 transcription 표시")

# 선택된 필터 옵션에 따라 적용할 필터 리스트 생성
filter_types = []
if show_all:
    filter_types.append("전체표시")
if show_special_char:
    filter_types.append("특수문자")
if show_text:
    filter_types.append("문자")
if show_numbers:
    filter_types.append("숫자")
if show_null_empty:
    filter_types.append("null 또는 빈 transcription 표시")

# BBOX 번호 필터 입력
bbox_id_filter = st.sidebar.text_input("원하는 BBOX 번호 입력", value="")

# 폰트 크기 조절 슬라이더 (BBOX 번호 입력 아래에 위치)
font_size = st.sidebar.slider("폰트 크기", min_value=10, max_value=40, value=15, step=1)
font = ImageFont.truetype(font_paths[selected_language], size=font_size)

# 이미지 목록 가져오기
image_dir = os.path.join(data_path, languages[selected_language], 'img', 'train')
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('jpg', 'png'))])

# 이미지 인덱스 초기화 (세션 상태 사용)
if "image_index" not in st.session_state:
    st.session_state.image_index = 0

# 이미지 탐색
with st.sidebar.expander("이미지 탐색", expanded=True):
    # 이미지 번호 입력 (1부터 시작하는 인덱스로 이동)
    image_number_input = st.text_input("이미지 인덱스 입력 (1 ~ 100)", "")
    
    # 입력한 이미지 인덱스를 사용하여 이동
    if image_number_input.isdigit():  # 숫자 확인
        image_index = int(image_number_input) - 1  # 1부터 시작하므로 -1
        if 0 <= image_index < len(image_files):  # 인덱스 범위 내 확인
            st.session_state.image_index = image_index  # 인덱스 갱신

    # 이전/다음 버튼을 사용한 탐색
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("이전 이미지"):
            st.session_state.image_index = (st.session_state.image_index - 1) % len(image_files)
    with col2:
        st.write(f"이미지 {st.session_state.image_index + 1} / {len(image_files)}")
    with col3:
        if st.button("다음 이미지"):
            st.session_state.image_index = (st.session_state.image_index + 1) % len(image_files)

# 선택한 이미지 파일
selected_image_file = image_files[st.session_state.image_index]


# 선택한 이미지 로드
selected_image_path = os.path.join(image_dir, selected_image_file)
st.write(f"### 이미지: {selected_image_file}")

# BBOX 주석과 함께 이미지 표시
image = open_image_correct_orientation(selected_image_path)
draw = ImageDraw.Draw(image)
annotation_data = annotations['images'].get(selected_image_file, {}).get('words', {})

# BBOX 카운트 변수 초기화
null_count = 0
empty_count = 0
general_count = 0
total_count = 0

def apply_filters(transcription, bbox_id):
    """사용자 선택에 따른 transcription 및 BBOX 필터링"""
    # 전체 표시 선택 시 필터링을 건너뜀
    if '전체표시' in filter_types:
        return True

    # null 또는 빈 transcription 필터
    if 'null 또는 빈 transcription 표시' in filter_types:
        if transcription is None:
            return "null"
        elif transcription == "":
            return "empty"
    
    # 글자 수 필터
    if transcription is None or len(transcription) < min_chars:
        return False

    # 필터 유형
    if '특수문자' in filter_types and re.search(r'[^\w\s\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f\u0100-\u017f]', transcription):
        return True
    if '문자' in filter_types and re.search(r'[a-zA-Z\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f\u0100-\u017f]', transcription):
        return True
    if '숫자' in filter_types and re.search(r'\d', transcription):
        return True

    # BBOX ID 필터 - 문자열로 변환하여 비교
    if bbox_id_filter and str(bbox_id_filter) != str(bbox_id):
        return False

    return False

# BBOX 그리기 및 주석 표시
for bbox_id, bbox_info in annotation_data.items():
    transcription = bbox_info['transcription']
    points = bbox_info.get('points', [])

    total_count += 1

    if not isinstance(points, list) or not all(isinstance(point, list) and len(point) == 2 for point in points):
        st.write(f"잘못된 좌표 형식: {points} (bbox_id: {bbox_id})")
        continue

    filter_result = apply_filters(transcription, bbox_id)
    if filter_result:
        points = [(int(x), int(y)) for x, y in points]
        if filter_result == "null":
            box_color = "#006400"
            null_count += 1
        elif filter_result == "empty":
            box_color = "blue"
            empty_count += 1
        else:
            box_color = "red"
            general_count += 1

        if len(points) == 4:
            draw.polygon(points, outline=box_color)
            text = f"{bbox_id}: {transcription}"
            text_position = (min(point[0] for point in points), min(point[1] for point in points) - 15)
            draw.text(text_position, text, fill=box_color, font=font)

# BBOX 카운트 결과 표시
st.write("### BBOX 카운트")
st.write(f"**총 BBOX:** {total_count}")
st.write(f"**일반 BBOX (빨간색):** {general_count}")
st.write(f"**Null BBOX (다크 그린):** {null_count}")
st.write(f"**Empty BBOX (파란색):** {empty_count}")

st.image(image, use_column_width=True)

# 초기화 - 선택된 이미지와 annotation 목록을 위한 상태 저장
if "selected_images" not in st.session_state:
    st.session_state.selected_images = []

# 이미지 선택 버튼 추가
if st.button("이 이미지를 선택 목록에 추가"):
    selected_annotation = {
        "image": selected_image_file,
        "annotations": annotation_data
    }
    st.session_state.selected_images.append(selected_annotation)
    st.success("이미지가 선택 목록에 추가되었습니다.")

# 선택된 이미지와 annotation 목록 표시
st.sidebar.title("선택한 이미지와 Annotation")
if st.session_state.selected_images:
    indices_to_delete = []  # 삭제할 인덱스를 저장할 리스트
    
    for i, selected_item in enumerate(st.session_state.selected_images):
        st.sidebar.write(f"### 이미지 {i + 1}")
        st.sidebar.write(f"파일명: {selected_item['image']}")

        selected_image_path = os.path.join(image_dir, selected_item['image'])
        selected_image = open_image_correct_orientation(selected_image_path).convert("RGB")
        draw = ImageDraw.Draw(selected_image)

        # 주석 추가하여 이미지에 표시
        for bbox_id, bbox_info in selected_item['annotations'].items():
            transcription = bbox_info['transcription']
            points = bbox_info.get('points', [])
            if len(points) == 4:
                points = [(int(x), int(y)) for x, y in points]
                draw.polygon(points, outline="red")
                text = f"{bbox_id}: {transcription}"
                text_position = (min(point[0] for point in points), min(point[1] for point in points) - 15)
                draw.text(text_position, text, fill="red", font=font)
        
        st.sidebar.image(selected_image, use_column_width=True)

        # 삭제 버튼
        if st.sidebar.button(f"삭제 - 이미지 {i + 1}", key=f"delete_{i}"):
            indices_to_delete.append(i)  # 삭제할 인덱스를 리스트에 추가

    # 삭제할 인덱스들을 순서대로 제거하여 `st.session_state` 업데이트
    for index in sorted(indices_to_delete, reverse=True):
        del st.session_state.selected_images[index]

else:
    st.sidebar.write("선택된 이미지가 없습니다.")