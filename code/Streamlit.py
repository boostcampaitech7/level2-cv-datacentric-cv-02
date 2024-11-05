import streamlit as st
import os
import json
import re
from PIL import Image, ImageDraw, ImageFont, ExifTags

# 이미지의 올바른 방향을 유지하는 함수
def open_image_correct_orientation(image_path):
    image = Image.open(image_path)
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
test_prediction_path = '/data/ephemeral/home/level2-cv-datacentric-cv-02/code/predictions/json_files'
languages = {
    'Chinese': 'chinese_receipt',
    'Japanese': 'japanese_receipt',
    'Thai': 'thai_receipt',
    'Vietnamese': 'vietnamese_receipt'
}

# 사이드바에서 언어 및 데이터셋 선택
st.sidebar.title("언어 및 데이터셋 선택")
selected_language = st.sidebar.selectbox("언어를 선택하세요", list(languages.keys()))
dataset_type = st.sidebar.radio("데이터셋을 선택하세요", ["Train", "Test"])

# JSON 주석 파일 로드
ufo_path = os.path.join(data_path, languages[selected_language], 'ufo')
train_json_path = os.path.join(ufo_path, 'train.json')

with open(train_json_path, 'r', encoding='utf-8') as f:
    train_annotations = json.load(f)

# EDA에 사용할 주석 데이터
annotations = train_annotations if dataset_type == "Train" else None

# Test 데이터의 경우 예측된 JSON 파일 복수 선택
selected_json_files = []
if dataset_type == "Test":
    st.sidebar.title("Test 예측 결과 선택")
    
    # JSON 파일 목록을 가져와 복수 선택을 허용합니다.
    try:
        json_files = [f for f in os.listdir(test_prediction_path) if f.endswith('.json')]
    except Exception as e:
        st.write(f"Error accessing directory: {e}")
    
    # JSON 파일 복수 선택
    if json_files:
        selected_json_files = st.sidebar.multiselect("JSON 파일을 선택하세요", json_files)

# 필터 옵션 설정
st.sidebar.title("필터 옵션")
min_chars = st.sidebar.number_input("최소 글자 수", min_value=0, value=0, step=1)
show_all_boxes = st.sidebar.checkbox("전체표시")  # 전체 표시 체크박스 추가
show_special_char = st.sidebar.checkbox("특수문자")
show_text = st.sidebar.checkbox("문자")
show_numbers = st.sidebar.checkbox("숫자")
show_null_empty = st.sidebar.checkbox("null 또는 빈 transcription 표시")

# 필터 리스트 초기화
filter_types = []
if show_special_char:
    filter_types.append("특수문자")
if show_text:
    filter_types.append("문자")
if show_numbers:
    filter_types.append("숫자")
if show_null_empty:
    filter_types.append("null 또는 빈 transcription 표시")

# 사용자 입력에 따른 BBOX ID 필터
bbox_id_filter = st.sidebar.text_input("원하는 BBOX 번호 입력", value="")

# 폰트 크기 조절
font_size = st.sidebar.slider("폰트 크기", min_value=10, max_value=40, value=15, step=1)
font = ImageFont.truetype(font_paths[selected_language], size=font_size)

# 필터 적용 함수
def apply_filters(transcription, bbox_id):
    """사용자 선택에 따른 transcription 및 BBOX 필터링"""
    
    

    # null 또는 빈 transcription 표시 필터가 선택된 경우 처리
    if 'null 또는 빈 transcription 표시' in filter_types:
        if transcription is None or transcription == "":
            return True  # null 또는 빈 문자열인 BBox 표시

    # 특수문자, 문자, 숫자 필터 중 하나라도 선택된 경우 해당 필터 조건에 따라 필터 적용
    if '특수문자' in filter_types and transcription:
        if re.search(r'[^\w\s\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f]', transcription):
            return True  # 특수문자가 포함된 BBox 표시
    if '문자' in filter_types and transcription:
        if re.search(r'[a-zA-Z\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f]', transcription):
            return True  # 문자가 포함된 BBox 표시
    if '숫자' in filter_types and transcription:
        if re.search(r'\d', transcription):
            return True  # 숫자가 포함된 BBox 표시

    # 필터 중 아무것도 선택되지 않은 경우 최소 글자 수와 BBox ID 필터 적용
    if not filter_types:
        # 최소 글자 수 필터
        if transcription is not None and len(transcription) < min_chars:
            return False  # 최소 글자 수를 충족하지 않는 경우 필터링

        # BBox ID 필터
        if bbox_id_filter and str(bbox_id_filter) != str(bbox_id):
            return False  # 입력한 BBox ID와 다른 경우 필터링
    # 전체표시가 선택된 경우 모든 BBox를 표시
    if show_all_boxes:
        return True
    # 조건에 맞는 BBox는 표시
    return False

# 이미지 목록 가져오기
image_dir = os.path.join(data_path, languages[selected_language], 'img', dataset_type.lower())
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('jpg', 'png'))])

# 이미지 인덱스 초기화 (세션 상태 사용)
if "image_index" not in st.session_state:
    st.session_state.image_index = 0

# 이미지 탐색
with st.sidebar.expander("이미지 탐색", expanded=True):
    image_number_input = st.text_input("이미지 인덱스 입력 (1 ~ 100)", "")
    if image_number_input.isdigit():
        image_index = int(image_number_input) - 1
        if 0 <= image_index < len(image_files):
            st.session_state.image_index = image_index
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
selected_image_path = os.path.join(image_dir, selected_image_file)

if dataset_type == "Train":
    # Train 데이터 시각화
    image = open_image_correct_orientation(selected_image_path)
    draw = ImageDraw.Draw(image)
    annotation_data = annotations['images'].get(selected_image_file, {}).get('words', {})

    for bbox_id, bbox_info in annotation_data.items():
        transcription = bbox_info['transcription']
        points = bbox_info.get('points', [])
        
        # 필터 적용
        if apply_filters(transcription, bbox_id):
            if len(points) == 4:
                points = [(int(x), int(y)) for x, y in points]
                draw.polygon(points, outline="red")
                text = f"{bbox_id}: {transcription}"
                text_position = (min(point[0] for point in points), min(point[1] for point in points) - 15)
                draw.text(text_position, text, fill="red", font=font)

    st.image(image, use_column_width=True, caption="Train 데이터 시각화")

else:
    # Test 데이터 시각화 - 각 JSON 파일마다 별도로 표시
    for selected_json in selected_json_files:
        image = open_image_correct_orientation(selected_image_path)
        draw = ImageDraw.Draw(image)
        
        # JSON 파일에서 예측 결과 불러오기
        json_path = os.path.join(test_prediction_path, selected_json)
        with open(json_path, 'r', encoding='utf-8') as f:
            prediction_data = json.load(f)

        if selected_image_file in prediction_data["images"]:
            test_annotations = prediction_data["images"][selected_image_file].get('words', {})
            
            for bbox_id, bbox_info in test_annotations.items():
                transcription = bbox_info.get('transcription', '')
                points = bbox_info['points']
                
                if apply_filters(transcription, bbox_id):
                    points = [(int(x), int(y)) for x, y in points]
                    draw.polygon(points, outline="blue")
                    text = f"{bbox_id}: {transcription}"
                    text_position = (points[0][0], points[0][1] - 15)
                    draw.text(text_position, text, fill="blue", font=font)
        
        st.image(image, use_column_width=True, caption=f"{selected_json} 적용 결과")