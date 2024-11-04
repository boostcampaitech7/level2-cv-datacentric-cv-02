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
filter_type = st.sidebar.radio("Transcription 필터 유형", ['None', '특수문자', '문자', '숫자'])

# null 또는 빈 transcription 표시 옵션을 따로 분리
null_filter = st.sidebar.checkbox("null 또는 빈 transcription 표시")

# BBOX 번호 필터 입력
bbox_id_filter = st.sidebar.text_input("원하는 BBOX 번호 입력", value="")

# 폰트 크기 조절 슬라이더 (BBOX 번호 입력 아래에 위치)
font_size = st.sidebar.slider("폰트 크기", min_value=10, max_value=40, value=15, step=1)
font = ImageFont.truetype(font_paths[selected_language], size=font_size)

# 이미지 목록 가져오기
image_dir = os.path.join(data_path, languages[selected_language], 'img', 'train')
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('jpg', 'png'))])

# 이미지 번호 매칭을 위한 딕셔너리 생성 (appen_ 또는 appen2_ 뒤의 숫자를 추출)
image_number_map = {}
for i, f in enumerate(image_files):
    match = re.search(r'appen(?:2_)?_(\d+)', f)  # appen_ 또는 appen2_ 뒤의 숫자를 추출
    if match:
        image_number_map[match.group(1)] = i  # 숫자를 키로 하여 인덱스 값을 저장

# 이미지 인덱스 초기화 (세션 상태 사용)
if "image_index" not in st.session_state:
    st.session_state.image_index = 0

# 이미지 탐색 (화면의 오른쪽 상단에 위치)
with st.sidebar.expander("이미지 탐색", expanded=True):
    # 이미지 번호 입력
    image_number_input = st.text_input("이미지 번호 입력(버튼 사용시 입력값 삭제 바람)", "")
    
    # 이미지 번호로 검색
    if image_number_input and image_number_input in image_number_map:
        st.session_state.image_index = image_number_map[image_number_input]

    selected_image_file = image_files[st.session_state.image_index]

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("이전 이미지"):
            st.session_state.image_index = (st.session_state.image_index - 1) % len(image_files)
            selected_image_file = image_files[st.session_state.image_index]
    with col2:
        st.write(f"이미지 {st.session_state.image_index + 1} / {len(image_files)}")
    with col3:
        if st.button("다음 이미지"):
            st.session_state.image_index = (st.session_state.image_index + 1) % len(image_files)
            selected_image_file = image_files[st.session_state.image_index]

# 선택한 이미지 로드
selected_image_path = os.path.join(image_dir, selected_image_file)
image_id = selected_image_file.split('_')[3]  # appen2_ 또는 appen_ 뒤의 번호 추출
st.write(f"### 이미지: {selected_image_file}")

# BBOX 주석과 함께 이미지 표시
image = Image.open(selected_image_path)
draw = ImageDraw.Draw(image)
annotation_data = annotations['images'].get(selected_image_file, {}).get('words', {})

# BBOX 카운트 변수 초기화
null_count = 0
empty_count = 0
general_count = 0
total_count = 0

def apply_filters(transcription, bbox_id):
    """사용자 선택에 따른 transcription 및 BBOX 필터링"""
    # null 또는 빈 transcription 필터
    if null_filter:
        if transcription is None:
            return "null"
        elif transcription == "":
            return "empty"
        else:
            return False  # null 또는 빈 값이 아니면 필터링
    
    # 일반 필터 적용
    # BBOX ID 필터
    if bbox_id_filter and bbox_id_filter != bbox_id:
        return False
    
    # 글자 수 필터
    if transcription is None or len(transcription) < min_chars:
        return False
    
    # 필터 유형
    if filter_type == '특수문자':
        if not re.search(r'[^\w\s\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f\u0100-\u017f]', transcription):
            return False
    elif filter_type == '문자':
        if not re.search(r'[a-zA-Z\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f\u0100-\u017f]', transcription):
            return False
    elif filter_type == '숫자':
        if not re.search(r'\d', transcription):
            return False
    
    return True

# BBOX 그리기 및 주석 표시
for bbox_id, bbox_info in annotation_data.items():
    transcription = bbox_info['transcription']
    points = bbox_info.get('points', [])

    # BBOX 개수 카운트
    total_count += 1  # 총 BBOX 카운트

    # 좌표 형식 확인 (디버깅용)
    if not isinstance(points, list) or not all(isinstance(point, list) and len(point) == 2 for point in points):
        st.write(f"잘못된 좌표 형식: {points} (bbox_id: {bbox_id})")
        continue  # 좌표 형식이 올바르지 않으면 해당 BBOX를 건너뜀

    # 필터 적용
    filter_result = apply_filters(transcription, bbox_id)
    if filter_result:
        # 좌표를 정수로 변환하여 안전하게 사용
        points = [(int(x), int(y)) for x, y in points]
        
        # BBOX 그리기 (null과 빈 값에 대해 다른 색상 적용)
        if filter_result == "null":
            box_color = "#006400"  # null 값인 경우 다크 그린
            null_count += 1
        elif filter_result == "empty":
            box_color = "blue"  # 빈 문자열인 경우 진한 파란색
            empty_count += 1
        else:
            box_color = "red"  # 일반 필터에 해당하는 경우 빨간색
            general_count += 1

        # BBOX 그리기
        if len(points) == 4:
            draw.polygon(points, outline=box_color)
            
            # 텍스트 경계 상자를 가져와서 크기를 계산
            text = f"{bbox_id}: {transcription}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 텍스트를 BBOX 바깥에 표시하기 위해 위치 조정
            text_position = (min(point[0] for point in points), min(point[1] for point in points) - text_height - 5)

            # 텍스트 추가
            draw.text(text_position, text, fill=box_color, font=font)

# BBOX 카운트 결과 표시
st.write("### BBOX 카운트 - null 또는 빈 transcription 표시 체크해야 null, empty BBOX 표시됨")
st.write(f"**총 BBOX:** {total_count}")
st.write(f"**일반 BBOX (빨간색):** {general_count}")
st.write(f"**Null BBOX (다크 그린):** {null_count}")
st.write(f"**Empty BBOX (파란색):** {empty_count}")

# 필터링된 이미지와 주석 표시
st.image(image, use_column_width=True)