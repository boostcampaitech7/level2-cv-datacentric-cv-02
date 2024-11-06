import json
import os

# 기존의 CORD 데이터셋 JSON 경로와 이미지 경로
image_dir = '../data/cord_images'
json_dir = '../data/cord_json'
ufo_data = {"images": {}}
output_path = '../data/ufo/ufo_cord_data.json'

# JSON 파일을 순회하면서 UFO 형식으로 변환
for json_file_name in os.listdir(json_dir):
    image_id = json_file_name.split('.')[0]  # "image_1" 형태로 아이디 추출
    img_path = os.path.join(image_dir, f"{image_id}.jpg")
    
    with open(os.path.join(json_dir, json_file_name), 'r', encoding='utf-8') as f:
        cord_data = json.load(f)
    
    # 'ufo_data'에 새로운 이미지 정보 추가
    ufo_data["images"][image_id] = {
        "img_path": img_path,
        "words": []
    }

    # 텍스트와 바운딩 박스 정보 추가
    for line in cord_data.get('valid_line', []):
        for word_info in line.get('words', []):
            transcription = word_info.get('text', "")
            
            # 'quad' 정보로부터 x_min, y_min, x_max, y_max 추출
            quad = word_info.get('quad', {})
            x_min = min(quad['x1'], quad['x2'], quad['x3'], quad['x4'])
            y_min = min(quad['y1'], quad['y2'], quad['y3'], quad['y4'])
            x_max = max(quad['x1'], quad['x2'], quad['x3'], quad['x4'])
            y_max = max(quad['y1'], quad['y2'], quad['y3'], quad['y4'])

            # UFO 형식에 맞게 단어 추가
            ufo_data["images"][image_id]["words"].append({
                "transcription": transcription,
                "bbox": [x_min, y_min, x_max, y_max]
            })

# UFO 형식 JSON 파일로 저장
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(ufo_data, f, ensure_ascii=False, indent=4)