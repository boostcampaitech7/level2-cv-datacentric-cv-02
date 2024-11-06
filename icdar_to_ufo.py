import os
import json
import cv2
from glob import glob
from collections import defaultdict

# ICDAR 2015 데이터셋 경로 설정
image_dir = 'ICDAR_2015/ch4_training_images'  # 이미지 경로
label_dir = 'ICDAR_2015/ch4_training_localization_transcription_gt'  # 레이블 경로
output_dir = 'outputs'  # UFO 형식 저장 경로
ufo_data = dict(images=dict())

# UFO 형식으로 변환하는 함수
def convert_to_ufo(image_dir, label_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 레이블 파일 경로 가져오기
    label_files = glob(os.path.join(label_dir, '*.txt'))
    cnt = 0
    for label_file in label_files:
        # 레이블 파일 읽기
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(label_file[57:-4])
        image_name = label_file[57:-4] + '.jpg'
        ufo_data["images"].update({
            image_name: {
                "paragraphs": {},
                "words" : {}
                }
            }
        )
        idx = 0
        # 각 이미지에 대한 UFO 형식의 데이터 저장
        for line in lines:
            # 각 라인의 정보를 분리
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            
            # 이미지 파일 이름, 텍스트, 좌표 정보 추출
            text = parts[8]
            if text == '###':
                continue
            cnt += 1
            
            parts[0] = parts[0].replace('\ufeff', '')
            points = [
                [float(parts[0]), float(parts[1])],
                [float(parts[2]), float(parts[3])],
                [float(parts[4]), float(parts[5])],
                [float(parts[6]), float(parts[7])]
            ]

            # UFO 형식의 데이터 저장
            key = f"{idx + 1:04d}"

            ufo_data["images"][image_name]["words"][key] = {
                "transcription": text,
                "points": points
            }
            idx += 1
    


    # UFO 형식으로 저장할 JSON 파일 이름 설정
    output_file = os.path.join(output_dir, 'train.json')
        
    # UFO 형식으로 데이터 저장
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(ufo_data, out_f, ensure_ascii=False, indent=4)

    print(f"Converted {image_name} to UFO format.")
    print(cnt)

if __name__ == "__main__":
    convert_to_ufo(image_dir, label_dir, output_dir)
