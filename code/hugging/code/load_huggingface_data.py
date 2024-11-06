from PIL import Image
from io import BytesIO
import json
import pandas as pd
import os

# 첫 번째 파일 로드
df1 = pd.read_parquet('../data/train-00000-of-00004-b4aaeceff1d90ecb.parquet')
# 두 번째 파일 로드
df2 = pd.read_parquet('../data/train-00001-of-00004-7dbbe248962764c5.parquet')

image_dir = '../data/cord_images'
json_dir = '../data/cord_json'
os.makedirs(image_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

# 첫 번째 파일의 이미지와 JSON 저장
for index, row in df1.iterrows():
    image_data = row['image']['bytes']
    image = Image.open(BytesIO(image_data))
    image.save(f'{image_dir}/image_{index + 1}.jpg')

    ground_truth_str = row['ground_truth']  
    ground_truth_dict = json.loads(ground_truth_str)
    with open(f'{json_dir}/image_{index + 1}.json', 'w', encoding='utf-8') as json_file:
        json.dump(ground_truth_dict, json_file, ensure_ascii=False, indent=4)

# 두 번째 파일의 이미지와 JSON 저장 (index를 201부터 시작)
start_index = len(df1) + 1
for index, row in df2.iterrows():
    image_data = row['image']['bytes']
    image = Image.open(BytesIO(image_data))
    image.save(f'{image_dir}/image_{start_index + index}.jpg')

    ground_truth_str = row['ground_truth']  
    ground_truth_dict = json.loads(ground_truth_str)
    with open(f'{json_dir}/image_{start_index + index}.json', 'w', encoding='utf-8') as json_file:
        json.dump(ground_truth_dict, json_file, ensure_ascii=False, indent=4)