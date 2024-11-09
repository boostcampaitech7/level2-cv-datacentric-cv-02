import os
import shutil

# 원본 CSV 파일이 있는 디렉토리 경로
csv_directory = '/data/ephemeral/home/level2-cv-datacentric-cv-02/code/predictions'

# 변환된 JSON 파일을 저장할 새 디렉토리 경로
json_directory = '/data/ephemeral/home/level2-cv-datacentric-cv-02/code/predictions/json_files'
os.makedirs(json_directory, exist_ok=True)  # 새 디렉토리 생성 (이미 존재할 경우 무시)

# 디렉토리 내의 모든 .csv 파일을 .json으로 이름 변경하여 새 디렉토리에 저장
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        base = os.path.splitext(filename)[0]
        new_filename = f"{base}.json"
        
        # 원본 파일 경로와 새 파일 경로 설정
        csv_path = os.path.join(csv_directory, filename)
        json_path = os.path.join(json_directory, new_filename)
        
        # JSON 파일로 복사
        shutil.copy2(csv_path, json_path)

print("All .csv files have been copied and renamed to .json in the new directory.")