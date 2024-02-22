import os

def count_total_lines_in_txt_files(directory_path):
    total_lines = 0

    # 특정 디렉토리 내의 모든 파일 가져오기
    file_list = [f for f in os.listdir(directory_path) if f.endswith('.txt')]

    # 각 파일에 대해 행 수 세기
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            total_lines += len(lines)

    return total_lines

# 특정 디렉토리 경로 설정
directory_path = 'C:/Users/VCL/Desktop/donghannah/datasets/train/labels'

# 디렉토리 내의 모든 txt 파일의 행 수를 더해서 출력
result = count_total_lines_in_txt_files(directory_path)
print(f'Total lines in all txt files: {result}')
