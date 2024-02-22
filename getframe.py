import cv2

# 동영상 파일 경로 설정
video_path = 'C:/Users/VCL/Desktop/drone_videos/DJI_0129.mp4'  # 동영상 파일 경로를 적절히 변경하세요

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 동영상의 한 프레임을 읽습니다.
ret, frame = cap.read()

if ret:
    # 프레임 저장할 파일명 설정
    output_file = 'capture_1.jpg'

    # 프레임을 jpg 형식으로 저장
    cv2.imwrite(output_file, frame)

    # 캡처 객체 해제
    cap.release()

    print(f'프레임을 {output_file}로 저장했습니다.')
else:
    print('동영상에서 프레임을 읽을 수 없습니다.')

