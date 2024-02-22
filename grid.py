import cv2
import numpy as np
import pickle
import json

cap = cv2.VideoCapture('.\\testvideos\\230824_jeongseok.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

new_width = 1500
aspect_ratio = frame_height / frame_width
new_height = int(new_width * aspect_ratio)

width, height = 90, 200  # parking space size
with open('.\\map\\230824_jeongseok', 'rb') as f:
    posList = list(pickle.load(f))
# Coordinates of each parking spot in pixels (top left coordinate)

CELL_SIZE = 50
WINDOW_NAME = "Grid Editor"

GRID_ROWS = new_height // CELL_SIZE
GRID_COLS = new_width // CELL_SIZE
# Determine map size
print(GRID_ROWS, GRID_COLS)

grid = np.full((GRID_ROWS, GRID_COLS), 4)

specific_cell = (1417 // CELL_SIZE, 495 // CELL_SIZE)
grid[specific_cell[1], specific_cell[0]] = 2


with open('.\\0\\src\\230824_jeongseok.json', 'r') as file:
    parking_data = json.load(file)

# 주차 공간 상태 업데이트
for space in parking_data["parking_status"]:
    coordinates = space["coordinates"]
    # 주차 공간의 중심 좌표 계산
    center_x = int((coordinates[0][0] + coordinates[2][0]) / 2) // CELL_SIZE
    center_y = int((coordinates[0][1] + coordinates[2][1]) / 2) // CELL_SIZE

    # 주차 공간 상태에 따라 그리드 업데이트
    if space["occupied"]:
        grid[center_y, center_x] = 1
    else:
        grid[center_y, center_x] = 0

try:
    with open("grid_state.pkl", "rb") as f:
        saved_grid = pickle.load(f)
        if saved_grid.shape == grid.shape:
            grid = saved_grid
except FileNotFoundError:
    pass
    # Initialize the grid
    

for pos in posList:
    try:
        pos = (pos[0] // CELL_SIZE, pos[1] // CELL_SIZE)
        height_ = height // CELL_SIZE
        width_ = width // CELL_SIZE
        # print(pos[0], pos[1], height_, width_)
        grid[pos[1]: pos[1] + height_+1, pos[0]: pos[0] + width_+1] = 1
        # print(grid)
    except:
        pass
# The corresponding parking spot is 1 in advance
#print(grid)
# Load saved grid state if available

# can be loaded

def draw_grid(frame):
    for row in range(0, frame.shape[0], CELL_SIZE):
        cv2.line(frame, (0, row), (frame.shape[1], row), (255, 0, 0), 1)
    for col in range(0, frame.shape[1], CELL_SIZE):
        cv2.line(frame, (col, 0), (col, frame.shape[0]), (255, 0, 0), 1)
# draw grid lines
def mouse_callback(event, x, y, flags, param):
    global grid, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        col = x // CELL_SIZE
        row = y // CELL_SIZE

        if grid[row, col] == 4:
            grid[row, col] = 3
            cell = frame[row * CELL_SIZE : (row + 1) * CELL_SIZE, col * CELL_SIZE : (col + 1) * CELL_SIZE]
            cell[:, :] = [255, 255, 255]  # 흰색으로 업데이트
        elif grid[row, col] == 3:
            grid[row, col] = 4
            cell = frame[row * CELL_SIZE : (row + 1) * CELL_SIZE, col * CELL_SIZE : (col + 1) * CELL_SIZE]
            cell[:, :] = [0, 0, 0]  # 검은색으로 업데이트 (또는 다른 색상)
        print(grid)
        # 변경된 그리드를 이용해 프레임을 다시 그리고 표시
        frame_resized = cv2.resize(frame, (new_width, new_height))
        blended_frame = cv2.addWeighted(frame_resized, 0.5, img, 0.5, 0)
        cv2.imshow(WINDOW_NAME, blended_frame)

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

# ... 이전 코드 ...

# 첫 프레임 읽기
success, img = cap.read()
if not success:
    print("Failed to read the video frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# 비디오 프레임 크기 조정
img = cv2.resize(img, (new_width, new_height))
cv2.imshow('img', img)

# 3채널 색상 이미지로 그리드 프레임 초기화
frame = np.zeros((GRID_ROWS * CELL_SIZE, GRID_COLS * CELL_SIZE, 3), dtype=np.uint8)
draw_grid(frame)

# 그리드 상태에 따라 색상 업데이트
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        if grid[row, col] == 1:  # occupied
            frame[row * CELL_SIZE:(row + 1) * CELL_SIZE, col * CELL_SIZE:(col + 1) * CELL_SIZE] = [255, 255, 0]  # Red
        elif grid[row, col] == 2:  # specific cell
            frame[row * CELL_SIZE:(row + 1) * CELL_SIZE, col * CELL_SIZE:(col + 1) * CELL_SIZE] = [0, 255, 0]  # Green
        elif grid[row, col] == 3:  # default
            frame[row * CELL_SIZE:(row + 1) * CELL_SIZE, col * CELL_SIZE:(col + 1) * CELL_SIZE] = [255, 255, 255]  # White
        elif grid[row, col] == 0:  # unoccupied
            frame[row * CELL_SIZE:(row + 1) * CELL_SIZE, col * CELL_SIZE:(col + 1) * CELL_SIZE] = [255, 255, 0]  # Yellow

# 그리드 프레임 크기 조정 및 비디오 프레임과 합성
frame_resized = cv2.resize(frame, (new_width, new_height))
blended_frame = cv2.addWeighted(frame_resized, 0.5, img, 0.5, 0)
cv2.imshow(WINDOW_NAME, blended_frame)

# 사용자가 'q'를 누를 때까지 대기
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
