import cv2
import pickle
import cvzone
import numpy as np
from ultralytics import YOLO

model = YOLO('C:\\Users\\chominkyung\\Desktop\\d0ng1nah\\d0ng1nah\\runs\\detect\\car3\\weights\\best.pt')
cap = cv2.VideoCapture('testvideos.mp4')

with open('.\\map\\230824_jeongseok', 'rb') as f:
    posList = pickle.load(f)
# 각주차 자리의 좌표들 픽셀


def is_point_inside_rectangle(px, py, x, y, width, height):
    if x <= px < x + width and y <= py < y + height:
        return True
    return False

def checkSpaces(centers):

    # parking_statuses = {}
    spaces = 0
    for idx, data in enumerate(posList, start=1):
        lt, lb, rb, rt = data['coordinates']
        closest_dist = float('inf')
        closest_center = None
        pos_center = (lt[0]+rb[0])/2, (lt[1]+rb[1])/2
        for center in centers:
            dist = np.sqrt(((center[0] - pos_center[0]))**2 + ((center[1] - pos_center[1])**2))

            if dist < closest_dist:
                closest_dist = dist
                closest_center = center

        if closest_center:
            if closest_dist < 5:  # Adjust the threshold as needed
                color = (0, 0, 200)
                thic = 2
                spaces += 1
                data['occupied'] = True
                # parking_statuses[pos_center] = 1  # occupied
            else:
                color = (0, 200, 0)
                thic = 2
                data['occupied'] = False
                # parking_statuses[pos_center] = 0  # free

        polygon_points = np.array(data['coordinates'], dtype=np.int32)
        cv2.polylines(img, [polygon_points], isClosed=True, color=color, thickness=thic)
    # print(posList)

    return posList


def make_map(parking_statuses):
    horizontal_count = -(-new_width // width)
    vertical_count = -(-new_height // height)

    map = np.full((vertical_count, horizontal_count), 3)

    entrypos = (3,0)
    map[entrypos[1]][entrypos[0]] = 2

    coordinates = list(parking_statuses.keys())
    for coordinate in coordinates:
        x = int(coordinate[0] * horizontal_count / new_width)
        y = int(coordinate[1] * vertical_count / new_height)

        if 0 <= x < horizontal_count and 0 <= y < vertical_count:
            map[y][x] = parking_statuses[coordinate]

    return map



def count_zeros_around_zero(map):
    zero_positions = np.argwhere(map == 0)
    counts = []
    distances = []

    for zero_pos in zero_positions:
        row, col = zero_pos
        
        count = 0 
          # 오른쪽 이웃 확인
        if col + 1 < map.shape[1] and map[row, col + 1] == 0: #map.shape[1]:열
            count += 1
        # 왼쪽 이웃 확인
        if col - 1 >= 0 and map[row, col - 1] == 0:
            count += 1
        # 위쪽 이웃 확인
        if row - 1 >= 0 and map[row - 1, col] == 0:
            count += 1
        # 아래쪽 이웃 확인
        if row + 1 < map.shape[0] and map[row + 1, col] == 0: #map.shape[0]:행
            count += 1
            
        counts.append(count)

        distance = abs(zero_pos[0] - 0) + abs(zero_pos[1] - 3)
        distances.append(distance)

    return zero_positions, counts, distances

###기존코드###
# def calculate_reward(distance, count):
#     # Extract the coordinates and the number of adjacent empty spots
#     # Define a reward for each empty adjacent spot
#     reward_per_empty_spot = 3

#     # The total reward is the negative distance (because we want to minimize it)
#     # plus the number of adjacent empty spots times the reward per spot
#     total_reward = -distance + count * reward_per_empty_spot

#     return total_reward



fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1200, 675))
# 영상 저장

def update_nearest_zero_to_two(map):
    entrypos = (3, 0)
    nearest_distance = float('inf')
    nearest_position = None

    for y in range(len(map)):
        for x in range(len(map[0])):
            if map[y][x] == 2:
                distance = abs(x - entrypos[0]) + abs(y - entrypos[1])
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_position = (x, y)

    if nearest_position:
        zero_positions, counts, distances = count_zeros_around_zero(map)
        nearest_zero_index = np.argmin(distances) # np.argmin 함수는 배열에서 최솟값의 인덱스 반환함
        nearest_zero_position = zero_positions[nearest_zero_index]
        
        # Update the nearest zero to 4
        map[nearest_zero_position[0]][nearest_zero_position[1]] = 4

def update_most_zeros_to_five(map):
    zero_positions, counts, distances = count_zeros_around_zero(map)

    if len(zero_positions) > 0:
        max_zero_index = np.argmax(counts)
        max_zero_position = zero_positions[max_zero_index]

        map[max_zero_position[0]][max_zero_position[1]] = 5


map = []

while True:
    success, img = cap.read()
    original_width = img.shape[1]
    original_height = img.shape[0]
    # 크기 (큼)

    new_width = 1500
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)

    img = cv2.resize(img, (new_width, new_height)) #작게 만듦

    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        exit()

    results = model.predict(img)
    # 바운딩박스 정보

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    centers = []

    for r in results:
        boxes = r.boxes #바운딩박스 정보

    for box in boxes:
        box_cpu = box.cpu().numpy()
        center = [(box_cpu.xyxy[0][0] + box_cpu.xyxy[0][2]) / 2, (box_cpu.xyxy[0][1] + box_cpu.xyxy[0][3]) / 2]
        center = int(center[0]), int(center[1])

        if center[0] < new_width and center[1] < new_height:
            centers.append(center)
            #centers: 바운딩박스 중앙점 모음
    
    
    zero_positions, counts, distances = count_zeros_around_zero(map)
    parking_statuses = checkSpaces(centers)
    map = make_map(parking_statuses)
    update_nearest_zero_to_two(map)
    update_most_zeros_to_five(map)
    print(map)    
    
    map_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(len(map)):
        for x in range(len(map[0])):
            if map[y][x] == 4:
                color = (0, 255, 255)  # Yellow
            elif map[y][x] == 5:
                color = (255, 0, 255)  # Purple
            else:
                color = (0, 0, 0)  # Black (background)

            cv2.rectangle(map_image, (x * width, y * height), ((x + 1) * width, (y + 1) * height), color, -1)

    blended_img = cv2.addWeighted(img, 0.7, map_image, 0.3, 0)  # Blend map image with the original image

   

    # 결과 출력
    ###기존 코드###
    # score = []
    # for i, zero_pos in enumerate(zero_positions):
    #     print(f"0 위치: {zero_pos}, 2와의 최소 거리: {distances[i]}, 주변 0 원소 개수: {counts[i]}")
    #     score.append(calculate_reward(distances[i], counts[i]))
        

    # indexed_arr = list(enumerate(score))

    # # Sort the indexed array in descending order based on the values
    # sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
    # # Extract and return the indices in the sorted order
    # sorted_indices = [index for index, _ in sorted_arr]
    # place = 1
    # for i in sorted_indices:
    #     print(place, ": 번째",zero_positions[i])
    #     place += 1
    
    
    ###수정코드###
    # score = []
    # for i, zero_pos in enumerate(zero_positions):
    #     print(f"0 위치: {zero_pos}, 주변 0 원소 개수: {counts[i]}")
    #     score.append(counts[i])
        

    # indexed_arr = list(enumerate(score))

    # # Sort the indexed array in descending order based on the values
    # sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
    # # Extract and return the indices in the sorted order
    # sorted_indices = [index for index, _ in sorted_arr]
    # place = 1
    # for i in sorted_indices:
    #     print(place, ": 번째",zero_positions[i])
    #     place += 1
    
    cv2.imshow("Image", blended_img)
    key = cv2.waitKey(1000)

    out.write(img)

    if key == ord('q'):
        break

out.release()