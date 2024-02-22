import numpy as np
import heapq
import matplotlib.pyplot as plt
import pickle




def heuristic(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def astar(map, start, goal):
    rows, cols = len(map), len(map[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    open_list = []
    closed_list = set()
    parent_map = {}  # 부모 노드를 저장하는 딕셔너리

    g = 0
    h = heuristic(start, goal)
    f = g + h
    heapq.heappush(open_list, (f, g, start))

    while open_list:
        f, g, current = heapq.heappop(open_list)
        closed_list.add(current)

        if current == goal:
            # 최단 경로 탐색이 완료되면 부모 노드를 따라가며 최단 경로를 생성
            path = []
            while current in parent_map:
                path.append(current)
                current = parent_map[current]
            path.append(start)
            path.reverse()
            return path

        for direction in directions:
            dx, dy = direction
            x, y = current[0] + dx, current[1] + dy

            if 0 <= x < rows and 0 <= y < cols:
                if (x, y) not in closed_list and map[x][y] not in [1]:
                    new_g = g + 1
                    new_h = heuristic((x, y), goal)
                    new_f = new_g + new_h
                    heapq.heappush(open_list, (new_f, new_g, (x, y)))
                    parent_map[(x, y)] = current  # 부모 노드 저장

    return None  # 도달할 수 없는 경우


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

        distance = abs(zero_pos[0] - 12) + abs(zero_pos[1] - 0)
        distances.append(distance)

    return zero_positions, counts, distances


def calculate_reward(distance, count):
    # Extract the coordinates and the number of adjacent empty spots
    # Define a reward for each empty adjacent spot
    reward_per_empty_spot = 3

    # The total reward is the negative distance (because we want to minimize it)
    # plus the number of adjacent empty spots times the reward per spot
    total_reward = -distance + count * reward_per_empty_spot

    return total_reward


map = np.array([[3,2,3,3,3,3,3,3,3,3],
                [3,1,1,3,1,1,3,1,1,3],
                [3,1,1,3,1,1,3,0,0,3],
                [3,1,1,3,1,0,3,1,0,3],
                [3,1,1,3,0,1,3,0,1,3],
                [3,1,1,3,1,1,3,1,1,3],
                [3,1,1,2,1,1,3,1,1,3]])


zero_positions, counts, distances = count_zeros_around_zero(map)

# 결과 출력
score = []
for i, zero_pos in enumerate(zero_positions):
    print(f"0 위치: {zero_pos}, 2와의 최소 거리: {distances[i]}, 주변 0 원소 개수: {counts[i]}")
    score.append(calculate_reward(distances[i], counts[i]))
    

indexed_arr = list(enumerate(score))

# Sort the indexed array in descending order based on the values
sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)

# Extract and return the indices in the sorted order
sorted_indices = [index for index, _ in sorted_arr]
place = 1
for i in sorted_indices:
    print(place, ": 번째",zero_positions[i])
    place += 1
    
 
zero_positions_tuples = [(pos[0], pos[1]) for pos in zero_positions]   

start_point = (13, 3)
goal_point = zero_positions_tuples[sorted_indices[0]]
print(zero_positions_tuples[sorted_indices[0]])

path = astar(map, start_point, goal_point)
if path is not None:
    print("최단 경로:")
    for point in path:
        print(point)
else:
    print("목표 지점에 도달할 수 없습니다.")

if path is not None:
    path_x, path_y = zip(*path)
    plt.imshow(map, cmap='gray')
    plt.plot(path_y, path_x, color='red', marker='o', markersize=10)
    plt.plot(start_point[1], start_point[0], color='green', marker='o', markersize=10)
    plt.plot(goal_point[1], goal_point[0], color='blue', marker='o', markersize=10)
    plt.show()