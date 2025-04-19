import numpy as np
import cv2
import pandas as pd
import time
import pyautogui
import datetime
from tqdm import tqdm
from gpt import get_decription_csv, get_decription_csv_sensenova, merge_csv_files
from solver import get_categority_csv
from common import csv_to_matrix
pyautogui.FAILSAFE = True  # 启用安全措施

# 定义上下左右四个方向
directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
is_print_matrix = True


# 将CSV文件转换为矩阵



# 获取矩阵中每个方块的图像
def get_matrix(image_path, row, column, crop_width, crop_height, generate_image=False):
    img1 = cv2.imread(image_path)
    img1 = cv2.resize(img1, (450, 630), interpolation=cv2.INTER_AREA)
    height = img1.shape[0]
    width = img1.shape[1]
    dx = height / row
    dy = width / column
    image_matrix = [[] for _ in range(row)]

    for i in range(row):
        for j in range(column):
            x = int(dx * i)
            y = int(dy * j)
            next_x = int(x + dx)
            next_y = int(y + dy)
            clip = round_clip(img1[x:next_x, y:next_y], crop_width, crop_height)
            image_matrix[i].append(clip)
    return image_matrix

# 裁剪图像
def round_clip(img, crop_width, crop_height):
    h, w, _ = img.shape
    return img[crop_height:h - crop_height, crop_width:w - crop_width]

# 获取每个类别的图像
def get_category_images(matrix, image_matrix):
    category_images = {}
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if matrix[y, x] not in category_images:
                category_images[matrix[y, x]] = image_matrix[y][x]
    return category_images

# 可视化矩阵
def visualize_matrix(matrix, category_images, cur_block=None, new_block=None, same_block=None):
    micro_h = 39
    micro_w = 39
    crop_width = 3
    crop_height = 3
    target_h = 14 * (micro_h + crop_height * 2)
    target_w = 10 * (micro_w + crop_width * 2)
    new_micro_h = micro_h + crop_height * 2
    new_micro_w = micro_w + crop_width * 2
    new_image = np.ones((target_h, target_w, 3), dtype=np.uint8) * np.array([0, 51, 153], dtype=np.uint8)

    for y, row in enumerate(matrix):
        for x, category_index in enumerate(row):
            if category_index != -1:
                new_image[y * new_micro_h + crop_height:(y + 1) * new_micro_h - crop_height,
                x * new_micro_w + crop_width:(x + 1) * new_micro_w - crop_width, :] = category_images[category_index]
    if cur_block is not None:
        cv2.rectangle(new_image, (new_block[1] * new_micro_w, new_block[0] * new_micro_h),
                      ((new_block[1] + 1) * new_micro_w, (new_block[0] + 1) * new_micro_h), (0, 112, 224), 3)
        cv2.rectangle(new_image, (same_block[1] * new_micro_w, same_block[0] * new_micro_h),
                      ((same_block[1] + 1) * new_micro_w, (same_block[0] + 1) * new_micro_h), (224, 0, 224), 1, lineType=cv2.LINE_AA)
        cv2.rectangle(new_image, (cur_block[1] * new_micro_w, cur_block[0] * new_micro_h),
                      ((cur_block[1] + 1) * new_micro_w, (cur_block[0] + 1) * new_micro_h), (0, 0, 224), 3)
        cv2.putText(new_image, "cur", (cur_block[1] * new_micro_w, cur_block[0] * new_micro_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 224), 2)
    return new_image

# 打印分割线
def print_split_line(width, start="", end=""):
    print(start, end="")
    for i in range(width):
        print("——", end="\t")
    print(end)

# 打印序列号
def print_serial_number(width, start, end=""):
    print(start, end="\t\t|\t")
    for j in range(width):
        print(j, end="\t")
    print(end)

# 打印矩阵
def print_matrix(matrix, current_steps, point1=None, point2=None):
    print()
    if point2 is not None and point2 is not None:
        print("Step: %s, Source Block: %s, Target Block: %s" %
              (current_steps, point1, point2))
    else:
        print("Step: %s " % current_steps)
    if not is_print_matrix:
        return
    print_split_line(len(matrix[0]) + 2, start="-  ", end="-")
    print_serial_number(len(matrix[0]), start="|", end="|")
    print_split_line(len(matrix[0]) + 2, start="|  ", end="|")
    for i in range(len(matrix)):
        print("| ", i, end="\t|\t")
        for j in range(len(matrix[0])):
            print(matrix[i][j], end="\t")
        print("|")
    print_split_line(len(matrix[0]) + 2, start="-  ", end="-")

# 找到终点
def find_end_point(matrix, x, y, direction):
    while True:
        dx = x + direction[0]
        dy = y + direction[1]
        if 0 <= dx < len(matrix) and 0 <= dy < len(matrix[0]):
            if matrix[dx][dy] == -1:
                end_point = [x, y]
                break
            else:
                x = dx
                y = dy
        else:
            end_point = [x, y]
            break
    return end_point

# 找到最近的远程点
def find_nearest_remote_point(matrix, x, y, direction):
    while True:
        dx = x + direction[0]
        dy = y + direction[1]
        if 0 <= dx < len(matrix) and 0 <= dy < len(matrix[0]):
            if matrix[dx][dy] == -1:
                x = dx
                y = dy
            else:
                nearest_remote_point = [x, y]
                break
        else:
            nearest_remote_point = [x, y]
            break
    return nearest_remote_point

# 获取移动距离
def get_direction_distance(x, y, matrix, direction):
    end_point = find_end_point(matrix, x, y, direction)
    nearest_remote_point = find_nearest_remote_point(matrix, end_point[0], end_point[1], direction)
    return [nearest_remote_point[0] - end_point[0], nearest_remote_point[1] - end_point[1]]

# 判断位置是否有效
def is_valid(matrix, dx, dy):
    return 0 <= dx < len(matrix) and 0 <= dy < len(matrix[0])

# 找到相同的方块
def find_same_block(matrix, x, y, nx, ny, block):
    for direction in directions:
        dx = nx + direction[0]
        dy = ny + direction[1]

        while True:
            if dx == x and dy == y:
                break
            if is_valid(matrix, dx, dy):
                if matrix[dx][dy] == -1:
                    dx += direction[0]
                    dy += direction[1]
                elif matrix[dx][dy] == block:
                    return [dx, dy]
                else:
                    break
            else:
                break
    return None


# 移动方块
def move_block(matrix, x, y, dx, dy, direction):
    if dx == 0 and dy == 0:
        return
    end_point = find_end_point(matrix, x, y, direction)
    current_x = end_point[0]
    current_y = end_point[1]
    while current_x != x or current_y != y:
        matrix[current_x + dx][current_y + dy] = matrix[current_x][current_y]
        matrix[current_x][current_y] = -1
        current_x = current_x - direction[0]
        current_y = current_y - direction[1]


# 判断是否被阻挡
def is_blocked(matrix, x, y, same_block, direction):
    if same_block is not None and (same_block[0] - x == 0 or same_block[1] - y == 0):
        nx = x
        ny = y
        while True:
            nx += direction[0]
            ny += direction[1]
            if nx == same_block[0] and ny == same_block[1]:
                break
            if is_valid(matrix, nx, ny):
                if matrix[nx][ny] == -1:
                    continue
                else:
                    return True
            else:
                break
    return False


# 尝试移动方块
def try_move_block(matrix, x, y, dx, dy, direction):
    block = matrix[x][y]

    if direction[0] == 0:
        to_x = False
        real_distance = dy
    else:
        to_x = True
        real_distance = dx

    if real_distance > 0:
        start = 0
        end = real_distance + 1
    else:
        start = real_distance
        end = 1

    for i in range(start, end):
        if to_x:
            move_x = i
            move_y = 0
        else:
            move_x = 0
            move_y = i
        same_block = find_same_block(matrix, x, y, x + move_x, y + move_y, block)
        blocked = is_blocked(matrix, x, y, same_block, direction)
        if blocked:
            continue

        if same_block is not None:
            new_matrix = matrix.copy()
            new_matrix[same_block[0]][same_block[1]] = -1
            new_matrix[x][y] = -1
            move_block(new_matrix, x, y, move_x, move_y, direction)
            return new_matrix, same_block, move_x, move_y
    return None, None, None, None


# 获取矩阵的哈希值
def get_matrix_hash(matrix):
    return hash(matrix.tostring())


# DFS + 记忆化搜索
min_steps = float('inf')
max_tries = 2e4
max_tries = 2
min_path = []
visited = set()
tries = 0

def dfs(matrix, path, steps):
    global min_steps, min_path, visited, tries
    tries += 1
    # print(tries, np.sum(matrix != -1), min_steps)
    matrix_hash = get_matrix_hash(matrix)
    if min_steps != float('inf'):
        return
    if tries > max_tries:
        return
    if matrix_hash in visited:
        return
    visited.add(matrix_hash)

    if np.all(matrix == -1):
        if steps < min_steps:
            min_steps = steps
            min_path = path
        return

    row, col = matrix.shape
    # 优先从中间区域开始遍历
    mid_row = row // 2
    mid_col = col // 2
    offsets = [(0, 0)]
    for offset_row, offset_col in offsets:
        for i in range(-8, 9):
            for j in range(-8, 9):
                x = mid_row + i + offset_row
                y = mid_col + j + offset_col
                if 0 <= x < row and 0 <= y < col and matrix[x][y] != -1:
                    for direction in directions:
                        distance = get_direction_distance(x, y, matrix, direction)
                        new_matrix, same_block, move_x, move_y = try_move_block(matrix, x, y, distance[0], distance[1], direction)
                        if new_matrix is not None:
                            print(f"distance {distance} x {x} y {y} matrix[x][y] {matrix[x][y]} same_block {same_block} move_x {move_x} move_y {move_y}")
                            print("old_matrix\n", matrix)
                            print("new_matrix\n", new_matrix)
                            new_path = path + [((x, y), same_block, (move_x, move_y))]
                            dfs(new_matrix, new_path, steps + 1)

    # 遍历剩余区域
    for x in range(row):
        for y in range(col):
            if matrix[x][y] != -1:
                for direction in directions:
                    distance = get_direction_distance(x, y, matrix, direction)
                    new_matrix, same_block, move_x, move_y = try_move_block(matrix, x, y, distance[0], distance[1], direction)
                    if new_matrix is not None:
                        new_path = path + [((x, y), same_block, (move_x, move_y))]
                        dfs(new_matrix, new_path, steps + 1)


# 游戏开始
def game_start(matrix, category_images, visualize, mouse_control):
    global min_steps, min_path, tries
    dfs(matrix, [], 0)
    if min_steps == float('inf'):
        print("No solution found.")
        return
    else:
        print(f"Solution Found Step {min_steps} Tried {tries}")
        print("Path: ", min_path)

    current_steps = 0
    # x_min = 400 * 2
    x_min = 40
    x_max = x_min + 1840 - 1096
    # y_min = 232 * 2
    y_min = 330
    y_max = y_min + 1468 - 416
    x_gap = (x_max - x_min) / 10
    y_gap = (y_max - y_min) / 14

    if mouse_control:
        pyautogui.click((x_min + x_gap // 2) // 2, (y_min - y_gap // 2) // 2, button='left')
        time.sleep(0.1)

    for (x, y), same_block, (move_x, move_y) in tqdm(min_path):
        current_steps += 1
        left_vis = visualize_matrix(matrix, category_images, (x, y), (x+move_x, y+move_y), same_block)
        matrix[same_block[0]][same_block[1]] = -1
        matrix[x][y] = -1
        # distance = get_direction_distance(x, y, matrix, direction)
        move_x = int(move_x)
        move_y = int(move_y)
        direction = [move_x // abs(move_x) if move_x != 0 else move_x, move_y // abs(move_y) if move_y != 0 else move_y]
        # print((x, y), same_block, (move_x, move_y), direction)
        # x hang y lie
        move_block(matrix, x, y, move_x, move_y, direction)
        if visualize:
            print_matrix(matrix, current_steps, [x, y], same_block)
            right_vis = visualize_matrix(matrix, category_images)
            cv2.imshow('left_vis', left_vis)
            cv2.imshow('right_vis', right_vis)
            cv2.waitKey(0)
            time.sleep(0)
        if mouse_control:
            pyautogui.moveTo((x_min + x_gap * y + x_gap // 2) // 2, (y_min + y_gap * x + y_gap // 2) // 2,
                             duration=0.2)
            pyautogui.mouseDown()
            pyautogui.moveTo((x_min + x_gap * (y + move_y) + x_gap // 2) // 2, (y_min + y_gap * (x + move_x) + y_gap // 2) // 2,
                             duration=0.2)
            pyautogui.mouseUp()
            pyautogui.click((x_min + x_gap * same_block[1] + x_gap // 2) // 2, (y_min + y_gap * same_block[0] + y_gap // 2) // 2, button='left')
            time.sleep(0.1)


def game_pipeline():
    debug = True
    if debug:
        # timestamp = "2025-04-18_20-19-00"
        # timestamp = "2025-04-18_20-09-12"
        timestamp = "2025-04-19_16-58-59"
        # new_image = cv2.imread(f'data/debug_{timestamp}.jpg')
    else:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        opencv_image = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'data/screen_{timestamp}.png', opencv_image)
        print("Screen shot saved.")

        x_min = 40
        x_max = x_min + 1840 - 1096
        y_min = 330
        y_max = y_min + 1468 - 416
        new_image = opencv_image[y_min:y_max, x_min:x_max]
        cv2.imwrite(f'data/debug_{timestamp}.jpg', new_image)
        print(f"Debug image saved.")

    get_categority_csv(f'data/debug_{timestamp}.jpg', f'data/content_{timestamp}.csv')
    matrix, icon_name_matrix = csv_to_matrix(f'data/content_{timestamp}.csv')
    image_matrix = get_matrix(f'data/debug_{timestamp}.jpg',
                              14, 10, 3, 3)
    category_images = get_category_images(matrix, image_matrix)
    print(matrix)
    print(icon_name_matrix)
    for row in icon_name_matrix:
        for col in row:
            print(f"{col[:3]:>8}", end=' ')
        print()

    # for key, value in category_images.items():
    #     print(key, value.shape)

    visualize = False
    mouse_control = False
    game_start(matrix, category_images, visualize, mouse_control)
    print("Game Success!")


if __name__ == '__main__':
    game_pipeline()
