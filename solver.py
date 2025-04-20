import random

import numpy as np
import cv2
import pandas as pd
import time
import pyautogui
import datetime
from tqdm import tqdm
from gpt import get_decription_csv, get_decription_csv_sensenova, merge_csv_files
from common import csv_to_matrix
# from split import get_categority_csv
pyautogui.FAILSAFE = True  # 启用安全措施

# 定义上下左右四个方向
directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
is_print_matrix = True


import torch
import torchvision.models as models
from torchvision import transforms


model = models.mobilenet_v3_small(pretrained=True)
model.eval()
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def cosine_similarity(vec1, vec2):
    rtn = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return rtn


def extract_feature(image):
    """使用CNN提取特征向量"""
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy()


def get_categority_csv(image_path, csv_path):
    image_matrix, img1 = get_matrix(image_path,
                                    14, 10, 3, 3)
    catgeory_matrix = np.zeros((14, 10), dtype=int)
    feature_matrix = [[] for _ in range(14)]
    for i in range(14):
        for j in range(10):
            feature_matrix[i].append(extract_feature(image_matrix[i][j]))
    category_cnt = 0
    threshold = 0.975
    blank = cv2.imread("data/blank.jpg")
    blank_feature = extract_feature(blank)
    for ki in range(14):
        for kj in range(10):
            if catgeory_matrix[ki][kj] == 0:
                if cosine_similarity(feature_matrix[ki][kj], blank_feature) > 0.1:
                    catgeory_matrix[ki][kj] = -1
                else:
                    catgeory_matrix[ki][kj] = category_cnt + 1
                    category_cnt += 1
                    for i in range(14):
                        for j in range(10):
                            if catgeory_matrix[i][j] == 0 and cosine_similarity(feature_matrix[ki][kj],
                                                                                feature_matrix[i][j]) > threshold:
                                catgeory_matrix[i][j] = catgeory_matrix[ki][kj]

    with open(csv_path, 'w') as f:
        f.write('位置x,位置y,名称\n')
        for ki in range(14):
            for kj in range(10):
                f.write(f"{kj},{ki},C{catgeory_matrix[ki][kj]:03d}\n")
    print(f"csv written at {csv_path}")


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
    return image_matrix, img1

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
def find_end_point(matrix, x, y, direction, vis=False):
    while True:
        dx = x + direction[0]
        dy = y + direction[1]
        if vis:
            print(f" x y {x} {y} direction {direction} dx dy {dx} {dy}")
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
    # return max((nearest_remote_point[0] - end_point[0]), abs(nearest_remote_point[1] - end_point[1]))

    if direction[0] == 0:
        dis = (nearest_remote_point[1] - end_point[1]) // direction[1]
    else:
        dis = (nearest_remote_point[0] - end_point[0]) // direction[0]
    # if dis != 0:
    #     print("direction distance", matrix, x, y, direction, end_point, nearest_remote_point)
    #     print("dis= ", dis)
    return dis

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
def move_block(matrix, x, y, dx, dy, direction, vis=False):
    '''
    需要进行合法状态检测
    '''
    distance = max(abs(dx), abs(dy))
    succ = True
    end_point = find_end_point(matrix, x, y, direction, vis=vis)
    length = max(abs(end_point[0] - x), abs(end_point[1] - y))
    exception = f"matrix {matrix} x y {x} {y} matrix[x][y] {matrix[x][y]} dx dy {dx} {dy} direction {direction} "
    exception += f" endpoint {end_point}"
    # print(f"Move Block {exception}")
    if distance == 0:
        return succ
    for i in range(0, length+1, 1):
        cx = end_point[0] - direction[0] * i
        cy = end_point[1] - direction[1] * i
        nx = cx + dx
        ny = cy + dy
        if (0 <= cx < matrix.shape[0] and 0 <= cy < matrix.shape[1] and
                0 <= nx < matrix.shape[0] and 0 <= ny < matrix.shape[1] and matrix[nx][ny] == -1):
            # print(matrix)
            # print(f"i {i} cx cy {cx} {cy} matrix[cx][cy] {matrix[cx][cy]} nx ny {nx} {ny} matrix[nx][ny] {matrix[nx][ny]}")
            matrix[nx][ny] = matrix[cx][cy]
            matrix[cx][cy] = -1
        else:
            succ = False
            exception = f"matrix {matrix} x y {x} {y} matrix[x][y] {matrix[x][y]} dx dy {dx} {dy} direction {direction} "
            exception += f" endpoint {end_point} cx cy {cx} {cy} nx ny {nx} {ny}"
            raise Exception(f"Can Not Move! {exception}")
    return succ

    # if dx == 0 and dy == 0:
    #     return
    # end_point = find_end_point(matrix, x, y, direction)
    # current_x = end_point[0]
    # current_y = end_point[1]
    # while current_x != x or current_y != y:
    #     # print(f"current_x {current_x}, dx {dx}, current_y {current_y}, dy {dy}")
    #     matrix[current_x + dx][current_y + dy] = matrix[current_x][current_y]
    #     matrix[current_x][current_y] = -1
    #     current_x = current_x - direction[0]
    #     current_y = current_y - direction[1]


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
max_tries = 2e5
min_path = []
visited = set()
tries = 0
last_time = None
global_visited = set()

def sample_with_weights_unique(lst, weights, n):
    result = []
    remaining_lst = lst[:]
    remaining_weights = weights[:]

    while len(result) < n:
        # 从剩余元素中按权重采样
        sample = random.choices(population=remaining_lst, weights=remaining_weights, k=1)[0]
        if sample not in result:
            result.append(sample)
        # 更新剩余元素和权重
        index_to_remove = remaining_lst.index(sample)
        del remaining_lst[index_to_remove]
        del remaining_weights[index_to_remove]

    return result


def dfs(matrix, path, steps):
    global min_steps, min_path, visited, tries, last_time
    tries += 1
    # print(tries, np.sum(matrix != -1), min_steps)
    # depth =
    if tries % 1000 == 0:
        print(f"tries {tries}/{max_tries} visited {len(visited)} {tries/max_tries * 100:.4f}% Time Used: {time.time() - last_time:.2f}s")
    matrix_hash = get_matrix_hash(matrix)
    if min_steps != float('inf'):
        return
    if tries > max_tries:
        # print("tries exceed")
        return
    if matrix_hash in visited:
        # print("visited")
        return
    visited.add(matrix_hash)

    if np.all(matrix == -1):
        if steps < min_steps:
            min_steps = steps
            min_path = path
        return

    row, col = matrix.shape

    pending_list = []
    sample_weight = []

    # (x < mid_row - 5 or x > mid_row + 5 or y < mid_col - 5 or y > mid_col + 5)
    # 遍历剩余区域
    for x in range(row):
        for y in range(col):
            if matrix[x][y] != -1:
                weight_r1 = 1
                for m in range(len(directions)):
                    for k in range(1, 4):
                        rx = x + directions[m][0] * k
                        ry = y + directions[m][1] * k
                        if 0 <= rx < row and 0 <= ry < col and matrix[rx][ry] == -1:
                            weight_r1 += 1000 / (k + 1)
                for direction in directions:
                    distance = get_direction_distance(x, y, matrix, direction)
                    for k in range(distance, -1, -1):
                        move_x = direction[0] * k
                        move_y = direction[1] * k
                        same_block = find_same_block(matrix, x, y, x + move_x, y + move_y,
                                                     matrix[x][y])
                        blocked = is_blocked(matrix, x, y, same_block, direction)
                        if blocked or same_block is None:
                            continue
                        else:
                            pending_list.append([(x, y), same_block, (move_x, move_y), direction])
                            sample_weight.append(weight_r1)
                            # break

    tqdm_disable = True
    sample_weight = [x / sum(sample_weight) for x in sample_weight]
    # print(sample_weight)
    # pending_list = sample_with_weights_unique(pending_list, weights=sample_weight, n=len(pending_list))
    random.shuffle(pending_list)
    # pending_list =
    for item in tqdm(pending_list, desc=f'depth={len(path)}', disable=tqdm_disable):
        (x, y), same_block, (move_x, move_y), direction = item
        new_matrix = matrix.copy()

        # print("pending")
        # print(new_matrix)
        # print(f"depth {len(path)} x,{x} y {y}, matrix[x][y] {new_matrix[x][y]} move_x {move_x}, move_y {move_y}, same_block {same_block}, direction {direction}")
        # new_matrix[x][y] = -1
        # new_matrix[same_block[0]][same_block[1]] = -1
        new_path = path + [((x, y), same_block, (move_x, move_y), direction)]
        # move_block(new_matrix, x, y, move_x, move_y, direction)

        move_block(new_matrix, x, y, move_x, move_y, direction)
        new_matrix[same_block[0]][same_block[1]] = -1
        new_matrix[x + move_x][y + move_y] = -1
        # print(new_matrix)
        if get_matrix_hash(new_matrix) not in visited:
            dfs(new_matrix, new_path, steps + 1)

        # break

    # print(f"depth {len(path)} dfs cnt {len(pending_list)} tries {tries} max_tries {max_tries} set size {len(visited)}")



# 游戏开始
def game_start(matrix, category_images, visualize, mouse_control):
    global min_steps, min_path, tries, last_time, visited, max_tries, global_visited
    no_ans = True
    # seed_list = [24, 42, 13, 20, 100, 101, 102, 103, 104]
    seed_list = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 225, 16*16, 17*17, 18*18, 19*19, 20*20]
    max_tries_list = [3000, 10000]
    visited = set()
    global_visited = set()
    for max_tries_value in max_tries_list:
        max_tries = max_tries_value
        for seed in seed_list:
            last_time = time.time()
            tries = 0
            random.seed(seed)
            visited = set()
            min_path = []
            min_steps = float('inf')
            dfs(matrix, [], 0)
            if min_steps == float('inf'):
                print(f"Seed {seed} failed")
            else:
                print(f"Solution Found Step {min_steps} Tried {tries}")
                print("Path: ", min_path)
                no_ans = False
                break
        if not no_ans:
            break
    if no_ans:
        print("No solution found.")
        return False

    current_steps = 0
    # x_min = 400 * 2
    x_min = 40
    x_max = x_min + 1840 - 1096
    # y_min = 232 * 2
    y_min = 330
    y_max = y_min + 1468 - 416
    x_gap = (x_max - x_min) / 10
    y_gap = (y_max - y_min) / 14

    mouse_time_scale = 0.05

    if mouse_control:
        # pyautogui.click((x_min - x_gap // 2) // 2, (y_min + y_gap // 2) // 2, button='left')
        pyautogui.moveTo(((x_min + x_gap * 2) // 2, (y_min - y_gap) // 2))
        pyautogui.mouseDown()
        pyautogui.mouseUp()
        pyautogui.click((x_min + x_gap * 2) // 2, (y_min - y_gap) // 2, button='left')
        pyautogui.click((x_min + x_gap * 2) // 2, (y_min - y_gap) // 2, button='left')
        time.sleep(0.1)

    print("matrix", matrix)
    for (x, y), same_block, (move_x, move_y), direction in tqdm(min_path):
        current_steps += 1
        left_vis = visualize_matrix(matrix, category_images, (x, y), (x+move_x, y+move_y), same_block)
        # distance = get_direction_distance(x, y, matrix, direction)
        move_x = int(move_x)
        move_y = int(move_y)
        # direction = [move_x // abs(move_x) if move_x != 0 else 0, move_y // abs(move_y) if move_y != 0 else 0]
        # print((x, y), same_block, (move_x, move_y), direction)
        # x hang y lie
        move_block(matrix, x, y, move_x, move_y, direction, vis=False)
        matrix[same_block[0]][same_block[1]] = -1
        matrix[x+move_x][y+move_y] = -1

        if visualize:
            print_matrix(matrix, current_steps, [x, y], same_block)
            right_vis = visualize_matrix(matrix, category_images)
            cv2.imshow('left_vis', left_vis)
            cv2.imshow('right_vis', right_vis)
            cv2.waitKey(0)
            time.sleep(0)
        if mouse_control:
            pyautogui.moveTo((x_min + x_gap * y + x_gap // 2) // 2, (y_min + y_gap * x + y_gap // 2) // 2,
                             duration=mouse_time_scale*2)
            pyautogui.mouseDown()
            pyautogui.moveTo((x_min + x_gap * (y + move_y) + x_gap // 2) // 2, (y_min + y_gap * (x + move_x) + y_gap // 2) // 2,
                             duration=mouse_time_scale*2)
            pyautogui.mouseUp()
            pyautogui.click((x_min + x_gap * same_block[1] + x_gap // 2) // 2, (y_min + y_gap * same_block[0] + y_gap // 2) // 2, button='left')
            # time.sleep(mouse_time_scale)
    return True

def game_pipeline(debug=False):
    if debug:
        timestamp = "2025-04-19_22-41-52" # Level 16 seed 25 tries 4831
    else:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        opencv_image = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        print(f"No Debug TimeStep {timestamp}")
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
    # matrix = np.ones((14, 10), dtype=int)
    # for i in range(14):
    #     for j in range(10):
    #         matrix[i][j] = i * 10 + j
    # matrix[0][1] = -1
    # matrix[0][2] = -1
    # matrix[1][2] = 0


    matrix, icon_name_matrix = csv_to_matrix(f'data/content_{timestamp}.csv')
    image_matrix, img1 = get_matrix(f'data/debug_{timestamp}.jpg',
                              14, 10, 3, 3)
    category_images = get_category_images(matrix, image_matrix)
    # print(icon_name_matrix)
    print(matrix)

    visualize = debug
    mouse_control = not debug
    # mouse_control = True
    succ = game_start(matrix, category_images, visualize, mouse_control)
    if succ:
        print("Game Success!")
    else:
        print("Game Failed!")
    return succ


if __name__ == '__main__':
    debug = False
    for i in tqdm(range(1, 30), 'level'):
        succ = game_pipeline(debug=debug)
        if not succ:
            break
        if debug:
            break
        print("waiting for next level...")
        for j in tqdm(range(15), desc="sleeping"):
            time.sleep(1)
