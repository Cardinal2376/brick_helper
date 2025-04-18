import cv2
import numpy as np
from itertools import combinations


def preprocess_image(image_path, grid_rows=10, grid_cols=14):
    """
    预处理图像，分割出每个棋盘格的子图
    :param image_path: 截图路径
    :param grid_rows: 行数
    :param grid_cols: 列数
    :return: 子图列表（按行优先顺序存储），每个元素是（x,y,sub_image）
    """
    # 读取图像并转为灰度图
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取图像尺寸
    h, w = gray.shape[:2]

    # 计算每个格子的尺寸（假设均匀分布）
    cell_w = w // grid_cols
    cell_h = h // grid_rows

    sub_images = []
    for y in range(grid_rows):
        for x in range(grid_cols):
            # 计算子图坐标（考虑可能的边界误差）
            x1 = x * cell_w
            y1 = y * cell_h
            x2 = min((x + 1) * cell_w, w)
            y2 = min((y + 1) * cell_h, h)

            # 提取子图并调整为固定尺寸（32x32用于pHash）
            sub_img = gray[y1:y2, x1:x2]
            sub_img = cv2.resize(sub_img, (32, 32))

            sub_images.append((x, y, sub_img))  # 存储坐标和子图

    return sub_images


def phash(image):
    """
    计算感知哈希值（用于图像相似性比较）
    :param image: 灰度图像（32x32）
    :return: 64位哈希值（二进制字符串）
    """
    # 计算DCT（离散余弦变换）
    dct = cv2.dct(np.float32(image))
    # 取左上角8x8区域（低频部分）
    dct_low = dct[:8, :8]
    # 计算均值（排除直流分量）
    avg = np.mean(dct_low)
    # 生成哈希：大于均值的位置为1，否则为0
    hash_str = ''.join(['1' if x > avg else '0' for x in dct_low.flatten()])
    return hash_str


def hamming_distance(hash1, hash2):
    """计算两个哈希值的汉明距离"""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def find_matching_pairs(sub_images, threshold=5):
    """
    寻找匹配的格子对
    :param sub_images: 预处理后的子图列表
    :param threshold: 汉明距离阈值（越小匹配越严格）
    :return: 匹配对列表 [( (x1,y1), (x2,y2) ), ...]
    """
    # 预计算所有子图的哈希值
    hash_list = [((x, y), phash(img)) for (x, y, img) in sub_images]
    pos_list = [(x, y) for (x, y, img) in sub_images]

    # 生成所有可能的两两组合
    matching_pairs = []
    for (i, j) in combinations(range(len(hash_list)), 2):
        (pos1, hash1) = hash_list[i]
        (pos2, hash2) = hash_list[j]

        # 计算汉明距离
        distance = hamming_distance(hash1, hash2)

        # 小于阈值认为匹配
        if distance <= threshold:
            matching_pairs.append((pos_list[i], pos_list[j]))

    return matching_pairs


# 使用示例
if __name__ == "__main__":
    # 预处理图像，分割子图
    sub_images = preprocess_image("data/debug_split_0.jpg", grid_rows=7, grid_cols=5)

    # 寻找匹配对（阈值5可根据实际效果调整）
    matches = find_matching_pairs(sub_images, threshold=5)

    # 输出结果
    for (pos1, pos2) in matches:
        print(f"匹配对：({pos1[0]},{pos1[1]}) <-> ({pos2[0]},{pos2[1]})")