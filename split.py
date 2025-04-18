import time

import cv2
# import pyautogui
from solver import get_matrix
import numpy as np
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


def extract_feature(image):
    """使用CNN提取特征向量"""
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy()

def cosine_similarity(vec1, vec2):
    rtn = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return rtn


def split():
    path = "data/screen_2025-04-13_13-16-40.png"
    image = cv2.imread(path)

    x_min = 40
    x_max = x_min + 1840 - 1096
    y_min = 330
    y_max = y_min + 1468 - 416
    print(x_min, y_min, x_max, y_max)
    new_image = image[y_min:y_max, x_min:x_max]

    x_min = 40
    x_max = x_min + 1840 - 1096
    y_min = 330
    y_max = y_min + 1468 - 416
    new_image = image[y_min:y_max, x_min:x_max]
    # cv2.imwrite(f'data/debug_.jpg', new_image)
    # print(f"Debug image saved.")
    for i in range(2):
        for j in range(2):
            new_image = image[y_min + (y_max - y_min) // 2 * i:y_min + (y_max - y_min) // 2 * (i + 1),
                        x_min + (x_max - x_min) // 2 * j:x_min + (x_max - x_min) // 2 * (j + 1)]
            cv2.imwrite(f'data/debug_split_{i}_{j}.jpg', new_image)
            print(f"{i} {j} Debug image saved.")
    # cv2.imwrite("data/game01_level1.jpg", new_image)
    # cv2.waitKey()



def mouse_control():
    pyautogui.FAILSAFE = True  # 启用安全措施

    while True:
        x, y = pyautogui.position()
        print(f"当前鼠标位置: ({x}, {y})")
        time.sleep(0.2)



def phash(image):
    """
    计算感知哈希值（用于图像相似性比较）
    :param image: 灰度图像（32x32）
    :return: 64位哈希值（二进制字符串）
    """
    # 计算DCT（离散余弦变换）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (32, 32))
    dct = cv2.dct(np.float32(image))
    # 取左上角8x8区域（低频部分）
    dct_low = np.array(dct[:10, :10])
    # dct_low = np.array(dct[:8, :8].flatten())
    # 计算均值（排除直流分量）
    avg = np.mean(dct_low)
    # 生成哈希：大于均值的位置为1，否则为0
    hash_str = ''.join(['1' if x > avg else '0' for x in dct_low.flatten()])
    return hash_str


def dhash(image, size=8):
    """基于梯度的哈希（捕捉边缘细节）"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (size+1, size))  # 9x8用于计算水平梯度
    diff = resized[:, 1:] > resized[:, :-1]  # 水平相邻像素比较
    print(diff)
    return ''.join(['1' if d else '0' for row in diff for d in row])

def color_hash(image, bins=4):
    """颜色直方图哈希（捕捉颜色分布）"""
    # 转为HSV空间（对颜色更敏感）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 计算三通道直方图（H:0-180, S:0-255, V:0-255）
    hist = cv2.calcHist([hsv], [0,1,2], None, [bins]*3, [0,180,0,255,0,255])
    # 归一化并二值化（根据均值）
    hist = hist.flatten() / hist.sum()
    avg = np.mean(hist)
    return ''.join(['1' if h > avg else '0' for h in hist])


def hamming_distance(hash1, hash2):
    """计算两个哈希值的汉明距离"""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def combined_hash(image):
    # 同时计算dHash（边缘）和color_hash（颜色）
    d = dhash(image)
    c = color_hash(image)
    return d + c  # 合并为长哈希


def get_categority_csv(image_path, csv_path):
    image_matrix, img1 = get_matrix(image_path,
                                    14, 10, 3, 3)
    catgeory_matrix = np.zeros((14, 10), dtype=int)
    feature_matrix = [[] for _ in range(14)]
    for i in range(14):
        for j in range(10):
            feature_matrix[i].append(extract_feature(image_matrix[i][j]))
    category_cnt = 0
    threshold = 0.98
    for ki in range(14):
        for kj in range(10):
            if catgeory_matrix[ki][kj] == 0:
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


def split_images():
    image_matrix, img1 = get_matrix(f'data/debug_2025-04-13_13-13-51.jpg',
                              14, 10, 3, 3)
    hash_matrix = [[] for _ in range(14)]
    for i in range(14):
        for j in range(10):
            hash_matrix[i].append(extract_feature(image_matrix[i][j]))
            # print(extract_feature(image_matrix[i][j]).shape)

    hamming_distance_matrix = np.zeros((14, 10))
    new_micro_w = 45
    new_micro_h = 45

    for ki in range(14):
        for kj in range(10):
            cur_img1 = img1.copy()
            print(f"ki {ki} kj {kj}")
            print('----' * 10)
            cnt = 0
            for i in range(14):
                for j in range(10):
                    hamming_distance_matrix[i][j] = cosine_similarity(hash_matrix[ki][kj], hash_matrix[i][j])
                    if hamming_distance_matrix[i][j] > 0.98:
                        cnt += 1
                        cv2.rectangle(cur_img1, (j * new_micro_w, i * new_micro_h),
                                      ((j + 1) * new_micro_w, (i + 1) * new_micro_h),
                                      (0, 112, 224), 3)
                    print(f"{hamming_distance_matrix[i][j]:.3f} ", end='')
                    if j == 9:
                        print('')
            print(f"ki {ki} kj {kj} cnt = {cnt}")
            cv2.imshow('img1', cur_img1)
            cv2.waitKey(0)

    # 7 8
    # cv2.imshow("00", image_matrix[0][0])
    # cv2.imshow("11", image_matrix[1][1])
    # cv2.imshow("12 9", image_matrix[12][9])
    # cv2.waitKey(0)

if __name__ == '__main__':
    get_categority_csv("data/debug_2025-04-13_13-13-51.jpg", "data/content_2025-04-13_13-13-51.csv")
    # split_images()

    # split()
    # mouse_control()
