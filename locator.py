import cv2
import numpy as np

# 使用示例

def locate():
    image_path = "pictures/20250412-151736.jpeg"
    frame = cv2.imread(image_path)
    x1 = int(0.33 * frame.shape[1])
    y1 = int(0.33 * frame.shape[0])
    x2 = int(0.67 * frame.shape[1])
    y2 = int(0.67 * frame.shape[0])

    new_frame = frame[y1:y2, x1:x2]
    cv2.imshow("new_frame", new_frame)
    cv2.waitKey()

def mllm():
    pass



if __name__ == '__main__':
    locate()

# 示例使用
# board = find_board("data/debug_2025-04-20_11-42-09.jpg")
# if board:
    # 这里可以进一步处理每个单元格（如识别图标）
    # print(board)
    # pass
