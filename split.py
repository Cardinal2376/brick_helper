import time

import cv2
import pyautogui



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


if __name__ == '__main__':
    split()
    # mouse_control()
