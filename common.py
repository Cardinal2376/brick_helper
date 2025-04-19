import pandas as pd
import numpy as np


def csv_to_matrix(csv_file_path, rows=14, cols=10):
    df = pd.read_csv(csv_file_path)
    matrix = np.zeros((rows, cols), dtype=int)
    icon_name_matrix = [[] for _ in range(rows)]
    icon_category_map = {}
    category_index = 1
    category_count = {}  # 用于统计每一个类别图标个数

    for index, row in df.iterrows():
        x, y, icon_name = row
        icon_name = icon_name.strip()
        if icon_name not in icon_category_map:
            if icon_name == "C-01":
                icon_category_map[icon_name] = -1
            else:
                icon_category_map[icon_name] = category_index
                category_index += 1
            category_count[icon_name] = 0
        matrix[y, x] = icon_category_map[icon_name]
        icon_name_matrix[y].append(icon_name)
        category_count[icon_name] += 1  # 该类别的计数加1
    num_categories = len(icon_category_map)
    print(num_categories)
    for key, value in category_count.items():
        print(key, value)
        if value % 2 == 1:
            raise Exception(f"Error Recognize at key {key} value {value}")
    return matrix, icon_name_matrix
