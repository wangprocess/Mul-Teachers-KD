import os
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd


def path_check(path):
    # 检查路径是否存在
    if not os.path.isdir(path):
        # 如果路径不存在，则创建
        os.makedirs(path)
        print(f"Path '{path}' has been created.")


def compute_mean_std(data_df):
    imgs = []
    for i in tqdm(range(len(data_df))):
        img = cv2.imread(data_df.iloc[i, 0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    imgs = np.array(imgs)
    imgs = imgs.astype(np.float32) / 255
    means = []
    stds = []
    for i in range(3):
        pixels = imgs[:, :, :, i].ravel()
        means.append(np.mean(pixels))
        stds.append(np.std(pixels))
    return means, stds


if __name__ == "__main__":
    csv_path = "../../data/DermnetData/HQ/Train/list.csv"
    df = pd.read_csv(csv_path)
    mean, std = compute_mean_std(df)
    print("mean:", mean)
    print("std:", std)
    """
    mean: [0.7831249, 0.5446379, 0.5653467]
    std: [0.1327309, 0.14866614, 0.16493498]
    """

    """
    mean: [0.3341159, 0.3341075, 0.33405247]
    std: [0.22006492, 0.22006334, 0.22004794]
    """