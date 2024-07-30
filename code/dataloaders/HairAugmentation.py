import os
import random

import cv2
import numpy as np
import pandas as pd
from albumentations import ImageOnlyTransform
from PIL import Image
from utils import path_check


class HairAugmentation(ImageOnlyTransform):
    def __init__(self, max_hairs: int = 10, hairs_dir: str = "../../data/Hairs", always_apply=False, p=0.8):
        self.max_hairs = max_hairs
        self.hairs_dir = hairs_dir
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        img_copy = img.copy()
        hairs_num = random.randint(0, self.max_hairs)
        if not hairs_num:
            return img_copy

        height, width, _ = img_copy.shape   # image shape
        hair_images = [hair_img for hair_img in os.listdir(self.hairs_dir) if 'png' in hair_img]

        hair = cv2.imread(os.path.join(self.hairs_dir, random.choice(hair_images)).replace("\\", "/"))
        hair = cv2.cvtColor(hair, cv2.COLOR_BGR2RGB)
        for _ in range(hairs_num):
            # hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            # hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair shape
            roi_ho = random.randint(0, height-h_height)
            roi_wo = random.randint(0, width-h_width)
            roi = img_copy[roi_ho:roi_ho+h_height, roi_wo:roi_wo+h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_RGBA2GRAY)

            ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            img_bg = cv2.bitwise_and(roi, roi, mask=mask)

            hair_fg = cv2.bitwise_and(hair, hair, mask=mask_inv)

            #Destination image
            dst = cv2.add(img_bg, hair_fg, dtype=cv2.CV_64F)
            img_copy[roi_ho:roi_ho+h_height, roi_wo:roi_wo+h_width] = dst

        return img_copy


def get_path_and_label_by_csv(data_dir):
    data_path = os.path.join(data_dir, "list.csv").replace("\\", "/")
    data = []
    df = pd.read_csv(data_path)
    for index, row in df.iterrows():
        img_path = row.iloc[0]
        label = row.iloc[1]
        classname = row.iloc[2]
        if os.path.exists(img_path):
            data.append([img_path, label, classname])
        else:
            print("{} does not exists".format(img_path))
            exit(0)
    print(len(data))
    return data


def hair_augmentation_and_save(data_dir, max_hairs, hairs_dir, p, save_dir):
    print(data_dir)
    data = get_path_and_label_by_csv(data_dir)
    data = np.asarray(data)
    img_paths = data[:, 0]
    labels = data[:, 1]
    classnames = data[:, 2]

    path_check(save_dir)
    csv_path = os.path.join(save_dir, "list.csv").replace("\\", "/")

    augmented_img_paths = []
    hair_augmentation = HairAugmentation(max_hairs=max_hairs, hairs_dir=hairs_dir, always_apply=False, p=p)
    for index, path in enumerate(img_paths):
        img = Image.open(path)
        img_np = np.asarray(img)
        augmented_img_np = hair_augmentation(image=img_np)["image"]
        augmented_img = Image.fromarray(augmented_img_np)

        save_path = os.path.join(save_dir, os.path.basename(path)).replace("\\", "/")
        augmented_img.save(save_path)
        augmented_img_paths.append(save_path)

    data_df = pd.DataFrame({'img_paths': augmented_img_paths, 'labels': labels, 'classnames': classnames})
    data_df.to_csv(csv_path, index=False)


def main1():
    data_dir = "../../data/Dermopathy/LQ/Train_OverSample"
    max_hairs = 20
    hair_dir = "../../data/Hairs"
    p = 0.8
    save_dir = "../../data/Dermopathy/LQ_Hairs_Aug/Train"
    hair_augmentation_and_save(data_dir, max_hairs, hair_dir, p, save_dir)
    print("植发完成")


def main2():
    data_dir = "../../data/Dermopathy/LQ/Test"
    max_hairs = 20
    hair_dir = "../../data/Hairs"
    p = 0.8
    save_dir = "../../data/Dermopathy/LQ_Hairs_Aug/Test"
    hair_augmentation_and_save(data_dir, max_hairs, hair_dir, p, save_dir)
    print("植发完成")


if __name__ == "__main__":
    main1()
    main2()
