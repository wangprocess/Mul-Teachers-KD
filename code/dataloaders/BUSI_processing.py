import math
import os
import numpy as np
import pandas as pd
import glob
import random
import shutil
from PIL import Image
from utils import path_check
from sklearn.model_selection import train_test_split
import albumentations as A
import cv2


LQ_transforms = A.Compose([
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=(3, 5)),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.7),

    # A.CLAHE(clip_limit=4.0, p=0.7),
    # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
    # A.RGBShift(10, 10, 10, p=0.5),
])


def LQ_process(data_dir, save_dir):
    count = 0
    path_check(save_dir)

    img_paths = []
    labels = []
    classnames = []
    df = pd.read_csv(os.path.join(data_dir, "list.csv").replace("\\", "/"))
    if len(df) == 0:
        print("data len == 0")
        return
    for index, row in df.iterrows():
        img = cv2.imread(row.iloc[0])
        img_after_aug = LQ_transforms(image=img)['image']
        save_path = os.path.join(save_dir, os.path.basename(row.iloc[0])).replace("\\", "/")
        cv2.imwrite(save_path, img_after_aug)
        img_paths.append(save_path)
        labels.append(row.iloc[1])
        classnames.append(row.iloc[2])
        count += 1
    LQ_df = pd.DataFrame({'img_paths': img_paths, 'labels': labels, 'classnames': classnames})
    # shuffe
    LQ_df = LQ_df.sample(frac=1).reset_index(drop=True)
    LQ_df.to_csv(os.path.join(save_dir, "list.csv").replace("\\", "/"), index=False)
    print("sum of LQ", count)


def get_data_and_label(data_dir):
    data = []
    data_dir = data_dir + "/*.png"
    img_paths = sorted(glob.glob(data_dir))
    for img_path in img_paths:
        img_path = img_path.replace("\\", "/")
        label = -1
        if "benign" in img_path:
            label = 0
        elif "malignant" in img_path:
            label = 1
        elif "normal" in img_path:
            label = 2

        if os.path.exists(img_path):
            data.append([img_path, label])
        else:
            print("{} does not exists".format(img_path))
            exit(0)
    print(len(data))
    return data


def resize_save_and_generate_csv(data, save_dir):
    path_check(save_dir)

    if len(data) == 0:
        return

    data = np.asarray(data)
    img_paths = data[:, 0]
    labels = data[:, 1]
    processed_img_paths = []
    processed_labels = []

    for index, path in enumerate(img_paths):
        img = Image.open(path)
        img_np = np.asarray(img)
        if img_np.ndim == 3:
            img = img.resize((width, height))
            save_path = os.path.join(save_dir, os.path.basename(path)).replace("\\", "/")
            img.save(save_path)
            processed_img_paths.append(save_path)
            processed_labels.append(labels[index])
        else:
            print("image {} shape: {} ".format(path, img_np.shape))
            return

    classnames = []

    for i in range(len(processed_labels)):
        for j in range(len(class_list)):
            if processed_labels[i] == str(j):
                classnames.append(class_list[j])
    data_df = pd.DataFrame({'img_paths': processed_img_paths, 'labels': processed_labels, 'classnames': classnames})
    data_df.to_csv(os.path.join(save_dir, "list.csv").replace("\\", "/"), index=False)


def filter_image_paths(paths):
    return [path for path in paths if "mask" not in path]


def Train_Val_Test_split():
    benign_data_dir = "../../data/BUSI/benign/*.png"
    malignant_data_dir = "../../data/BUSI/malignant/*.png"
    normal_data_dir = "../../data/BUSI/normal/*.png"
    benign_img_paths = sorted(glob.glob(benign_data_dir))
    benign_img_paths = filter_image_paths(benign_img_paths)
    print("benign num:", len(benign_img_paths))
    malignant_img_paths = sorted(glob.glob(malignant_data_dir))
    malignant_img_paths = filter_image_paths(malignant_img_paths)
    print("malignant num:", len(malignant_img_paths))
    normal_img_paths = sorted(glob.glob(normal_data_dir))
    normal_img_paths = filter_image_paths(normal_img_paths)
    print("normal num:", len(normal_img_paths))
    train_save_dir = "../../data/BUSI/Train"
    val_save_dir = "../../data/BUSI/Validation"
    test_save_dir = "../../data/BUSI/Test"

    train_ratio = 0.7
    val_ratio = 0.1

    # 划分训练集和剩余图片
    train_benign, remaining_benign = train_test_split(benign_img_paths, train_size=train_ratio, random_state=42)
    train_malignant, remaining_malignant = train_test_split(malignant_img_paths, train_size=train_ratio, random_state=42)
    train_normal, remaining_normal = train_test_split(normal_img_paths, train_size=train_ratio, random_state=42)

    # 划分验证集和测试集
    val_benign, test_benign = train_test_split(remaining_benign, train_size=val_ratio/(1-train_ratio), random_state=42)
    val_malignant, test_malignant = train_test_split(remaining_malignant, train_size=val_ratio/(1-train_ratio), random_state=42)
    val_normal, test_normal = train_test_split(remaining_normal, train_size=val_ratio/(1-train_ratio), random_state=42)

    # 移动图片到相应目录
    copy_images(train_benign, train_save_dir)
    copy_images(train_malignant, train_save_dir)
    copy_images(train_normal, train_save_dir)
    copy_images(val_benign, val_save_dir)
    copy_images(val_malignant, val_save_dir)
    copy_images(val_normal, val_save_dir)
    copy_images(test_benign, test_save_dir)
    copy_images(test_malignant, test_save_dir)
    copy_images(test_normal, test_save_dir)


def copy_images(images, save_dir):
    path_check(save_dir)
    for image_file in images:
        shutil.copy(image_file, save_dir)


def HQ_LQ_split():
    """
    We use one dataset to simulate high quality data and low quality data, this function used to split HQ and LQ
    ratio: the ratio to split train dataset
    """
    train_data_dir = "../../data/BUSI/Train"
    HQ_save_path = "../../data/Breast/HQ/Train"
    LQ_save_path1 = "../../data/Breast/LQ/Train_wo_LQ_process"
    LQ_save_path2 = "../../data/Breast/LQ/Train"
    ratio = 0.4
    data = get_data_and_label(train_data_dir)
    HQ_end_index = math.floor(len(data) * ratio) #向下取整,LQ的数量要比HQ多
    HQ_data = random.sample(data, HQ_end_index)
    LQ_data = [item for item in data if item not in HQ_data]
    resize_save_and_generate_csv(HQ_data, HQ_save_path)
    resize_save_and_generate_csv(LQ_data, LQ_save_path1)
    LQ_process(LQ_save_path1, LQ_save_path2)


def val_data_process():
    val_data_dir = "../../data/BUSI/Validation"
    val_save_dir = "../../data/Breast/HQ/Validation"
    val_data = get_data_and_label(val_data_dir)
    resize_save_and_generate_csv(val_data, val_save_dir)


def test_data_process():
    test_data_dir = "../../data/BUSI/Test"
    test_save_dir = "../../data/Breast/HQ/Test"
    test_data = get_data_and_label(test_data_dir)
    resize_save_and_generate_csv(test_data, test_save_dir)


def LQ_test_data_process():
    test_save_dir = "../../data/Breast/HQ/Test"
    LQ_test_save_dir = "../../data/Breast/LQ/Test"
    LQ_process(test_save_dir, LQ_test_save_dir)


if __name__ == "__main__":
    class_dict = {
        'BENIGN': 'Benign',
        'MALIGNANT': 'Malignant',
        'NORMAL': 'Normal'
    }
    class_list = list(class_dict.keys())
    height = 224
    width = 224

    Train_Val_Test_split()
    HQ_LQ_split()
    val_data_process()
    test_data_process()
    LQ_test_data_process()
