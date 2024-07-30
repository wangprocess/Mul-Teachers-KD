import math
import os
import numpy as np
import pandas as pd
from PIL import Image
from utils import path_check
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


def get_csv_filenames(directory):
    csv_filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            csv_filenames.append(filename)
    return csv_filenames


def get_data_and_label(data_dir, label_dir):
    csv_filenames = get_csv_filenames(label_dir)
    if len(csv_filenames) > 1:
        print("Q:a dataset with two csv label file?")
        return
    else:
        csv_path = os.path.join(label_dir, csv_filenames[0]).replace("\\", "/")
        df = pd.read_csv(csv_path, encoding="UTF-8", header=0)
        data = []
        for index, row in df.iterrows():
            img_name = row.iloc[0] + ".jpg"
            img_path = os.path.join(data_dir, img_name).replace("\\", "/")
            label = row.iloc[1:].values.argmax()
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
        print("data len == 0")
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


def HQ_LQ_split():
    """
    We use one dataset to simulate high quality data and low quality data, this function used to split HQ and LQ
    ratio: the ratio to split train dataset
    """
    train_data_dir = "../../data/ISIC2018/ISIC2018_Task3_Training_Input"
    train_label_dir = "../../data/ISIC2018/ISIC2018_Task3_Training_GroundTruth"
    HQ_save_path = "../../data/Dermopathy/HQ/Train"
    LQ_save_path1 = "../../data/Dermopathy/LQ/Train_wo_LQ_process"
    LQ_save_path2 = "../../data/Dermopathy/LQ/Train"
    ratio = 0.4
    data = get_data_and_label(train_data_dir, train_label_dir)
    HQ_end_index = math.floor(len(data) * ratio) #向下取整,LQ的数量要比HQ多
    HQ_data = data[0:HQ_end_index]
    LQ_data = data[HQ_end_index:]
    resize_save_and_generate_csv(HQ_data, HQ_save_path)
    resize_save_and_generate_csv(LQ_data, LQ_save_path1)
    LQ_process(LQ_save_path1, LQ_save_path2)


def val_data_process():
    val_data_dir = "../../data/ISIC2018/ISIC2018_Task3_Validation_Input"
    val_label_dir = "../../data/ISIC2018/ISIC2018_Task3_Validation_GroundTruth"
    val_save_dir = "../../data/Dermopathy/HQ/Validation"
    val_data = get_data_and_label(val_data_dir, val_label_dir)
    resize_save_and_generate_csv(val_data, val_save_dir)

def test_data_process():
    test_data_dir = "../../data/ISIC2018/ISIC2018_Task3_Test_Input"
    test_label_dir = "../../data/ISIC2018/ISIC2018_Task3_Test_GroundTruth"
    test_save_dir = "../../data/Dermopathy/HQ/Test"
    test_data = get_data_and_label(test_data_dir, test_label_dir)
    resize_save_and_generate_csv(test_data, test_save_dir)


def LQ_test_data_process():
    test_save_dir = "../../data/Dermopathy/HQ/Test"
    LQ_test_save_dir = "../../data/Dermopathy/LQ/Test"
    LQ_process(test_save_dir, LQ_test_save_dir)


if __name__ == "__main__":
    class_dict = {
        'MEL': 'Melanoma',
        'NV': 'Melanocytic nevus',
        'BCC': 'Basal cell carcinoma',
        'AKIEC': 'Actinic keratosis',
        'BKL': 'Benign keratosis',
        'DF': 'Dermatofibroma',
        'VASC': 'Vascular lesion'
    }
    class_list = list(class_dict.keys())
    height = 224
    width = 224

    HQ_LQ_split()
    val_data_process()
    test_data_process()
    LQ_test_data_process()
