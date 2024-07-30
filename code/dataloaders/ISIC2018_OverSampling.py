import os

import cv2
import pandas as pd
import albumentations as A
from utils import path_check

transforms = A.Compose([
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    A.RGBShift(10, 10, 10, p=0.5),
    A.ShiftScaleRotate(0.2, 0.2, 30, p=0.5)
])


def overSampling(data_dir, ratio, save_dir):
    count = 0
    path_check(save_dir)

    img_paths = []
    labels = []
    classnames = []

    df = pd.read_csv(data_dir)
    for index, row in df.iterrows():
        img = cv2.imread(row.iloc[0])
        save_path = os.path.join(save_dir, os.path.basename(row.iloc[0])).replace("\\", "/")
        cv2.imwrite(save_path, img)
        img_paths.append(save_path)
        labels.append(row.iloc[1])
        classnames.append(row.iloc[2])
        if row.iloc[2] != 'NV':
            img = cv2.imread(row.iloc[0])
            for i in range(ratio[row.iloc[2]]):
                img_after_aug = transforms(image=img)['image']
                save_path = os.path.join(save_dir, os.path.basename(row.iloc[0]).replace(".jpg", "_Aug")).replace("\\",
                                                                                                                  "/") + str(
                    i) + ".jpg"
                cv2.imwrite(save_path, img_after_aug)
                img_paths.append(save_path)
                labels.append(row.iloc[1])
                classnames.append(row.iloc[2])
                count += 1
    aug_df = pd.DataFrame({'img_paths': img_paths, 'labels': labels, 'classnames': classnames})
    # shuffe
    aug_df = aug_df.sample(frac=1).reset_index(drop=True)
    aug_df.to_csv(os.path.join(save_dir, "list.csv").replace("\\", "/"), index=False)
    print("sum of oversample", count)


def TrainData_OverSampling():
    data_dir = "../../data/Dermopathy/HQ/Train/list.csv"
    df = pd.read_csv(data_dir)
    label_counts = df.groupby('classnames').size()
    print("origin label distribution:")
    print(label_counts)
    ratio = {
        'MEL': label_counts['NV'] // label_counts['MEL'],
        'BCC': label_counts['NV'] // label_counts['BCC'],
        'AKIEC': label_counts['NV'] // label_counts['AKIEC'],
        'BKL': label_counts['NV'] // label_counts['BKL'],
        'DF': label_counts['NV'] // label_counts['DF'],
        'VASC': label_counts['NV'] // label_counts['VASC']
    }
    save_dir = "../../data/Dermopathy/HQ/Train_OverSample"
    overSampling(data_dir, ratio, save_dir)
    df_aug = pd.read_csv(os.path.join(save_dir, "list.csv").replace("\\", "/"))
    label_counts_aug = df_aug.groupby('classnames').size()
    print("OverSampling label distribution:")
    print(label_counts_aug)


def LQ_TrainData_OverSampling():
    data_dir = "../../data/Dermopathy/LQ/Train/list.csv"
    df = pd.read_csv(data_dir)
    label_counts = df.groupby('classnames').size()
    print("origin label distribution:")
    print(label_counts)
    ratio = {
        'MEL': label_counts['NV'] // label_counts['MEL'],
        'BCC': label_counts['NV'] // label_counts['BCC'],
        'AKIEC': label_counts['NV'] // label_counts['AKIEC'],
        'BKL': label_counts['NV'] // label_counts['BKL'],
        'DF': label_counts['NV'] // label_counts['DF'],
        'VASC': label_counts['NV'] // label_counts['VASC']
    }
    save_dir = "../../data/Dermopathy/LQ/Train_OverSample"
    overSampling(data_dir, ratio, save_dir)
    df_aug = pd.read_csv(os.path.join(save_dir, "list.csv").replace("\\", "/"))
    label_counts_aug = df_aug.groupby('classnames').size()
    print("OverSampling label distribution:")
    print(label_counts_aug)


if __name__ == "__main__":
    """
    Train data:
        AKIEC     184
        BCC       266
        BKL       566
        DF         56
        MEL       435
        NV       3435
        VASC       65
    """
    TrainData_OverSampling()
    LQ_TrainData_OverSampling()
