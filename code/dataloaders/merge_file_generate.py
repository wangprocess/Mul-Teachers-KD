import os
import numpy as np
import pandas as pd


def merge_file(csv1, csv2, save_dir, save_csv_name):
    df1 = pd.read_csv(os.path.join(csv1, "list.csv").replace("\\", "/"))
    df2 = pd.read_csv(os.path.join(csv2, "list.csv").replace("\\", "/"))
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df.to_csv(os.path.join(save_dir, save_csv_name).replace("\\", "/"), index=False)


def main1():
    csv1 = "../../data/Dermopathy/HQ/Train_OverSample"
    csv2 = "../../data/Dermopathy/LQ_Hairs_Aug/Train"
    save_dir = "../../data/Dermopathy"
    save_csv_name = "train_merge_list.csv"
    merge_file(csv1, csv2, save_dir, save_csv_name)


def main2():
    csv1 = "../../data/Breast/HQ/Train_OverSample"
    csv2 = "../../data/Breast/LQ/Train_OverSample"
    save_dir = "../../data/Breast"
    save_csv_name = "train_merge_list.csv"
    merge_file(csv1, csv2, save_dir, save_csv_name)


def main3():
    csv1 = "../../data/Breast/HQ/Train"
    csv2 = "../../data/Breast/LQ/Train"
    save_dir = "../../data/Breast"
    save_csv_name = "train_merge_list.csv"
    merge_file(csv1, csv2, save_dir, save_csv_name)


if __name__ == "__main__":
    main1()
    main2()


