import glob
import os
import cv2
import numpy as np
from PIL import Image
from albumentations import ImageOnlyTransform

# resieze the hairs image
if __name__=="__main__":
    img_Dir = "../../data/Hairs/*.png"
    save_Dir = "../../data/Hairs"
    img_path = sorted(glob.glob(img_Dir))
    img_path = [path.replace('\\', '/') for path in img_path]
    for path in img_path:
        print(path)
        img = Image.open(path)
        img_np = np.asarray(img)
        H, W, _ = img_np.shape
        print(img_np.shape)
        img = img.resize((W//2, H//2))
        save_path = os.path.join(save_Dir, os.path.basename(path)).replace("\\", "/")
        img.save(save_path)
        print("ok")
