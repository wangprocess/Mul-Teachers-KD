
import albumentations as A

# These Normalize values are derived from the average of the original images

Train_Transforms = A.Compose([

    A.Normalize(mean=(0.7831249, 0.5446379, 0.5653467), std=(0.1327309, 0.14866614, 0.16493498))
])

Val_Transforms = A.Compose([
    A.Normalize(mean=(0.7831249, 0.5446379, 0.5653467), std=(0.1327309, 0.14866614, 0.16493498))
])

BUSI_Train_Transforms = A.Compose([

    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

BUSI_Val_Transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

Dermnet_Train_Transforms = A.Compose([

    A.Normalize(mean=(0.5411724, 0.41455272, 0.38252777), std=(0.25504974, 0.21076338, 0.20402928))
])

Dermnet_Val_Transforms = A.Compose([
    A.Normalize(mean=(0.5411724, 0.41455272, 0.38252777), std=(0.25504974, 0.21076338, 0.20402928))
])