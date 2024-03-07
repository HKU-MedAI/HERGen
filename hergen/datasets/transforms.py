import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import cv2


def get_transforms(split: str = "train", image_size: int = 224):
    # mean = 0.471
    # std = 0.302

    if split == "train":
        image_transforms = transforms.Compose(
            [
                transforms.Resize(size=image_size + 32),
                transforms.RandomCrop(
                    size=[image_size, image_size],
                    pad_if_needed=True,
                ),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        image_transforms = transforms.Compose(
            [
                transforms.Resize(size=image_size + 32),
                transforms.CenterCrop(size=[image_size, image_size]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return image_transforms


# def get_transforms(split: str = "train", image_size: int = 512):
#     mean = 0.471
#     std = 0.302

#     # FIXME: remove normalization for biovil
#     # use albumentations for Compose and transforms
#     if split == "train":
#         image_transforms = A.Compose(
#             [
#                 # we want the long edge of the image to be resized to image_size,
#                 # and the short edge of the image to be padded to image_size on both sides,
#                 # such that the aspect ratio of the images are kept, while getting images of uniform size
#                 # (image_size x image_size)
#                 # LongestMaxSize: resizes the longer edge to image_size while maintaining the aspect ratio
#                 # INTER_AREA works best for shrinking images
#                 A.LongestMaxSize(max_size=image_size + 32,
#                                  interpolation=cv2.INTER_AREA),
#                 A.PadIfNeeded(min_height=image_size + 32, min_width=image_size + 32,
#                               border_mode=cv2.BORDER_CONSTANT),
#                 A.RandomCrop(height=image_size, width=image_size),
#                 A.Normalize(mean=mean, std=std),
#                 ToTensorV2(),
#             ]
#         )
#     else:
#         # don't apply data augmentations to val set (and test set)
#         image_transforms = A.Compose(
#             [
#                 A.LongestMaxSize(max_size=image_size,
#                                  interpolation=cv2.INTER_AREA),
#                 A.PadIfNeeded(min_height=image_size, min_width=image_size,
#                               border_mode=cv2.BORDER_CONSTANT),
#                 A.CenterCrop(height=image_size, width=image_size),
#                 A.Normalize(mean=mean, std=std),
#                 ToTensorV2(),
#             ]
#         )

#     return image_transforms
