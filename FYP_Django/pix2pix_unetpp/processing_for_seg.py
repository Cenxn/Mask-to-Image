from albumentations import *
from PIL import Image
import imgviz
import numpy as np
import os
import cv2
import imageio
import argparse


# This function will rewrite the original image.
# def get_args():
#     parser = argparse.ArgumentParser(description='Process Image for Segmentation',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--input', '-i', dest='input', type=str, help='Path of input image')
#     return parser.parse_args()


def resize(p=1):
    return Compose([
        Resize(height=256, width=256),
    ])


def save_colored_mask(save_path, mask):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def resize_and_save(aug_tech, fname, new_fname):
    image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    augmented = aug_tech(image=image)
    img = augmented['image']
    imageio.imsave(new_fname, img)  # save img
    os.remove(fname)
    print(f"[Processed and Removed]{fname} new path: {new_fname}")


def handle_single_img(img_path):
    trans_resize = resize()
    dir_name, full_file_name = os.path.split(img_path)
    new_fn = f'{full_file_name.split(".")[0]}-processed.png'
    new_p = os.path.join(dir_name, new_fn)
    resize_and_save(trans_resize, img_path, new_p)
    return new_fn
