import imgviz
from PIL import Image
import numpy as np
import os
import argparse

"""
"_background_": 0,      "face":1,
"hair":2,               "eye":3,
"eyeblow":4,            "nose":5,
"mouth":6,              "clothes":7,
"body":8,
"""
MASKS_PRIORITY = [7, 8, 1, 2, 3, 5, 6]

def get_args():
    parser = argparse.ArgumentParser(description='Transfer masks to singe mask',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', dest='input', type=str,
                        default='./pix2pix_unetpp/unetpp/data/output',
                        help='Directory of input images')
    parser.add_argument('--output', '-o', dest='output', type=str,
                        default='./pix2pix_unetpp/pix2pix/test_imgs/test_B',
                        help='Directory of ouput images')
    parser.add_argument('--output_for_pix2pix', type=str,
                        default='./pix2pix_unetpp/pix2pix/test_imgs/test',
                        help='Directory of gan test images')
    return parser.parse_args()


def trans_multi_msk_to_single(input_p, msk_name, output_p, output_gan):  #输入的是包括多个单个类mask的文件夹路径
    initial_msk = np.zeros([256,256], dtype=np.int32)
    masks = os.listdir(input_p)
    saved_p = os.path.join(output_p, msk_name + '.png')
    for idx in MASKS_PRIORITY:
        single_m = np.asarray(Image.open(os.path.join(input_p, masks[idx])), dtype=np.int32)
        for x in range(256):
            for y in range(256):
                if single_m[x][y] > 0:
                    initial_msk[x][y] = idx
    save_colored_mask(saved_p, initial_msk)

    # combine_test_imge_with_balck_A
    saved_pAB = os.path.join(output_gan, msk_name + '.png')
    image_A = np.zeros([256, 256], dtype=np.int32)
    image_AB = np.concatenate([image_A, initial_msk], 1)
    lbl_pil = Image.fromarray(image_AB.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(saved_pAB)


def save_colored_mask(save_path, mask):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


if __name__ == "__main__":
    args = get_args()
    MASKS_FLODER_LI = os.listdir(args.input)
    for folder in MASKS_FLODER_LI:
        folder_p = os.path.join(args.input, folder)
        trans_multi_msk_to_single(folder_p, folder, args.output, args.output_for_pix2pix)
        print('[Success]', folder_p)
