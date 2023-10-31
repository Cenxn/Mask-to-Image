from albumentations import *
from PIL import Image
import imgviz
import numpy as np
import os
import cv2
import imageio

IMAGE_PATH = r'F:\WarehouseOfRui\FYP\Cropped_imgs\Dataset_now_png\image'
MASK_PATH = r'F:\WarehouseOfRui\FYP\Cropped_imgs\Dataset_now_png\mask'


def resize(p=1):
    return Compose([
        Resize(height=256, width=256),
    ])


def save_colored_mask(save_path, mask):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)

def saveAugFile(aug_tech, fname):
    try:
        image = cv2.cvtColor(cv2.imread(os.path.join(IMAGE_PATH, fname + '.png')), cv2.COLOR_BGR2RGB)
    except:
        image = cv2.cvtColor(cv2.imread(os.path.join(IMAGE_PATH, fname + '.jpg')), cv2.COLOR_BGR2RGB)
    mask = np.asarray(Image.open(os.path.join(MASK_PATH, fname + '.png')), dtype=np.int32)
    # mask是使用调色板模式保存的 所以读取方式一定不能错

    augmented = aug_tech(image=image, mask=mask)
    img, msk = augmented['image'], augmented['mask']

    msk_save_pth = os.path.join(MASK_PATH, (fname.split('.')[0] + f'.png'))

    imageio.imsave(os.path.join(IMAGE_PATH, (fname.split('.')[0] + f'.png')), img)  # save img
    save_colored_mask(msk_save_pth, msk)  # save mask
    print("[Saved]", msk_save_pth)


img_ids = [id.split('.')[0] for id in os.listdir(IMAGE_PATH)]

trans_resize = resize()

for img_id in img_ids:
    saveAugFile(trans_resize, img_id)
    print('[FINISHED]IMAGE ID:{%s}' % img_id)