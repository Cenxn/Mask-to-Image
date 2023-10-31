import shutil
import os
import random

IMAGE_PATH  = r'./Dataset_now_png/image'
MASK_PATH   = r'Dataset_now_png/mask'
NEW_PATH    = r'F:\WarehouseOfRui\FYP\Cropped_imgs\DatasetForPix2Pix'


def create_new_path(img_dir, msk_dir, new_dicts, valid_rate=0.2):
    li_img = os.listdir(img_dir)
    random.shuffle(li_img)
    num = 0
    num_train = round(len(li_img)*(1 - valid_rate))
    for i in li_img:
        fileMaskName = i.split('.')[0] + '.png'
        if num < num_train:
            shutil.copyfile(os.path.join(img_dir, i), os.path.join(new_dicts[0], i))
            shutil.copyfile(os.path.join(msk_dir, fileMaskName), os.path.join(new_dicts[2], fileMaskName))
        else:
            shutil.copyfile(os.path.join(img_dir, i), os.path.join(new_dicts[1], i))
            shutil.copyfile(os.path.join(msk_dir, fileMaskName), os.path.join(new_dicts[3], fileMaskName))
        print("[Success]", i, 'and', fileMaskName)
        num += 1


def main():
    clses = ['images', 'masks']; file_t = ['train','val']
    new_paths = []
    for cls in clses:
        for type in file_t:
            p = os.path.join(NEW_PATH, cls, type)
            os.makedirs(p)
            new_paths.append(p)
            print("[Success Add]", p)
    create_new_path(IMAGE_PATH, MASK_PATH, new_paths)


main()