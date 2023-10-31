import os

PATH = r'F:\WarehouseOfRui\FYP\Cropped_imgs\SegAugDataset\selected_msk'

img_dir = os.listdir(PATH)
index = 246
for img in img_dir:
    if '-aug' in img:
        os.remove(os.path.join(PATH, img))
        print('[Finished] %s' % img)
    else:
        continue
