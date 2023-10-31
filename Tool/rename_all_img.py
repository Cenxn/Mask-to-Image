import os

RENAMED_PATH    = r'F:\WarehouseOfRui\FYP\Cropped_imgs\SegAugDataset\selected_msk'
# SAVE_PATH       = r'F:\WarehouseOfRui\FYP\Cropped_imgs\SegAugDataset\re_numbered'
# expect_name     = '_test.png'  # expect file type and 后缀

# img_dir = os.listdir(RENAMED_PATH)
# index = 246
# for img in img_dir:
#     ori_p = os.path.join(RENAMED_PATH, img)
#     new_p = os.path.join(SAVE_PATH, "%d%s" % (index, expect_name))
#
#     img = cv2.imread(ori_p)
#     cv2.imwrite(new_p, img)
#     index = index + 1
#     print('[Finished] %s to %s' % (ori_p, new_p))

img_dir = os.listdir(RENAMED_PATH)
index = 246
for img in img_dir:
    new_n = f'{index}.{img.split(".")[1]}'
    os.rename(os.path.join(RENAMED_PATH, img), os.path.join(RENAMED_PATH, new_n))
    index = index + 1
    print('[Finished] %s to %s' % (img, new_n))
