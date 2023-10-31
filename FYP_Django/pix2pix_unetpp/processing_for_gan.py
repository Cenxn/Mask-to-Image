import base64, os
import re
from shutil import copyfile

import imgviz
from PIL import Image
import numpy as np
import cv2

"""
"_background_": 0,      "face":1,
"hair":2,               "eye":3,
"eyeblow":4,            "nose":5,
"mouth":6,              "clothes":7,
"body":8,
"""
label_rgb_dic = {
    "#000000": 0,
    "#800000": 1,
    "#008000": 2,
    "#808000": 3,
    "#000080": 4,
    "#800080": 5,
    "#008080": 6,
    "#808080": 7,
    "#400000": 8,
}


def base64_to_image(src, folder):
    """
    Decode img
    :param src: image resource base64
        eg:
            src="data:image/gif;base64,R0lGODlhMwAxAIAAAAAAAP///
                yH5BAAAAAAALAAAAAAzADEAAAK8jI+pBr0PowytzotTtbm/DTqQ6C3hGX
                ElcraA9jIr66ozVpM3nseUvYP1UEHF0FUUHkNJxhLZfEJNvol06tzwrgd
                LbXsFZYmSMPnHLB+zNJFbq15+SOf50+6rG7lKOjwV1ibGdhHYRVYVJ9Wn
                k2HWtLdIWMSH9lfyODZoZTb4xdnpxQSEF9oyOWIqp6gaI9pI1Qo7BijbF
                ZkoaAtEeiiLeKn72xM7vMZofJy8zJys2UxsCT3kO229LH1tXAAAOw=="

    :return: img name
    """
    # 1. extract img
    result = re.search("data:image/(?P<ext>.*?);base64,(?P<data>.*)", src, re.DOTALL)
    if result:
        ext = result.groupdict().get("ext")
        data = result.groupdict().get("data")

    else:
        raise Exception("Do not parse!")

    # 2. decode base64
    img = base64.urlsafe_b64decode(data)

    # 3. save img
    filename = "customized_msk.{}".format(ext)
    filepath = os.path.join(folder, filename)
    with open(filepath, "wb") as f:
        f.write(img)
    return filename


def RGB_to_Hex(rgb):
    # RGB = rgb.split(',')
    RGB = rgb
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
#     print(color)
    return color


def save_colored_mask(save_path, mask):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def save_img_as_msk(msk_p, filename):
    msk = np.asarray(Image.open(msk_p), dtype=np.int32)
    mask = np.zeros([256, 256], dtype=np.int32)
    # RGB_to_Hex(msk[1][1][:3]) in label_rgb_dic.keys()
    for i in range (256):
        for j in range (256):
            hex = RGB_to_Hex(msk[i][j][:3])
            if hex in label_rgb_dic.keys():
                mask[i][j] = label_rgb_dic[hex]
    save_colored_mask(msk_p, mask)
    combine_test_imge_with_balck_A(mask,
        os.path.join('pix2pix_unetpp/pix2pix/test_imgs/test', filename))


def combine_test_imge_with_balck_A(image_b, path_AB):
    image_A = np.zeros([256,256], dtype=np.int32)
    image_B = image_b
    image_AB = np.concatenate([image_A, image_B], 1)
    lbl_pil = Image.fromarray(image_AB.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(path_AB)


def combine_test_imge_with_colorful_A(image_a, image_b, path_AB):
    im_A = cv2.imread(image_a, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(image_b, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)


def handle_sing_msk(base64, folder_p, msk_n=None):
    if base64 is None:
        combine_test_imge_with_colorful_A('pix2pix_unetpp/pre_load.png',
                                          os.path.join(folder_p, msk_n),
                                          'pix2pix_unetpp/pix2pix/test_imgs/test/customized_msk.png')
        return 'customized_msk.png'
    else:
        filename = base64_to_image(base64, folder_p)
        # save_img_as_msk(os.path.join(folder_p, filename), filename)
        combine_test_imge_with_colorful_A('pix2pix_unetpp/pre_load.png',
                                          os.path.join(folder_p, filename),
                                          'pix2pix_unetpp/pix2pix/test_imgs/test/customized_msk.png')
    return filename


def infer_one_for_gan(output_folder):
    os.system("python ./pix2pix_unetpp/pix2pix/test.py --dataroot pix2pix_unetpp/pix2pix/test_imgs/ --name customize_pix2pix --model pix2pix --direction BtoA --results_dir pix2pix_unetpp/pix2pix/test_imgs/result/ --dataset_mode aligned")
    copyfile(r'pix2pix_unetpp/pix2pix/test_imgs/result/customize_pix2pix/test_latest/images/customized_msk_fake_B.png',
             os.path.join(output_folder, 'customized_msk_fake_B.png'))
    print('[Success]', 'customized_msk_fake_B.png')
    return 'customized_msk_fake_B.png'


# save_img_as_msk(r'F:\WarehouseOfRui\garbage\customized_msk-1.png')
