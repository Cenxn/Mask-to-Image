import argparse
import logging
import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from unet import NestedUNet
from unet import UNet
from utils.dataset import BasicDataset
from utils.test_dataset import TestDataset
from config import UNetConfig
import seaborn as sns
import matplotlib.pylab as plt

cfg = UNetConfig()


def inference_one(net, image, device):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(image, cfg.scale))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        if cfg.deepsupervision:
            output = output[-1]

        if cfg.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image.size[1], image.size[0])),
                transforms.ToTensor()
            ]
        )
        
        if cfg.n_classes == 1:
            probs = tf(probs.cpu())
            mask = probs.squeeze().cpu().numpy()
            return mask > cfg.out_threshold
        else:
            masks = []
            for prob in probs:
                prob = tf(prob.cpu())
                mask = prob.squeeze().cpu().numpy()
                mask = mask > cfg.out_threshold
                masks.append(mask)
            return masks


def test_net(net, loader, device, cfg, output_dir):
    with tqdm(total=len(loader), desc='Validation round', unit='img', leave=False) as pbar:
        index = 0
        for batch in loader:
            imgs = batch['image']
            imgs = imgs.to(device=device, dtype=torch.float32)
            masks_preds = net(imgs)

            for masks_pred in masks_preds:
                pred = masks_pred
                pred = (pred > cfg.out_threshold).float()

                if cfg.deepsupervision:
                    pred = pred[-1]

                if cfg.n_classes > 1:
                    probs = F.softmax(pred, dim=1)
                else:
                    probs = torch.sigmoid(pred)

                probs = probs.squeeze(0)
                # print(probs)

                tf = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize((cfg.height, cfg.width)),
                        transforms.ToTensor()
                    ]
                )
                # print(probs.size())
                if cfg.n_classes == 1:
                    probs = tf(probs.cpu())
                    mask = probs.squeeze().cpu().numpy()
                    return mask > cfg.out_threshold
                else:
                    masks = []
                    for prob in probs:
                        prob = tf(prob.cpu())
                        mask = prob.squeeze().cpu().numpy()
                        mask = mask > cfg.out_threshold
                        masks.append(mask)
                    save_msk(masks, output_dir, index)
            index = index + 1


def save_msk(mask, output_dir, index):
    output_img_dir = osp.join(output_dir, str(index))
    os.makedirs(output_img_dir, exist_ok=True)

    if cfg.n_classes == 1:
        image_idx = Image.fromarray((mask * 255).astype(np.uint8))
        image_idx.save(osp.join(output_img_dir, index,'.png'))
    else:
        for idx in range(0, len(mask)):
            img_name_idx = str(index) + "_" + str(idx) + ".png"
            image_idx = Image.fromarray((mask[idx] * 255).astype(np.uint8))
            image_idx.save(osp.join(output_img_dir, img_name_idx))


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='./pix2pix_unetpp/unetpp/data/checkpoints/epoch_92.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', dest='input', type=str,
                        default=r'./pix2pix_unetpp/unetpp/data/input',  # ./data/test/input
                        help='Directory of input images')
    parser.add_argument('--output', '-o', dest='output', type=str, default='./pix2pix_unetpp/unetpp/data/output',
                        help='Directory of ouput images')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    input_imgs = os.listdir(args.input)

    net = eval(cfg.model)(cfg)
    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    test_dataset = TestDataset(args.input)
    test_loader = DataLoader(test_dataset, num_workers=0, pin_memory=True)
    test_net(net, test_loader, device, cfg, args.output)

    # os.system('python pix2pix_unetpp/unetpp/masks2mask.py')  # default transfer multiple single class masks into one
