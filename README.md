# Mask to Image
Part of the code of my final year project at XJTLU. 
### Django
[Demo Video 1](https://drive.google.com/file/d/1VdG2TlYldFl5UC_1MO8U5aeQQhnyGoC7/view?usp=drive_link)
[Demo Video 2](https://drive.google.com/file/d/1fkPhVcdM5KS16IidGMye9LuGzOMffxph/view?usp=drive_link)

The Django project for final web deployment.

**pix2pix_unetpp**
Contain the code for backend logic, including code process images for both GAN model and segmentation model. 
- Original Code to train segmentation model:[Unet++ model code](https://github.com/zonasw/unet-nested-multiple-classification/tree/master) 
- Original Code to train GAN model:[pix2pix-conditionalGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

### Tool
The tool code create by myself to create dataset. The annotate data example is at `Cropper_imgs`. The original images come from [Danbooru2019](https://www.gwern.net/Crops#figures), cropped by [Anime-Face-Detector](https://github.com/qhgz2013/anime-face-detector). The image annotation tool is [labelme](https://github.com/wkentaro/labelme).

- `jsonToDataset.ipynb` The code I write to transfer the original output of LabelMe into dataset.
- `visualization.ipynb` Some data visualization method, help me write the report, also cover the content of `augmentation.ipynb`, which is used to make data augmentation.
- `MasksToMask.ipynb` Converts the output of unet++ to a single image in a simple and crude way. 
