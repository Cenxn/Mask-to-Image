from django.shortcuts import render,HttpResponse
from django.conf import settings
import os
from pix2pix_unetpp.processing_for_seg import handle_single_img
from pix2pix_unetpp.processing_for_gan import handle_sing_msk,infer_one_for_gan

MAX_SIZE = 1024 * 1024 * 10


# Create your views here.
def sgementation(request):
    if request.method == 'GET':
        return render(request, 'index.html')

    # nid = str(uuid.uuid4())  # add a random code to prevent renaming

    try:
        img = request.FILES.get('up_img_to_seg')
        if img.size > MAX_SIZE:
            return render(request, 'index.html',{"error_msg" : "Please upload image smaller than 10 MB."})
    except AttributeError:
        return render(request, 'index.html',{"error_msg" : "Please upload at least a image."})

    img_name = img.name
    # file suffixes and filenames
    # mobile = os.path.splitext(img_name)[0]
    ext = os.path.splitext(img_name)[1]

    accept_type = [".png", ".jpeg", ".jpg"]
    if ext not in accept_type:
        return render(request, 'index.html', {"error_msg": "Please upload correct type (png/jpg/jpeg)."})

    # Redefine file names
    img_name = f'upload-img{ext}'
    # Load save path from configuration
    img_path = os.path.join(settings.IMG_UPLOAD_SEG, img_name)
    with open(img_path, 'ab') as fp:
        for chunk in img.chunks():
            fp.write(chunk)
    fp.close()

    n_fname = handle_single_img(img_path)
    os.system(f'python pix2pix_unetpp/unetpp/inference.py -i {settings.IMG_UPLOAD_SEG}')
    print('[Segmentation Success]')
    os.system(f'python pix2pix_unetpp/unetpp/masks2mask.py -o {settings.IMG_OUTPUT_SEG} '
              f'--output_for_pix2pix {settings.IMG_UPLOAD_GAN}')
    print('[Transfer Mask Success]')

    # Pass to front-end
    img_info = {
        'original_img': os.path.join(settings.IMG_RELATED_SEG, n_fname),
        'mask': os.path.join(settings.IMG_OUTPUT_SEG_R, '0.png'),
    }
    return render(request, 'index.html',{"img_paths": img_info})


def pix2pix(request):
    if request.method == 'GET':
        return render(request, 'pix2pix.html')
    if request.POST.get('canvas_base64'):
        fname = handle_sing_msk(request.POST.get('canvas_base64'), settings.IMG_UPLOAD_GAN)  # for html os.path.join(settings.IMG_RELATED_GAN, fname)
        img_name = infer_one_for_gan(settings.IMG_OUTPUT_GAN)

        # Pass to front-end
        img_info = {
            'upload_image': os.path.join(settings.IMG_RELATED_GAN, fname),
            'output_img': os.path.join(settings.IMG_OUTPUT_GAN_R, img_name),
        }

        return render(request, 'pix2pix.html', {'imgs_info': img_info})

    if request.FILES.get('up_img_to_gan'):
        try:
            img = request.FILES.get('up_img_to_gan')
            if img.size > MAX_SIZE:
                return render(request, 'pix2pix.html', {"error_msg": "Please upload image smaller than 10 MB."})
        except AttributeError:
            return render(request, 'pix2pix.html', {"error_msg": "Please upload at least a image."})

        img_name = img.name
        ext = os.path.splitext(img_name)[1]

        accept_type = [".png", ".jpeg", ".jpg"]
        if ext not in accept_type:
            return render(request, 'pix2pix.html', {"error_msg": "Please upload correct type (png/jpg/jpeg)."})

        # Redefine file names
        img_name = f'customized_msk{ext}'
        # Load save path from configuration
        img_path = os.path.join(settings.IMG_UPLOAD_GAN, img_name)

        os.remove(img_path)

        with open(img_path, 'ab') as fp:
            for chunk in img.chunks():
                fp.write(chunk)
        fp.close()

        n_fname = handle_sing_msk(base64=None, folder_p=settings.IMG_UPLOAD_GAN, msk_n=img_name)
        out_img = infer_one_for_gan(settings.IMG_OUTPUT_GAN)

        # Pass to front-end
        img_info = {
            'upload_image': os.path.join(settings.IMG_RELATED_GAN, img_name),
            'output_img': os.path.join(settings.IMG_OUTPUT_GAN_R, out_img),
        }
        return render(request, 'pix2pix.html', {"imgs_info": img_info})