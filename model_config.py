import os
from .utils import get_file

pre_trained_base_dir = "pretrained-weights"

def get_weight_path(init_model_name, pretrained_model_name, n_ch):
    weights_filename = None

    if pretrained_model_name == None:
        return None

    package_dir = __name__.split('.')
    base_dir = os.path.join(os.path.abspath('.'), package_dir[0], 'pretrained-weights')

    # -------------------------------------------------------
    # Classification
    # -------------------------------------------------------
    if (init_model_name, pretrained_model_name, n_ch) == ("vgg16", "imagenet", 3):
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        get_file(weights_filename, 'https://www.dropbox.com/s/gcwt6pejakfxgo3/vgg16-3ch.h5?dl=1')
    elif (init_model_name, pretrained_model_name, n_ch) == ("vgg16", "imagenet", 1):
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        get_file(weights_filename, 'https://www.dropbox.com/s/4xplve3tvskk580/vgg16-1ch.h5?dl=1')
    elif (init_model_name, pretrained_model_name, n_ch) == ("vgg16", "graynet", 1):
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        get_file(weights_filename, 'https://www.dropbox.com/s/6ncc0x9khuy4kf6/vgg16-1ch.h5?dl=1')
    elif (init_model_name, pretrained_model_name, n_ch) == ("resnet50", "imagenet", 3) :
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        get_file(weights_filename, 'https://www.dropbox.com/s/j3rh1zv1fcgvcmy/resnet50-3ch.h5?dl=1')
    elif (init_model_name, pretrained_model_name, n_ch) == ("resnet50", "imagenet", 1) :
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        get_file(weights_filename, 'https://www.dropbox.com/s/z0khv2rg6tswjoi/resnet50-1ch.h5?dl=1')
    elif (init_model_name, pretrained_model_name, n_ch) == ("resnet50", "graynet", 1) :
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        get_file(weights_filename, 'https://www.dropbox.com/s/xcaeogqh7m12h60/resnet50-1ch.h5?dl=1')
    elif (init_model_name, pretrained_model_name, n_ch) == ("inceptionv3", "imagenet", 3) :
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        get_file(weights_filename, 'https://www.dropbox.com/s/svajtvak4l2i5qu/inceptionv3-3ch.h5?dl=1')
    elif (init_model_name, pretrained_model_name, n_ch) == ("inceptionv3", "imagenet", 1) :
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        get_file(weights_filename, 'https://www.dropbox.com/s/p02t7m3w4u89fks/inceptionv3-1ch.h5?dl=1')
    elif (init_model_name, pretrained_model_name, n_ch) == ("inceptionv3", "graynet", 1) :
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        get_file(weights_filename, 'https://www.dropbox.com/s/bojdzs0rs4rxroc/inceptionv3-1ch.h5?dl=1')
    elif (init_model_name, pretrained_model_name, n_ch) == ("inception_resnetv2", "imagenet", 3) :
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        get_file(weights_filename, 'https://www.dropbox.com/s/zceajyuwzgzeuj7/inception_resnetv2-3ch.h5?dl=1')
    elif (init_model_name, pretrained_model_name, n_ch) == ("inception_resnetv2", "imagenet", 1) :
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        get_file(weights_filename, 'https://www.dropbox.com/s/dihjzzsjwake80f/inception_resnetv2-1ch.h5?dl=1')
    elif (init_model_name, pretrained_model_name, n_ch) == ("inception_resnetv2", "graynet", 1) :
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        get_file(weights_filename, 'https://www.dropbox.com/s/olyh8xymj5d44b5/inception_resnetv2-1ch.h5?dl=1')
    elif (init_model_name, pretrained_model_name, n_ch) == ("densnet121", "imagenet", 3):
        # weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        raise NotImplementedError
    elif (init_model_name, pretrained_model_name, n_ch) == ("densnet121", "imagenet", 1):
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        raise NotImplementedError
        get_file(weights_filename, '')
    elif (init_model_name, pretrained_model_name, n_ch) == ("densnet121", "graynet", 1):
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "{}-{}ch.h5".format(init_model_name, n_ch))
        get_file(weights_filename, 'https://www.dropbox.com/s/ly6p4obm8xq56hu/densnet121-1ch.h5?dl=1')
    #-------------------------------------------------------
    # Segmentation
    #-------------------------------------------------------
    #----- UNET (only encoder initialized with unetenc pretrained model)
    # # ImageNet pretrained model for Grayscale
    elif (init_model_name, pretrained_model_name, n_ch) == ("unetenc", "imagenet", 3):  ## Not trainned from ImageNet data
        raise NotImplementedError
    elif (init_model_name, pretrained_model_name, n_ch) == ("unetenc", "imagenet", 1): ## Not trainned from ImageNet data
        raise NotImplementedError
    elif (init_model_name, pretrained_model_name, n_ch) == ("unetenc", "graynet", 1):
        weights_filename = os.path.join(base_dir, pretrained_model_name, init_model_name, "unetenc-1ch.h5")
        get_file(weights_filename, 'https://www.dropbox.com/s/3orvp43a9w00rst/unetenc-1ch.h5?dl=1')
    #----- UNET_VGG16 (only encoder initialized)
    # ImageNet pretrained model for RGB
    elif (init_model_name, pretrained_model_name, n_ch) == ("unetvgg16", "imagenet", 3):
        weights_filename = os.path.join(base_dir, pretrained_model_name, "vgg16", "vgg16-3ch.h5")
        get_file(weights_filename, 'https://www.dropbox.com/s/gcwt6pejakfxgo3/vgg16-3ch.h5?dl=1')
    # ImageNet pretrained model for Grayscale
    elif (init_model_name, pretrained_model_name, n_ch) == ("unetvgg16", "imagenet", 1):
        weights_filename = os.path.join(base_dir, pretrained_model_name, "vgg16", "vgg16-1ch.h5")
        get_file(weights_filename, 'https://www.dropbox.com/s/4xplve3tvskk580/vgg16-1ch.h5?dl=1')
    # GrayNet pretrained model for Grayscale
    elif (init_model_name, pretrained_model_name, n_ch) == ("unetvgg16", "graynet", 1):
        weights_filename = os.path.join(base_dir, pretrained_model_name, "vgg16", "vgg16-1ch.h5")
        get_file(weights_filename, 'https://www.dropbox.com/s/6ncc0x9khuy4kf6/vgg16-1ch.h5?dl=1')
    #----- UNET_RESNET50 (only encoder initialized)
    # ImageNet pretrained model for RGB
    elif (init_model_name, pretrained_model_name, n_ch) == ("unetresnet50", "imagenet", 3):
        weights_filename = os.path.join(base_dir, pretrained_model_name, "resnet50", "resnet50-3ch.h5")
        get_file(weights_filename, 'https://www.dropbox.com/s/j3rh1zv1fcgvcmy/resnet50-3ch.h5?dl=1')
    # ImageNet pretrained model for Grayscale
    elif (init_model_name, pretrained_model_name, n_ch) == ("unetresnet50", "imagenet", 1):
        weights_filename = os.path.join(base_dir, pretrained_model_name, "resnet50", "resnet50-1ch.h5")
        get_file(weights_filename, 'https://www.dropbox.com/s/z0khv2rg6tswjoi/resnet50-1ch.h5?dl=1')
    # GrayNet pretrained model for Grayscale
    elif (init_model_name, pretrained_model_name, n_ch) == ("unetresnet50", "graynet", 1):
        weights_filename = os.path.join(base_dir, pretrained_model_name, "resnet50", "resnet50-1ch.h5")
        get_file(weights_filename, 'https://www.dropbox.com/s/xcaeogqh7m12h60/resnet50-1ch.h5?dl=1')
    else:
        print("[ERROR]", "INIT MODEL NAME:", init_model_name, "pretrained_model_name:", pretrained_model_name, "nch:", n_ch)
        raise NotImplementedError

    weights_fullpath = os.path.join(base_dir, pre_trained_base_dir, weights_filename)

    return weights_fullpath
