from keras import layers
from keras.layers import Concatenate, Input, \
    Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, UpSampling2D, \
    AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.densenet import DenseNet121
from .model_config import *

#-------------------------------------
# U-Net for segmentation
#-------------------------------------
class UNET():
    def __init__(self, num_chs, w_reg, *args, **kwargs):
        self.w_reg = w_reg
        self.num_chs = num_chs

    def __call__(self, input_tensor):
        ## Unet
        dn5_conv2, bypass1, bypass2, bypass3, bypass4 = self.encoder(input_tensor, self.num_chs, self.w_reg) ## encoder
        predict = self.decoder(dn5_conv2, self.num_chs, bypass1, bypass2, bypass3, bypass4, self.w_reg) ## decoder
        return predict

    @staticmethod
    def convblock(x, filters, kernel_size, padding="same", strides=(1, 1), w_reg=None, BN=True, name=""):
        bn_name = name + "_bn"
        conv_name = name + "_conv"
        relu_name = name + "_relu"

        bn_axis = 3
        # post-activation normalization
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=True,
                   kernel_regularizer=w_reg,
                   name=conv_name)(x)
        if BN:
            x = BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = Activation("relu", name=relu_name)(x)
        return x

    @staticmethod
    def encoder(input_tensor, num_chs, w_reg):
        convblock = UNET.convblock
        # -------------------------------------
        # Down-sampling
        # -------------------------------------
        dn1_conv1 = convblock(input_tensor, num_chs, (3, 3), w_reg=w_reg, name="dn1_conv1")
        dn1_conv2 = convblock(dn1_conv1, num_chs, (3, 3), w_reg=w_reg, name="dn1_conv2")
        dn1_pool = MaxPooling2D((2, 2), strides=(2, 2), name="dn1_pool")(dn1_conv2)

        dn2_conv1 = convblock(dn1_pool, num_chs * 2, (3, 3), w_reg=w_reg, name="dn2_conv1")
        dn2_conv2 = convblock(dn2_conv1, num_chs * 2, (3, 3), w_reg=w_reg, name="dn2_conv2")
        dn2_pool = MaxPooling2D((2, 2), strides=(2, 2), name="dn2_pool")(dn2_conv2)

        dn3_conv1 = convblock(dn2_pool, num_chs * 4, (3, 3), w_reg=w_reg, name="dn3_conv1")
        dn3_conv2 = convblock(dn3_conv1, num_chs * 4, (3, 3), w_reg=w_reg, name="dn3_conv2")
        dn3_pool = MaxPooling2D((2, 2), strides=(2, 2), name="dn3_pool")(dn3_conv2)

        dn4_conv1 = convblock(dn3_pool, num_chs * 8, (3, 3), w_reg=w_reg, name="dn4_conv1")
        dn4_conv2 = convblock(dn4_conv1, num_chs * 8, (3, 3), w_reg=w_reg, name="dn4_conv2")
        dn4_conv2 = Dropout(0.5)(dn4_conv2)
        dn4_pool = MaxPooling2D((2, 2), strides=(2, 2), name="dn4_pool")(dn4_conv2)

        # -------------------------------------
        # Middle
        # -------------------------------------
        dn5_conv1 = convblock(dn4_pool, num_chs * 16, (3, 3), w_reg=w_reg, name="dn5_conv1")
        dn5_conv2 = convblock(dn5_conv1, num_chs * 16, (3, 3), w_reg=w_reg, name="dn5_conv2")
        dn5_conv2 = Dropout(0.5)(dn5_conv2)
        last_conv = dn5_conv2

        return last_conv, dn4_conv2, dn3_conv2, dn2_conv2, dn1_conv2

    @staticmethod
    def decoder(input_tensor, num_chs, bypass1, bypass2, bypass3, bypass4, w_reg):
        convblock = UNET.convblock
        dn1_conv2 = bypass4
        dn2_conv2 = bypass3
        dn3_conv2 = bypass2
        dn4_conv2 = bypass1
        dn5_conv2 = input_tensor
        # -------------------------------------
        # Up-sampling
        # -------------------------------------
        up6 = UpSampling2D(size=(2, 2), name="up6")(dn5_conv2)
        up6 = convblock(up6, num_chs * 8, (3, 3), w_reg=w_reg, name="up6_deconv")
        up6_concat = layers.concatenate([dn4_conv2, up6], axis=3, name="up6_concat")
        up6_conv1 = convblock(up6_concat, num_chs * 8, (3, 3), w_reg=w_reg, name="up6_conv1")
        up6_conv2 = convblock(up6_conv1, num_chs * 8, (3, 3), w_reg=w_reg, name="up6_conv2")

        up7 = UpSampling2D(size=(2, 2), name="up7")(up6_conv2)
        up7 = convblock(up7, num_chs * 4, (3, 3), w_reg=w_reg, name="up7_deconv")
        up7_concat = layers.concatenate([dn3_conv2, up7], axis=3, name="up7_concat")
        up7_conv1 = convblock(up7_concat, num_chs * 4, (3, 3), w_reg=w_reg, name="up7_conv1")
        up7_conv2 = convblock(up7_conv1, num_chs * 4, (3, 3), w_reg=w_reg, name="up7_conv2")

        up8 = UpSampling2D(size=(2, 2), name="up8")(up7_conv2)
        up8 = convblock(up8, num_chs * 2, (3, 3), w_reg=w_reg, name="up8_deconv")
        up8_concat = layers.concatenate([dn2_conv2, up8], axis=3, name="up8_concat")
        up8_conv1 = convblock(up8_concat, num_chs * 2, (3, 3), w_reg=w_reg, name="up8_conv1")
        up8_conv2 = convblock(up8_conv1, num_chs * 2, (3, 3), w_reg=w_reg, name="up8_conv2")

        up9 = UpSampling2D(size=(2, 2), name="up9")(up8_conv2)
        up9 = convblock(up9, num_chs, (3, 3), w_reg=w_reg, name="up9_deconv")
        up9_concat = layers.concatenate([dn1_conv2, up9], axis=3, name="up9_concat")
        up9_conv1 = convblock(up9_concat, num_chs, (3, 3), w_reg=w_reg, name="up9_conv1")
        up9_conv2 = convblock(up9_conv1, num_chs, (3, 3), w_reg=w_reg, name="up9_conv2")

        final10_conv1 = convblock(up9_conv2, 2, (3, 3), w_reg=w_reg, name="final10_conv1")
        final10_conv2 = Conv2D(1, (1, 1), strides=(1, 1), padding="same", use_bias=True, kernel_regularizer=w_reg,
                               name="final10_conv2_conv")(final10_conv1)
        final10_conv2 = BatchNormalization(axis=3, name="final10_conv2_bn")(final10_conv2)
        predict = Activation("sigmoid", name="predict")(final10_conv2)

        return predict


def RESNET50_LMIC(input_tensor, w_reg, SEPARATE_ACT=False, return_tensor=False):
    def res_block(input_tensor, kernel_size, filters, stage, block, w_reg=None):
        filters1, filters2, filters3 = filters
        bn_axis = 3
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"
        relu_name_base = "act" + str(stage) + block

        x = Conv2D(filters1, (1, 1), name=conv_name_base + "2a", kernel_regularizer=w_reg)(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
        x = Activation("relu", name=relu_name_base + "_branch2a_relu")(x)

        x = Conv2D(filters2, kernel_size, padding="same", name=conv_name_base + "2b", kernel_regularizer=w_reg)(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
        x = Activation("relu", name=relu_name_base + "_branch2b_relu")(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + "2c", kernel_regularizer=w_reg)(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

        x = layers.add([x, input_tensor])
        x = Activation("relu", name=relu_name_base + "_relu")(x)
        return x

    def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), w_reg=None):
        filters1, filters2, filters3 = filters
        bn_axis = 3
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"
        relu_name_base = "act" + str(stage) + block

        x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + "2a", kernel_regularizer=w_reg)(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
        x = Activation("relu", name=relu_name_base + "_branch2a_relu")(x)

        x = Conv2D(filters2, kernel_size, padding="same", name=conv_name_base + "2b", kernel_regularizer=w_reg)(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
        x = Activation("relu", name=relu_name_base + "_branch2b_relu")(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + "2c", kernel_regularizer=w_reg)(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + "1", kernel_regularizer=w_reg)(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + "1")(shortcut)

        x = layers.add([x, shortcut])
        x = Activation("relu", name=relu_name_base + "_relu")(x)
        return x

    bn_axis = 3
    x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="conv1", kernel_regularizer=w_reg)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name="bn_conv1")(x)
    x = Activation("relu", name="conv1_relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name="pool1")(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1), w_reg=w_reg)
    x = res_block(x, 3, [64, 64, 256], stage=2, block="b", w_reg=w_reg)
    x = res_block(x, 3, [64, 64, 256], stage=2, block="c", w_reg=w_reg)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block="a", w_reg=w_reg)
    x = res_block(x, 3, [128, 128, 512], stage=3, block="b", w_reg=w_reg)
    x = res_block(x, 3, [128, 128, 512], stage=3, block="c", w_reg=w_reg)
    x = res_block(x, 3, [128, 128, 512], stage=3, block="d", w_reg=w_reg)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a", w_reg=w_reg)
    x = res_block(x, 3, [256, 256, 1024], stage=4, block="b", w_reg=w_reg)
    x = res_block(x, 3, [256, 256, 1024], stage=4, block="c", w_reg=w_reg)
    x = res_block(x, 3, [256, 256, 1024], stage=4, block="d", w_reg=w_reg)
    x = res_block(x, 3, [256, 256, 1024], stage=4, block="e", w_reg=w_reg)
    x = res_block(x, 3, [256, 256, 1024], stage=4, block="f", w_reg=w_reg)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a", w_reg=w_reg)
    x = res_block(x, 3, [512, 512, 2048], stage=5, block="b", w_reg=w_reg)
    x = res_block(x, 3, [512, 512, 2048], stage=5, block="c", w_reg=w_reg)

    gap = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    if return_tensor:
        return gap
    else:
        # Create model
        model = Model(input_tensor, x, name="resnet50")
        #print model.summary()
        return model


def INCEPTIONV3_LMIC(input_tensor, w_reg, SEPARATE_ACT=False, return_tensor=False):
    def conv2d_bn(x, filters, num_row, num_col, padding="same", strides=(1, 1), w_reg=None, name=""):
        bn_name = name + "_bn"
        conv_name = name + "_conv"
        relu_name = name + "_relu"

        bn_axis = 3
        x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False, kernel_regularizer=w_reg, name=conv_name)(x)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Activation("relu", name=relu_name)(x)
        return x

    channel_axis = 3

    #-------------------------------------
    # First Block
    #-------------------------------------
    x = conv2d_bn(input_tensor, 32, 3, 3, strides=(2, 2), w_reg=w_reg, name="stg0_blk0") # 512x512xC -> 256x256x32
    x = conv2d_bn(x, 32, 3, 3, w_reg=w_reg, name="stg0_blk1") # 256x256x32 -> 256x256x32
    x = conv2d_bn(x, 64, 3, 3, w_reg=w_reg, name="stg0_blk2") # 256x256x32 -> 256x256x64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="stg0_blk3_pool")(x) # 256x256x32 -> 128x128x64

    x = conv2d_bn(x, 80, 1, 1, w_reg=w_reg, name="stg1_blk0") # 128x128x64 -> 128x128x80
    x = conv2d_bn(x, 192, 3, 3, w_reg=w_reg, name="stg1_blk1") # 128x128x80 -> 128x128x192
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="stg1_blk2_pool")(x) # 128x128x192 -> 64x64x192
    #------------ 64x64x192 ------------

    #-------------------------------------
    # mixed0 : 1x1 + 1x1_5x5 + 1x1_3x3_3x3 + pool
    #-------------------------------------
    branch1x1 = conv2d_bn(x, 64, 1, 1, w_reg=w_reg, name="mixed0_brnch1x1_blk0") # 64x64x192 -> 64x64x64

    branch5x5 = conv2d_bn(x, 48, 1, 1, w_reg=w_reg, name="mixed0_brnch5x5_blk0") # 64x64x192 -> 64x64x48
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, w_reg=w_reg, name="mixed0_brnch5x5_blk1") # 64x64x48 -> 64x64x64

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, w_reg=w_reg, name="mixed0_brnch3x3dbl_blk0") # 64x64x192 -> 64x64x64
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, w_reg=w_reg, name="mixed0_brnch3x3dbl_blk1") # 64x64x64 -> 64x64x96
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, w_reg=w_reg, name="mixed0_brnch3x3dbl_blk2") # 64x64x96 -> 64x64x96

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same", name="mixed0_brnchpool_blk0_pool")(x) # 64x64x192 -> 64x64x192
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, w_reg=w_reg, name="mixed0_brnchpool_blk1") # 64x64x192 -> 64x64x32

    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name="mixed0")
    #------------ 64x64x256 ------------

    #-------------------------------------
    # mixed1 : 1x1 + 1x1_5x5 + 1x1_3x3_3x3 + pool
    #-------------------------------------
    branch1x1 = conv2d_bn(x, 64, 1, 1, w_reg=w_reg, name="mixed1_brnch1x1_blk0") # 64x64x256 -> 64x64x64

    branch5x5 = conv2d_bn(x, 48, 1, 1, w_reg=w_reg, name="mixed1_brnch5x5_blk0") # 64x64x256 -> 64x64x48
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, w_reg=w_reg, name="mixed1_brnch5x5_blk1") # 64x64x48 -> 64x64x64

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, w_reg=w_reg, name="mixed1_brnch3x3dbl_blk0") # 64x64x256 -> 64x64x64
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, w_reg=w_reg, name="mixed1_brnch3x3dbl_blk1") # 64x64x64 -> 64x64x96
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, w_reg=w_reg, name="mixed1_brnch3x3dbl_blk2") # 64x64x96 -> 64x64x96

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same", name="mixed1_brnchpool_blk0_pool")(x) # 64x64x256 -> 64x64x256
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, w_reg=w_reg, name="mixed1_brnchpool_blk1") # 64x64x256 -> 64x64x64
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name="mixed1")
    #------------ 64x64x288 ------------

    #-------------------------------------
    # mixed2 : 1x1 + 1x1_5x5 + 1x1_3x3_3x3 + pool
    #-------------------------------------
    branch1x1 = conv2d_bn(x, 64, 1, 1, w_reg=w_reg, name="mixed2_brnch1x1_blk0") # 64x64x288 -> 64x64x64

    branch5x5 = conv2d_bn(x, 48, 1, 1, w_reg=w_reg, name="mixed2_brnch5x5_blk0") # 64x64x288 -> 64x64x48
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, w_reg=w_reg, name="mixed2_brnch5x5_blk1") # 64x64x48 -> 64x64x64

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, w_reg=w_reg, name="mixed2_brnch3x3dbl_blk0") # 64x64x288 -> 64x64x64
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, w_reg=w_reg, name="mixed2_brnch3x3dbl_blk1") # 64x64x64 -> 64x64x96
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, w_reg=w_reg, name="mixed2_brnch3x3dbl_blk2") # 64x64x96 -> 64x64x96

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same", name="mixed2_brnchpool_blk0_pool")(x) # 64x64x288 -> 64x64x288
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, w_reg=w_reg, name="mixed2_brnchpool_blk1") # 64x64x288 -> 64x64x64
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name="mixed2")
    #------------ 64x64x288 ------------

    #-------------------------------------
    # mixed3 : 3x3 + 1x1_3x3_3x3 + pool
    #-------------------------------------
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), w_reg=w_reg, name="mixed3_brnch3x3_blk0") # 64x64x288 -> 32x32x384

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, w_reg=w_reg, name="mixed3_brnch3x3dbl_blk0") # 64x64x288 -> 64x64x64
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, w_reg=w_reg, name="mixed3_brnch3x3dbl_blk1") # 64x64x64 -> 64x64x96
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), w_reg=w_reg, name="mixed3_brnch3x3dbl_blk2") # 64x64x96 -> 32x32x96

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="mixed3_brnchpool_blk0_pool")(x) # 64x64x288 -> 32x32x288
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name="mixed3")
    #------------ 32x32x768 ------------

    #-------------------------------------
    # mixed4 : 1x1 + 1x1_1x7_7x1 + 1x1_7x1_1x7_7x1_1x7 + pool
    #-------------------------------------
    branch1x1 = conv2d_bn(x, 192, 1, 1, w_reg=w_reg, name="mixed4_brnch1x1_blk0") # 32x32x768 -> 32x32x192

    branch7x7 = conv2d_bn(x, 128, 1, 1, w_reg=w_reg, name="mixed4_brnch7x7_blk0") # 32x32x768 -> 32x32x128
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7, w_reg=w_reg, name="mixed4_brnch7x7_blk1") # 32x32 -> 32x32x128
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, w_reg=w_reg, name="mixed4_brnch7x7_blk2") # 32x32 -> 32x32x192

    branch7x7dbl = conv2d_bn(x, 128, 1, 1, w_reg=w_reg, name="mixed4_brnch7x7dbl_blk0") # 32x32x768 -> 32x32x128
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, w_reg=w_reg, name="mixed4_brnch7x7dbl_blk1") # 32x32x128 -> 32x32x128
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7, w_reg=w_reg, name="mixed4_brnch7x7dbl_blk2") # 32x32x128 -> 32x32x128
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, w_reg=w_reg, name="mixed4_brnch7x7dbl_blk3") # 32x32x128 -> 32x32x128
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, w_reg=w_reg, name="mixed4_brnch7x7dbl_blk4") # 32x32x128 -> 32x32x192

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same", name="mixed4_brnchpool_blk0_pool")(x) # 32x32x768 -> 32x32x768
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, w_reg=w_reg, name="mixed4_brnchpool_blk1") # 32x32x768 -> 32x32x192
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name="mixed4")
    #------------ 32x32x768 ------------

    #-------------------------------------
    # mixed5 : 1x1 + 1x1_1x7_7x1 + 1x1_7x1_1x7_7x1_1x7 + pool
    # mixed6 : 1x1 + 1x1_1x7_7x1 + 1x1_7x1_1x7_7x1_1x7 + pool
    #-------------------------------------
    for i in range(2):
        num = 5 + i
        branch1x1 = conv2d_bn(x, 192, 1, 1, w_reg=w_reg, name="mixed{}_brnch1x1_blk0".format(num))

        branch7x7 = conv2d_bn(x, 160, 1, 1, w_reg=w_reg, name="mixed{}_brnch7x7_blk0".format(num))
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7, w_reg=w_reg, name="mixed{}_brnch7x7_blk1".format(num))
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, w_reg=w_reg, name="mixed{}_brnch7x7_blk2".format(num))

        branch7x7dbl = conv2d_bn(x, 160, 1, 1, name="mixed{}_brnch7x7dbl_blk0".format(num))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, w_reg=w_reg, name="mixed{}_brnch7x7dbl_blk1".format(num))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7, w_reg=w_reg, name="mixed{}_brnch7x7dbl_blk2".format(num))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, w_reg=w_reg, name="mixed{}_brnch7x7dbl_blk3".format(num))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, w_reg=w_reg, name="mixed{}_brnch7x7dbl_blk4".format(num))

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same", name="mixed{}_brnchpool_blk0_pool".format(num))(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, w_reg=w_reg, name="mixed{}_brnchpool_blk1".format(num))
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name="mixed" + str(5 + i))
    #------------ 32x32x768 ------------

    #-------------------------------------
    # mixed7 : 1x1 + 1x1_1x7_7x1 + 1x1_7x1_1x7_7x1_1x7 + pool
    #-------------------------------------
    branch1x1 = conv2d_bn(x, 192, 1, 1, w_reg=w_reg, name="mixed7_brnch1x1_blk0") # 32x32x768 -> 32x32x192

    branch7x7 = conv2d_bn(x, 192, 1, 1, w_reg=w_reg, name="mixed7_brnch7x7_blk0") # 32x32x768 -> 32x32x192
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7, w_reg=w_reg, name="mixed7_brnch7x7_blk1") # 32x32x192 -> 32x32x192
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, w_reg=w_reg, name="mixed7_brnch7x7_blk2") # 32x32x192 -> 32x32x192

    branch7x7dbl = conv2d_bn(x, 192, 1, 1, w_reg=w_reg, name="mixed7_brnch7x7dbl_blk0") # 32x32x768 -> 32x32x192
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1, w_reg=w_reg, name="mixed7_brnch7x7dbl_blk1") # 32x32x192 -> 32x32x192
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, w_reg=w_reg, name="mixed7_brnch7x7dbl_blk2") # 32x32x192 -> 32x32x192
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1, w_reg=w_reg, name="mixed7_brnch7x7dbl_blk3") # 32x32x192 -> 32x32x192
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, w_reg=w_reg, name="mixed7_brnch7x7dbl_blk4") # 32x32x192 -> 32x32x192

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same", name="mixed7_brnchpool_blk0_pool")(x) # 32x32x768 -> 32x32x768
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, w_reg=w_reg, name="mixed7_brnchpool_blk1") # 32x32x768 -> 32x32x192
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name="mixed7")
    #------------ 32x32x768 ------------

    #-------------------------------------
    # mixed8 : 1x1_3x3 + 1x1_1x7_7x1_3x3 + pool
    #-------------------------------------
    branch3x3 = conv2d_bn(x, 192, 1, 1, w_reg=w_reg, name="mixed8_brnch3x3_blk0") # 32x32 -> 32x32
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), w_reg=w_reg, name="mixed8_brnch3x3_blk1") # 32x32 -> 14x14x320

    branch7x7x3 = conv2d_bn(x, 192, 1, 1, w_reg=w_reg, name="mixed8_brnch7x7x3_blk0") # 32x32 -> 32x32
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7, w_reg=w_reg, name="mixed8_brnch7x7x3_blk1") # 32x32 -> 32x32
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1, w_reg=w_reg, name="mixed8_brnch7x7x3_blk2") # 32x32 -> 32x32
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), w_reg=w_reg, name="mixed8_brnch7x7x3_blk3") # 32x32 -> 14x14x192

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="mixed8_brnchpool_blk0_pool")(x) # 32x32 -> 14x14x768
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name="mixed8")
    #------------ 14x14x1280 ------------

    #-------------------------------------
    # mixed9  : 1x1 + 1x1_(1x3+3x1) + 1x1_3x3_(1x3+3x1) + pool
    # mixed10 : 1x1 + 1x1_(1x3+3x1) + 1x1_3x3_(1x3+3x1) + pool
    #-------------------------------------
    for i in range(2):
        num = 9 + i
        branch1x1 = conv2d_bn(x, 320, 1, 1, w_reg=w_reg, name="mixed{}_brnch1x1_blk0".format(num))

        branch3x3 = conv2d_bn(x, 384, 1, 1, w_reg=w_reg, name="mixed{}_brnch3x3_blk0".format(num))
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3, w_reg=w_reg, name="mixed{}_brnch3x3_blk1".format(num))
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1, w_reg=w_reg, name="mixed{}_brnch3x3_blk2".format(num))
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=channel_axis, name="mixed{}_brnch3x3_blk3_concat".format(num))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1, w_reg=w_reg, name="mixed{}_brnch3x3dbl_blk0".format(num))
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3, w_reg=w_reg, name="mixed{}_brnch3x3dbl_blk1".format(num))
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3, w_reg=w_reg, name="mixed{}_brnch3x3dbl_blk2".format(num))
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1, w_reg=w_reg, name="mixed{}_brnch3x3dbl_blk3".format(num))
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis, name="mixed{}_brnch3x3dbl_blk4_concat".format(num))

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same", name="mixed{}_brnchpool_blk0_pool".format(num))(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, w_reg=w_reg, name="mixed{}_brnchpool_blk1".format(num))
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name="mixed" + str(9 + i))
    #------------ 16x16x2048 ------------

    gap = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    if return_tensor:
        return gap
    else:
        # Create model
        model = Model(input_tensor, x, name="inceptionv3")
        #print model.summary()

        return model


def INCEPTION_RESNET_V2_LMIC(input_tensor, w_reg, SEPARATE_ACT=False, return_tensor=False):
    def conv2d_bn(x, filters, kernel_size, strides=1, padding="same", activation="relu", use_bias=False, w_reg=None, name=None):

        ### Convolution
        conv_name = name + "_conv"
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, kernel_regularizer=w_reg,  name=conv_name)(x)

        ### Batch Normalization
        if not use_bias:
            bn_axis = 3
            bn_name = name + "_bn"
            x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

        ### Activation
        if activation is not None:
            relu_name = name + "_relu"
            x = Activation(activation, name=relu_name)(x)
        return x

    def inception_resnet_block(x, scale, block_type, block_idx, w_reg=None, activation="relu"):
        if block_type == "block35":
            branch_0 = conv2d_bn(x, 32, 1, w_reg=w_reg, name="{}_{}_brnch0_blk0".format(block_type, block_idx))
            branch_1 = conv2d_bn(x, 32, 1, w_reg=w_reg, name="{}_{}_brnch1_blk0".format(block_type, block_idx))
            branch_1 = conv2d_bn(branch_1, 32, 3, w_reg=w_reg, name="{}_{}_brnch1_blk1".format(block_type, block_idx))
            branch_2 = conv2d_bn(x, 32, 1, w_reg=w_reg, name="{}_{}_brnch2_blk0".format(block_type, block_idx))
            branch_2 = conv2d_bn(branch_2, 48, 3, w_reg=w_reg, name="{}_{}_brnch2_blk1".format(block_type, block_idx))
            branch_2 = conv2d_bn(branch_2, 64, 3, w_reg=w_reg, name="{}_{}_brnch2_blk2".format(block_type, block_idx))
            branches = [branch_0, branch_1, branch_2]
        elif block_type == "block17":
            branch_0 = conv2d_bn(x, 192, 1, w_reg=w_reg, name="{}_{}_brnch0_blk0".format(block_type, block_idx))
            branch_1 = conv2d_bn(x, 128, 1, w_reg=w_reg, name="{}_{}_brnch1_blk0".format(block_type, block_idx))
            branch_1 = conv2d_bn(branch_1, 160, [1, 7], w_reg=w_reg, name="{}_{}_brnch1_blk1".format(block_type, block_idx))
            branch_1 = conv2d_bn(branch_1, 192, [7, 1], w_reg=w_reg, name="{}_{}_brnch1_blk2".format(block_type, block_idx))
            branches = [branch_0, branch_1]
        elif block_type == "block8":
            branch_0 = conv2d_bn(x, 192, 1, w_reg=w_reg, name="{}_{}_brnch0_blk0".format(block_type, block_idx))
            branch_1 = conv2d_bn(x, 192, 1, w_reg=w_reg, name="{}_{}_brnch1_blk0".format(block_type, block_idx))
            branch_1 = conv2d_bn(branch_1, 224, [1, 3], w_reg=w_reg, name="{}_{}_brnch1_blk1".format(block_type, block_idx))
            branch_1 = conv2d_bn(branch_1, 256, [3, 1], w_reg=w_reg, name="{}_{}_brnch1_blk2".format(block_type, block_idx))
            branches = [branch_0, branch_1]

        channel_axis = 3
        block_name = "{}_{}".format(block_type, block_idx)
        mixed_name = "{}_mixed".format(block_name)
        mixed = Concatenate(axis=channel_axis, name=mixed_name)(branches)
        up = conv2d_bn(mixed, K.int_shape(x)[channel_axis], 1, activation=None, use_bias=True, w_reg=w_reg, name=mixed_name)

        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=K.int_shape(x)[1:], arguments={"scale": scale}, name=block_name)([x, up])
        if activation is not None:
            relu_name = block_name + "_relu"
            x = Activation(activation, name=relu_name)(x)
        return x

    channel_axis = 3

    #-------------------------------------
    # Stem block
    #-------------------------------------
    x = conv2d_bn(input_tensor, 32, 3, strides=2, w_reg=w_reg, name="stg0_blk0") # 512x512xC -> 256x256x32
    x = conv2d_bn(x, 32, 3, w_reg=w_reg, name="stg0_blk1") # 256x256x32 -> 256x256x32
    x = conv2d_bn(x, 64, 3, w_reg=w_reg, name="stg0_blk2") # 256x256x32 -> 256x256x64
    x = MaxPooling2D(3, strides=2, padding="same", name="stg0_blk3_pool")(x) # 256x256x64 -> 128x128x64

    x = conv2d_bn(x, 80, 1, w_reg=w_reg, name="stg1_blk0") # 256x256x64 -> 128x128x80
    x = conv2d_bn(x, 192, 3, w_reg=w_reg, name="stg1_blk1") # 128x128x80 -> 128x128x192
    x = MaxPooling2D(3, strides=2, padding="same", name="stg1_blk2_pool")(x) # 128x128x192 -> 64x64x192
    #------------ 64x64x192 ------------

    #-------------------------------------
    # Mixed 5b (Inception-A block): 1x1 + 1x1_5x5 + 1x1_3x3_3x3 + pool
    #-------------------------------------
    branch_0 = conv2d_bn(x, 96, 1, w_reg=w_reg, name="mixed5b_brnch0_blk0") # 64x64x192 -> 64x64x96

    branch_1 = conv2d_bn(x, 48, 1, w_reg=w_reg, name="mixed5b_brnch1_blk0") # 64x64x192 -> 64x64x48
    branch_1 = conv2d_bn(branch_1, 64, 5, w_reg=w_reg, name="mixed5b_brnch1_blk1") # 64x64x48 -> 64x64x64

    branch_2 = conv2d_bn(x, 64, 1, w_reg=w_reg, name="mixed5b_brnch2_blk0") # 64x64x192 -> 64x64x64
    branch_2 = conv2d_bn(branch_2, 96, 3, w_reg=w_reg, name="mixed5b_brnch2_blk1") # 64x64x64 -> 64x64x96
    branch_2 = conv2d_bn(branch_2, 96, 3, w_reg=w_reg, name="mixed5b_brnch2_blk2") # 64x64x96 -> 64x64x96

    branch_pool = AveragePooling2D(3, strides=1, padding="same", name="mixed5b_brnchpool_blk0_pool")(x) # 64x64x192 -> 64x64x192
    branch_pool = conv2d_bn(branch_pool, 64, 1, w_reg=w_reg, name="mixed5b_brnchpool_blk1") # 64x64x192 -> 64x64x64

    x = Concatenate(axis=channel_axis, name="mixed5b")([branch_0, branch_1, branch_2, branch_pool])
    #------------ 64x64x320 ------------

    #-------------------------------------
    # 10x block35 (Inception-ResNet-A block): x + scale*(1x1 + 1x1_3x3 + 1x1_3x3_3x3)
    #-------------------------------------
    for block_idx in range(1, 11):
        x = inception_resnet_block(x, scale=0.17, block_type="block35", block_idx=block_idx, w_reg=w_reg)

    # last layer name = block35_10_relu
    #------------ 64x64x320 ------------

    #-------------------------------------
    # Mixed 6a (Reduction-A block): 1x1 + 1x1_3x3 + 1x1_3x3_3x3)
    #-------------------------------------
    branch_0 = conv2d_bn(x, 384, 3, strides=2, w_reg=w_reg, name="mixed6a_brnch0_blk0") # 64x64x320 -> 32x32x384

    branch_1 = conv2d_bn(x, 256, 1, w_reg=w_reg, name="mixed6a_brnch1_blk0") # 64x64x320 -> 64x64x256
    branch_1 = conv2d_bn(branch_1, 256, 3, w_reg=w_reg, name="mixed6a_brnch1_blk1") # 64x64x256 -> 64x64x256
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, w_reg=w_reg, name="mixed6a_brnch1_blk2") # 64x64x256 -> 32x32x384

    branch_pool = MaxPooling2D(3, strides=2, padding="same", name="mixed6a_brnchpool_blk0_pool")(x) # 64x64x320 -> 32x32x320

    x = Concatenate(axis=channel_axis, name="mixed6a")([branch_0, branch_1, branch_pool])
    #------------ 32x32x1088 ------------

    #-------------------------------------
    # 20x block17 (Inception-ResNet-B block): x + scale*(1x1 + 1x1_1x7_7x1)
    #-------------------------------------
    for block_idx in range(1, 21):
        x = inception_resnet_block(x, scale=0.1, block_type="block17", block_idx=block_idx, w_reg=w_reg)

    # last layer name = block17_20_relu
    #------------ 32x32x1088 ------------

    #-------------------------------------
    # Mixed 7a (Reduction-B block): 1x1 + 1x1_3x3 + 1x1_3x3_3x3)
    #-------------------------------------
    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1, w_reg=w_reg, name="mixed7a_brnch0_blk0") # 32x32x1088 -> 32x32x256
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, w_reg=w_reg, name="mixed7a_brnch0_blk1") # 32x32x256 -> 16x16x384

    branch_1 = conv2d_bn(x, 256, 1, w_reg=w_reg, name="mixed7a_brnch1_blk0") # 32x32x1088 -> 32x32x256
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, w_reg=w_reg, name="mixed7a_brnch1_blk1") # 32x32x256 -> 16x16x288

    branch_2 = conv2d_bn(x, 256, 1, w_reg=w_reg, name="mixed7a_brnch2_blk0") # 32x32x1088 -> 32x32x256
    branch_2 = conv2d_bn(branch_2, 288, 3, w_reg=w_reg, name="mixed7a_brnch2_blk1") # 32x32x256 -> 32x32x288
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, w_reg=w_reg, name="mixed7a_brnch2_blk2") # 32x32x288 -> 16x16x320

    branch_pool = MaxPooling2D(3, strides=2, padding="same", name="mixed7a_brnchpool_blk0_pool")(x) # 32x32x1088 -> 16x16x1088

    x = Concatenate(axis=channel_axis, name="mixed7a")([branch_0, branch_1, branch_2, branch_pool])
    #------------ 16x16x2080 ------------

    #-------------------------------------
    # 10x block8 (Inception-ResNet-C block): x + scale*(1x1 + 1x1_1x3_3x1)
    #-------------------------------------
    for block_idx in range(1, 10):
        x = inception_resnet_block(x, scale=0.2, block_type="block8", block_idx=block_idx, w_reg=w_reg)
    x = inception_resnet_block(x, scale=1., activation=None, block_type="block8", block_idx=10, w_reg=w_reg)

    # last layer name = block8_10_relu
    #------------ 16x16x2080 ------------

    #-------------------------------------
    # Final convolution block: 8 x 8 x 1536
    #-------------------------------------
    x = conv2d_bn(x, 1536, 1, w_reg=w_reg, name="conv7b") # 16x16x2080 -> 16x16x1536

    gap = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    if return_tensor:
        return gap
    else:
        # Create model
        model = Model(input_tensor, x, name="inception_resnet_v2")
        #print model.summary()

        return model



#-------------------------------------
# ImageNet models
#-------------------------------------
def VGG16_LMIC(input_tensor, w_reg, return_tensor=False):

    # Block 1
    x = Conv2D(64, (3, 3), activation=None, padding="same", name="block1_conv1", kernel_regularizer=w_reg)(input_tensor)
    x = Activation("relu", name="block1_conv1_relu")(x)
    x = Conv2D(64, (3, 3), activation=None, padding="same", name="block1_conv2", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block1_conv2_relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation=None, padding="same", name="block2_conv1", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block2_conv1_relu")(x)
    x = Conv2D(128, (3, 3), activation=None, padding="same", name="block2_conv2", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block2_conv2_relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation=None, padding="same", name="block3_conv1", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block3_conv1_relu")(x)
    x = Conv2D(256, (3, 3), activation=None, padding="same", name="block3_conv2", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block3_conv2_relu")(x)
    x = Conv2D(256, (3, 3), activation=None, padding="same", name="block3_conv3", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block3_conv3_relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation=None, padding="same", name="block4_conv1", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block4_conv1_relu")(x)
    x = Conv2D(512, (3, 3), activation=None, padding="same", name="block4_conv2", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block4_conv2_relu")(x)
    x = Conv2D(512, (3, 3), activation=None, padding="same", name="block4_conv3", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block4_conv3_relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation=None, padding="same", name="block5_conv1", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block5_conv1_relu")(x)
    x = Conv2D(512, (3, 3), activation=None, padding="same", name="block5_conv2", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block5_conv2_relu")(x)
    x = Conv2D(512, (3, 3), activation=None, padding="same", name="block5_conv3", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block5_conv3_relu")(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    gap = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    if return_tensor:
        return gap
    else:
        # Create model
        model = Model(input_tensor, x, name="vgg16")
        #print model.summary()

        return model


def VGG19_LMIC(input_tensor, w_reg, return_tensor=False):

    # Block 1
    x = Conv2D(64, (3, 3), activation=None, padding="same", name="block1_conv1", kernel_regularizer=w_reg)(input_tensor)
    x = Activation("relu", name="block1_conv1_relu")(x)
    x = Conv2D(64, (3, 3), activation=None, padding="same", name="block1_conv2", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block1_conv2_relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation=None, padding="same", name="block2_conv1", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block2_conv1_relu")(x)
    x = Conv2D(128, (3, 3), activation=None, padding="same", name="block2_conv2", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block2_conv2_relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation=None, padding="same", name="block3_conv1", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block3_conv1_relu")(x)
    x = Conv2D(256, (3, 3), activation=None, padding="same", name="block3_conv2", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block3_conv2_relu")(x)
    x = Conv2D(256, (3, 3), activation=None, padding="same", name="block3_conv3", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block3_conv3_relu")(x)
    x = Conv2D(256, (3, 3), activation=None, padding="same", name="block3_conv4", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block3_conv4_relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation=None, padding="same", name="block4_conv1", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block4_conv1_relu")(x)
    x = Conv2D(512, (3, 3), activation=None, padding="same", name="block4_conv2", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block4_conv2_relu")(x)
    x = Conv2D(512, (3, 3), activation=None, padding="same", name="block4_conv3", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block4_conv3_relu")(x)
    x = Conv2D(512, (3, 3), activation=None, padding="same", name="block4_conv4", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block4_conv4_relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation=None, padding="same", name="block5_conv1", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block5_conv1_relu")(x)
    x = Conv2D(512, (3, 3), activation=None, padding="same", name="block5_conv2", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block5_conv2_relu")(x)
    x = Conv2D(512, (3, 3), activation=None, padding="same", name="block5_conv3", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block5_conv3_relu")(x)
    x = Conv2D(512, (3, 3), activation=None, padding="same", name="block5_conv4", kernel_regularizer=w_reg)(x)
    x = Activation("relu", name="block5_conv4_relu")(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    gap = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    if return_tensor:
        return gap
    else:
        # Create model
        model = Model(input_tensor, x, name="vgg19")
        #print model.summary()

        return model