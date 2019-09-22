from .models import *
from .model_config import *
name = "GrayNetKeras"


def init_input_tensor(input_shape, input_tensor, weights='graynet', **kwargs):
    if input_tensor is None:
        input_tensor = Input(shape=input_shape)

    if input_tensor.shape[-1] == 1 and weights=='graynet':
        x = input_tensor
    elif input_tensor.shape[-1] == 1:
        x = Concatenate(axis=-1)([input_tensor, input_tensor, input_tensor])
    elif input_tensor.shape[-1] == 3:
        x = input_tensor
    else:
        print("[ERROR] Wrong input tensor or shape")
        raise NotImplementedError
    return x


def Densenet121_GrayNet(input_tensor=None, input_shape=None, weights='graynet', **kwargs):
    # Make input tensor
    input_tensor = init_input_tensor(input_shape, input_tensor, weights=weights)
    model_name = 'densenet121'

    # Build body
    if weights == 'graynet':
        print("Input tensor shape:", input_tensor.shape)
        weights = get_weight_path('densnet121', 'graynet', input_tensor.shape[-1])

    print("weight:", weights)
    densnet_model = DenseNet121(include_top=False, weights=None, input_tensor=input_tensor, pooling='avg')
    gap = densnet_model(input_tensor)

    model = Model(inputs=input_tensor, outputs=gap, name=model_name)
    model.load_weights(weights, by_name=True, skip_mismatch=True)

    return model

def UnetEncoder_GrayNet(input_tensor=None, input_shape=None, weights='graynet', **kwargs):
    # input_tensor = init_input_tensor(input_shape, input_tensor)
    model_name = 'unetenc'

    last_conv, bypass1, bypass2, bypass3, bypass4 = UNET.encoder(input_tensor, num_chs=64, **kwargs)
    gap = GlobalAveragePooling2D()(last_conv)
    model = Model(inputs=input_tensor, outputs=gap, name=model_name)

    # Load GrayNet weight
    if weights is not None:
        weights_path = get_weight_path(model_name, weights, input_tensor.shape[-1])
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def ResNet50_GrayNet(input_tensor=None, input_shape=None, weights='graynet', **kwargs):
    # input_tensor = init_input_tensor(input_shape, input_tensor)
    model_name = 'resnet50'

    gap = RESNET50_LMIC(input_tensor, return_tensor=True, **kwargs)
    model = Model(inputs=input_tensor, outputs=gap, name=model_name)

    # Load GrayNet weight
    if weights is not None:
        weights_path = get_weight_path(model_name, weights, input_tensor.shape[-1])
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def InceptionV3_GrayNet(input_tensor=None, input_shape=None, weights='graynet', **kwargs):
    # input_tensor = init_input_tensor(input_shape, input_tensor)
    model_name = 'inceptionv3'

    gap = INCEPTIONV3_LMIC(input_tensor, return_tensor=True, **kwargs)
    model = Model(inputs=input_tensor, outputs=gap, name=model_name)

    # Load GrayNet weight
    if weights is not None:
        weights_path = get_weight_path(model_name, weights, input_tensor.shape[-1])
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model

