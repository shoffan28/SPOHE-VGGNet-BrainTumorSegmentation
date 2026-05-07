import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    Conv2DTranspose, Concatenate, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19


def conv_block(input_tensor, num_filters, dropout_rate=0.1):
    x = Conv2D(num_filters, 3, padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    return x


def decoder_block(input_tensor, skip_features, num_filters, dropout_rate=0.1):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(input_tensor)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters, dropout_rate)
    return x


def build_spohe_vggnet(input_shape=(256, 256, 3), dropout_rate=0.1):
    inputs = Input(input_shape)

    vgg19 = VGG19(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )

    s1 = vgg19.get_layer("block1_conv2").output
    s2 = vgg19.get_layer("block2_conv2").output
    s3 = vgg19.get_layer("block3_conv4").output
    s4 = vgg19.get_layer("block4_conv4").output

    b1 = vgg19.get_layer("block5_conv4").output

    d1 = decoder_block(b1, s4, 512, dropout_rate)
    d2 = decoder_block(d1, s3, 256, dropout_rate)
    d3 = decoder_block(d2, s2, 128, dropout_rate)
    d4 = decoder_block(d3, s1, 64, dropout_rate)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="SPOHE_VGGNet")

    return model
