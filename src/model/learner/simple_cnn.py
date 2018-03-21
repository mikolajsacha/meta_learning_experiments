from keras import Model, Input
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
import keras.backend as K


def conv_block(n_filters: int, k: int, pool_size):
    def conv_fun(inputs):
        y = Conv2D(n_filters, (k, k), padding='same')(inputs)
        #  y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = MaxPooling2D(pool_size=(pool_size, pool_size))(y)
        return y
    return conv_fun


def build_simple_cnn(input_shape, num_outputs, n_filters=32) -> Model:
    if len(input_shape) != 3:
        raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

    # Permute dimension order if necessary
    if K.image_dim_ordering() == 'tf':
        input_shape = (input_shape[1], input_shape[2], input_shape[0])

    inputs = Input(shape=input_shape)

    n_conv_blocks = 4
    y = inputs

    for _ in range(n_conv_blocks):
        y = conv_block(n_filters=n_filters, k=3, pool_size=2)(y)

    y = Flatten()(y)
    output = Dense(num_outputs, activation='softmax')(y)

    return Model(inputs=inputs, outputs=output)
