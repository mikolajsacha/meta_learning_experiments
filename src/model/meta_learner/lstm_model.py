import math
from typing import List

from keras import Model, Input
import keras.backend as K
import tensorflow as tf
from keras.engine import Layer
from keras.initializers import RandomNormal, Constant
from keras.layers import Concatenate, LSTM, Dense, Multiply, Flatten, Subtract, Lambda

from src.model.meta_learner.meta_model import MetaLearnerModel
from src.model.meta_learner.meta_predict_model import MetaPredictLearnerModel
from src.model.util import get_trainable_params_count, gradient_preprocessing_left, gradient_preprocessing_right


# We use two meta-models, separate for prediction (training learner) and for meta-training (training meta-learner)
# This makes implementation simpler
# Models share weights. Model for prediction is simple one-to-one RNN with its inputs attached to outputs of learner
# Model for training is a multi-timestep RNN (many-to-one), where number of timesteps is the depth of BPTT


def get_common_lstm_model_layers(hidden_state_size: int, lr_bias: float, f_bias: float, initializer_std: float) \
        -> List[Layer]:
    return [
        Lambda(lambda x: gradient_preprocessing_left(x), name='preprocess_l'),
        Lambda(lambda x: gradient_preprocessing_right(x), name='preprocess_r'),
        Concatenate(axis=-1, name='concat_lstm_inputs'),
        LSTM(hidden_state_size, stateful=True, return_state=True, name='lstm'),
        Dense(hidden_state_size, name='learning_rate_dense_1', activation='relu'),
        Dense(hidden_state_size, name='forget_rate_dense_1', activation='relu'),
        Dense(1, name='learning_rate_dense_2', activation='sigmoid',
              kernel_initializer=RandomNormal(mean=0.0, stddev=initializer_std),
              bias_initializer=Constant(value=lr_bias)),
        Dense(1, name='forget_rate_dense_2', activation='sigmoid',
              kernel_initializer=RandomNormal(mean=0.0, stddev=initializer_std),
              bias_initializer=Constant(value=f_bias))
    ]


def lstm_train_meta_learner(backprop_depth: int, batch_size: int, common_layers: List[Layer]) -> Model:
    """
    :return: Keras Model for Meta-Learner used during training of Meta-Learner using BPTT
    """

    grads_input = Input(batch_shape=(batch_size, backprop_depth, 1), name='grads_input_train')
    loss_input = Input(batch_shape=(batch_size, backprop_depth, 1), name='loss_input_train')
    params_input = Input(batch_shape=(batch_size, 1), name='params_input_train')

    preprocessed_grads_l = common_layers[0](grads_input)
    preprocessed_grads_r = common_layers[1](grads_input)
    preprocessed_loss_l = common_layers[0](loss_input)
    preprocessed_loss_r = common_layers[1](loss_input)

    full_lstm_input = common_layers[2]([preprocessed_grads_l, preprocessed_grads_r,
                                        preprocessed_loss_l, preprocessed_loss_r])

    lstm_full_output = common_layers[3](full_lstm_input)
    lstm_output = lstm_full_output[0]

    lr_dense = common_layers[4](lstm_output)
    forget_dense = common_layers[5](lstm_output)

    lr_factor = common_layers[6](lr_dense)
    forget_factor = common_layers[7](forget_dense)

    final_grad = Lambda(lambda x: x[:, -1, :], name='final_grad_input')(grads_input)

    left = Multiply(name='output_left')([forget_factor, params_input])
    right = Multiply(name='output_right')([lr_factor, final_grad])

    output = Subtract(name='output')([left, right])

    return Model(inputs=[grads_input, loss_input, params_input], outputs=output)


# noinspection PyProtectedMember
def lstm_predict_meta_learner(learner: Model, backprop_depth: int, batch_size: int,
                              common_layers: List[Layer], debug_mode: bool) -> MetaPredictLearnerModel:
    """
    :return: MetaPredictLearnerModel for Meta-Learner used during training of Meta-Learner using BPTT
    """
    grads_tensor = K.concatenate([K.flatten(g) for g in K.gradients(learner.total_loss,
                                                                    learner._collected_trainable_weights)], axis=0)
    # reshape loss/grads so they have shape required for LSTM: (batch_size, 1, 1)
    grads_tensor = K.reshape(grads_tensor, shape=(batch_size, 1, 1))
    loss_tensor = tf.fill(value=learner.total_loss, dims=[batch_size, 1, 1])

    grads_input = Input(tensor=grads_tensor, batch_shape=(batch_size, 1, 1), name='grads_input_predict')
    loss_input = Input(tensor=loss_tensor, batch_shape=(batch_size, 1, 1), name='loss_input_predict')

    params_tensor = K.concatenate([K.flatten(p) for p in learner._collected_trainable_weights])
    params_input = Input(tensor=params_tensor, batch_shape=(batch_size,), name='params_input_predict')

    preprocessed_grads_l = common_layers[0](grads_input)
    preprocessed_grads_r = common_layers[1](grads_input)
    preprocessed_loss_l = common_layers[0](loss_input)
    preprocessed_loss_r = common_layers[1](loss_input)

    full_lstm_input = common_layers[2]([preprocessed_grads_l, preprocessed_grads_r,
                                        preprocessed_loss_l, preprocessed_loss_r])

    lstm_full_output = common_layers[3](full_lstm_input)
    lstm_output = lstm_full_output[0]
    states_outputs = lstm_full_output[1:]

    lr_dense = common_layers[4](lstm_output)
    forget_dense = common_layers[5](lstm_output)

    lr_factor = common_layers[6](lr_dense)
    forget_factor = common_layers[7](forget_dense)

    flat_grads = Flatten(name='flatten_grads_input')(grads_input)

    left = Multiply(name='output_left')([forget_factor, params_input])
    right = Multiply(name='output_right')([lr_factor, flat_grads])

    output = Subtract(name='output')([left, right])

    return MetaPredictLearnerModel(learner=learner, train_mode=True, backpropagation_depth=backprop_depth,
                                   inputs=[grads_input, loss_input, params_input],
                                   input_tensors=[grads_tensor, loss_tensor, params_tensor],
                                   states_outputs=states_outputs, outputs=output,
                                   debug_mode=debug_mode)


def inverse_sigmoid(x: float):
    return -math.log(1.0 / x - 1.0)


# noinspection PyProtectedMember
def lstm_meta_learner(learner: Model,
                      debug_mode: bool = False,
                      hidden_state_size: int = 20,
                      initial_learning_rate: float = 0.05,
                      initial_forget_rate: float = 0.9999,
                      initializer_std: float = 0.001,
                      backpropagation_depth: int = 20) -> MetaLearnerModel:
    # initialize weights, so in the beginning model resembles SGD
    # forget rate is close to 1 and lr is set to some constant value

    lr_bias = inverse_sigmoid(initial_learning_rate)
    f_bias = inverse_sigmoid(initial_forget_rate)

    common_layers = get_common_lstm_model_layers(hidden_state_size, lr_bias, f_bias, initializer_std)
    meta_batch_size = get_trainable_params_count(learner)

    train_meta_learner = lstm_train_meta_learner(backpropagation_depth, meta_batch_size, common_layers)

    predict_meta_learner = lstm_predict_meta_learner(learner, backpropagation_depth, meta_batch_size,
                                                     common_layers, debug_mode)

    return MetaLearnerModel(predict_meta_learner, train_meta_learner, debug_mode)
