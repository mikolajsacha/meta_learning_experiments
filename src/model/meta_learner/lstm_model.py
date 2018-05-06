import math
from typing import List

from keras import Model, Input
import keras.backend as K
import tensorflow as tf
from keras.engine import Layer
from keras.initializers import RandomNormal, Constant
from keras.layers import Concatenate, LSTM, Dense, Multiply, Flatten, Subtract, Lambda

from src.isotropy.lanczos import TopKEigenvaluesBatched
from src.model.meta_learner.meta_model import MetaLearnerModel
from src.model.meta_learner.meta_predict_model import MetaPredictLearnerModel
from src.model.util import get_trainable_params_count, gradient_preprocessing_left, gradient_preprocessing_right


# We use two meta-models, separate for prediction (training learner) and for meta-training (training meta-learner)
# This makes implementation simpler
# Models share weights. Model for prediction is simple one-to-one RNN with its inputs attached to outputs of learner
# Model for training is a multi-timestep RNN (many-to-one), where number of timesteps is the depth of BPTT
from src.training.training_configuration import TrainingConfiguration


def get_common_lstm_model_layers(hidden_state_size: int,
                                 constant_lr_model: bool,
                                 lr_bias: float, f_bias: float, initializer_std: float) \
        -> List[Layer]:

    layers = [
        Lambda(lambda x: gradient_preprocessing_left(x), name='preprocess_l'),
        Lambda(lambda x: gradient_preprocessing_right(x), name='preprocess_r')
    ]

    if constant_lr_model:
        layers += [
            Dense(1, name='learning_rate_constant', activation='sigmoid',
                  kernel_initializer=Constant(value=0.0),
                  bias_initializer=Constant(value=lr_bias)),
            ]
    else:
        layers += [
            Concatenate(axis=-1, name='concat_lstm_inputs'),
            LSTM(hidden_state_size, stateful=True, return_state=True, return_sequences=True,
                 name='lstm_1', activation='relu'),
            LSTM(hidden_state_size, stateful=True, return_state=True,
                 name='lstm_2', activation='relu'),
            Dense(1, name='learning_rate_dense', activation='sigmoid',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=initializer_std),
                  bias_initializer=Constant(value=lr_bias)),
            Dense(1, name='forget_rate_dense', activation='sigmoid',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=initializer_std),
                  bias_initializer=Constant(value=f_bias))
        ]
    return layers


def preprocess_inputs(inps, common_layers):
    preprocessed = []
    for inp in inps:
        preprocessed_l = common_layers[0](inp)
        preprocessed_r = common_layers[1](inp)
        preprocessed.append(preprocessed_l)
        preprocessed.append(preprocessed_r)
    return preprocessed


def lstm_train_meta_learner(with_hessian_eigenvalue_features: bool, with_hessian_eigenvector_features: bool,
                            constant_lr_model: bool, backprop_depth: int, batch_size: int,
                            common_layers: List[Layer]) -> Model:
    """
    :return: Keras Model for Meta-Learner used during training of Meta-Learner using BPTT
    """

    grads_input = Input(batch_shape=(batch_size, backprop_depth, 1), name='grads_input_train')
    loss_input = Input(batch_shape=(batch_size, backprop_depth, 1), name='loss_input_train')
    params_input = Input(batch_shape=(batch_size, 1), name='params_input_train')

    inputs = [grads_input, loss_input, params_input]

    lstm_inputs = preprocess_inputs([grads_input, loss_input], common_layers)

    if with_hessian_eigenvector_features or with_hessian_eigenvalue_features:
        eigenval_input = Input(batch_shape=(batch_size, backprop_depth, 1), name='hessian_eigenval_input_train')
        lstm_inputs += preprocess_inputs([eigenval_input], common_layers)
        inputs.append(eigenval_input)

    if with_hessian_eigenvector_features:
        eigenvec_input = Input(batch_shape=(batch_size, backprop_depth, 1), name='hessian_eigenvec_input_train')
        inputs.append(eigenvec_input)
        lstm_inputs += preprocess_inputs([eigenvec_input], common_layers)

    if constant_lr_model:
        final_grad = Lambda(lambda x: x[:, -1, :], name='final_grad_input')(grads_input)
        lr_factor = common_layers[2](final_grad)
        right = Multiply(name='output_right')([lr_factor, final_grad])

        output = Subtract(name='output')([params_input, right])
        return Model(inputs=inputs, outputs=output)

    full_lstm_input = common_layers[2](lstm_inputs)
    lstm_full_output = common_layers[3](full_lstm_input)
    lstm_output = lstm_full_output[0]

    lstm_full_output = common_layers[4](lstm_output)
    lstm_output = lstm_full_output[0]

    lr_factor = common_layers[5](lstm_output)
    forget_factor = common_layers[6](lstm_output)

    final_grad = Lambda(lambda x: x[:, -1, :], name='final_grad_input')(grads_input)

    left = Multiply(name='output_left')([forget_factor, params_input])
    right = Multiply(name='output_right')([lr_factor, final_grad])

    output = Subtract(name='output')([left, right])

    return Model(inputs=inputs, outputs=output)


# noinspection PyProtectedMember
def lstm_predict_meta_learner(learner: Model, eigenvals_callback: TopKEigenvaluesBatched,
                              configuration: TrainingConfiguration, batch_size: int,
                              common_layers: List[Layer]) -> MetaPredictLearnerModel:
    """
    :return: MetaPredictLearnerModel for Meta-Learner used during training of Meta-Learner using BPTT
    """
    grads_tensor = K.concatenate([K.flatten(g) for g in K.gradients(learner.total_loss,
                                                                    learner._collected_trainable_weights)], axis=0)
    # reshape loss/grads so they have shape required for LSTM: (batch_size, 1, 1)
    grads_tensor = K.reshape(grads_tensor, shape=(batch_size, 1, 1))
    grads_input = Input(tensor=grads_tensor, batch_shape=(batch_size, 1, 1), name='grads_input_predict')

    loss_tensor = tf.reshape(learner.total_loss, shape=[])
    loss_input = Input(tensor=loss_tensor, batch_shape=(1, ), name='loss_input_predict')

    loss_repeated_input = Lambda(lambda x: tf.fill(value=tf.reshape(x, shape=[]), dims=[batch_size, 1, 1]))(loss_input)

    params_tensor = K.concatenate([K.flatten(p) for p in learner._collected_trainable_weights])
    params_input = Input(tensor=params_tensor, batch_shape=(batch_size,), name='params_input_predict')

    input_tensors = [grads_tensor, loss_tensor, params_tensor]
    inputs = [grads_input, loss_input, params_input]

    lstm_inputs = preprocess_inputs([grads_input, loss_repeated_input], common_layers)

    if configuration.with_hessian_eigenvector_features:
        eigenval_tensor = tf.reshape(eigenvals_callback.spectral_norm_tensor_2, shape=[])
        eigenval_input = Input(tensor=eigenval_tensor, batch_shape=(1, ),
                               name='hessian_eigenval_input_predict')

        eigenvec_tensor = tf.reshape(eigenvals_callback.eigenvector_tensor, shape=[batch_size, 1, 1])
        eigenvec_input = Input(tensor=eigenvec_tensor, batch_shape=(batch_size, 1, 1),
                               name='hessian_eigenvec_input_predict')

        eigenval_repeated_input = Lambda(lambda x: tf.fill(value=tf.reshape(x, shape=[]),
                                                           dims=[batch_size, 1, 1]))(eigenval_input)

        input_tensors.append(eigenval_tensor)
        input_tensors.append(eigenvec_tensor)

        inputs.append(eigenval_input)
        inputs.append(eigenvec_input)

        lstm_inputs += preprocess_inputs([eigenval_repeated_input, eigenvec_input], common_layers)

    elif configuration.with_hessian_eigenvalue_features:
        eigenval_tensor = tf.reshape(eigenvals_callback.spectral_norm_tensor, shape=[])
        eigenval_input = Input(tensor=eigenval_tensor, batch_shape=(1, ),
                               name='hessian_eigenval_input_predict')

        eigenval_repeated_input = Lambda(lambda x: tf.fill(value=tf.reshape(x, shape=[]),
                                                           dims=[batch_size, 1, 1]))(eigenval_input)

        input_tensors.append(eigenval_tensor)
        inputs.append(eigenval_input)
        lstm_inputs += preprocess_inputs([eigenval_repeated_input], common_layers)

    if configuration.constant_lr_model:
        final_grad = Lambda(lambda x: x[:, -1, :], name='final_grad_input')(grads_input)
        lr_factor = common_layers[2](final_grad)
        right = Multiply(name='output_right')([lr_factor, final_grad])

        output = Subtract(name='output')([params_input, right])
        intermediate_outputs = [lr_factor]
        states_outputs = []

    else:
        full_lstm_input = common_layers[2](lstm_inputs)
        lstm_full_output = common_layers[3](full_lstm_input)
        lstm_output = lstm_full_output[0]
        states_outputs = lstm_full_output[1:]

        lstm_full_output = common_layers[4](lstm_output)
        lstm_output = lstm_full_output[0]
        states_outputs += lstm_full_output[1:]

        lr_factor = common_layers[5](lstm_output)
        forget_factor =\
            common_layers[6](lstm_output)

        flat_grads = Flatten(name='flatten_grads_input')(grads_input)

        left = Multiply(name='output_left')([forget_factor, params_input])
        right = Multiply(name='output_right')([lr_factor, flat_grads])

        output = Subtract(name='output')([left, right])

        # can be measured for analysis
        intermediate_outputs = [forget_factor, lr_factor]

    return MetaPredictLearnerModel(learner=learner, configuration=configuration, train_mode=True, inputs=inputs,
                                   input_tensors=input_tensors, states_outputs=states_outputs, outputs=output,
                                   intermediate_outputs=intermediate_outputs)


def inverse_sigmoid(x: float):
    return -math.log(1.0 / x - 1.0)


# noinspection PyProtectedMember
def lstm_meta_learner(learner: Model,
                      eigenvals_callback: TopKEigenvaluesBatched,
                      configuration: TrainingConfiguration) -> MetaLearnerModel:
    # initialize weights, so in the beginning model resembles SGD
    # forget rate is close to 1 and lr is set to some constant value

    lr_bias = inverse_sigmoid(configuration.initial_lr)
    f_bias = inverse_sigmoid(0.9999)

    common_layers = get_common_lstm_model_layers(configuration.hidden_state_size, configuration.constant_lr_model,
                                                 lr_bias, f_bias, 0.001)
    meta_batch_size = get_trainable_params_count(learner)

    train_meta_learner = lstm_train_meta_learner(configuration.with_hessian_eigenvalue_features,
                                                 configuration.with_hessian_eigenvector_features,
                                                 configuration.constant_lr_model,
                                                 configuration.backpropagation_depth, meta_batch_size, common_layers)
    predict_meta_learner = lstm_predict_meta_learner(learner, eigenvals_callback, configuration,
                                                     meta_batch_size, common_layers)

    return MetaLearnerModel(predict_meta_learner, train_meta_learner, configuration.debug_mode)
