from typing import List

from keras import Model
import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.engine.training import _standardize_input_data


def get_trainable_params_count(model: Model) -> int:
    """
    :param model: Keras model
    :return: total number of trainable weights of the model
    """
    return int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))


def gradient_preprocessing_left(value: tf.Tensor, p=10.0) -> tf.Tensor:
    return tf.where(condition=tf.abs(value) >= tf.exp(-p),
                    x=tf.log(tf.abs(value)) / p,
                    y=-tf.ones_like(value))


def gradient_preprocessing_right(value: tf.Tensor, p=10.0) -> tf.Tensor:
    return tf.where(condition=tf.abs(value) >= tf.exp(-p),
                    x=tf.sign(value),
                    y=tf.exp(p) * value)


# noinspection PyProtectedMember
def get_input_tensors(model: Model) -> List[tf.Tensor]:
    inputs = model._feed_inputs + model._feed_targets + model._feed_sample_weights
    if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs += [K.learning_phase()]
    return inputs


# noinspection PyProtectedMember
def standardize_train_inputs(model: Model, x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
    x, y, sample_weights = model._standardize_user_data(x, y, sample_weight=None)
    if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        ins = x + y + sample_weights + [0.]
    else:
        ins = x + y + sample_weights
    return ins


# noinspection PyProtectedMember
def standardize_predict_inputs(model: Model, x: np.ndarray) -> List[np.ndarray]:
    x = _standardize_input_data(x, model._feed_input_names,
                                model._feed_input_shapes)
    if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        ins = x + [0.]
    else:
        ins = x
    return ins
