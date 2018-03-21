from keras import Model, Input
import keras.backend as K
import tensorflow as tf
from keras.initializers import Constant
from keras.layers import Concatenate, LSTM, Dense, Multiply, Flatten, Subtract, Lambda

from src.model.meta_learner.lstm_model import inverse_sigmoid
from src.model.meta_learner.meta_model import MetaLearnerModel
from src.model.meta_learner.meta_predict_model import MetaPredictLearnerModel
from src.model.util import get_trainable_params_count


# this is a test model that has only one trainable weight (bias of Dense - kernel should be set to zeros)


def test_train_meta_learner(backprop_depth: int, batch_size: int, lr_bias: float) -> Model:
    """
    :return: Keras Model for Meta-Learner used during training of Meta-Learner using BPTT
    """

    # same inputs as in other models to use same API
    grads_input = Input(batch_shape=(batch_size, backprop_depth, 1), name='grads_input_train')
    loss_input = Input(batch_shape=(batch_size, backprop_depth, 1), name='loss_input_train')
    params_input = Input(batch_shape=(batch_size, 1), name='params_input_train')

    full_input = Concatenate(axis=-1, name='concat_grads_loss')([grads_input, loss_input])
    lstm = LSTM(2, name='lstm')
    lstm.trainable = False
    dummy_lstm = lstm(full_input)

    lr = Dense(1, name='dense', kernel_initializer=Constant(value=0.0),
               bias_initializer=Constant(value=lr_bias),
               activation='sigmoid')(dummy_lstm)

    final_grad = Lambda(lambda x: x[:, -1, :], name='final_grad_input')(grads_input)

    right = Multiply(name='output_right')([lr, final_grad])

    output = Subtract(name='output')([params_input, right])

    return Model(inputs=[grads_input, loss_input, params_input], outputs=output)


# noinspection PyProtectedMember
def test_predict_meta_learner(learner: Model, backprop_depth: int, batch_size: int) -> MetaPredictLearnerModel:
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

    full_lstm_input = Concatenate(axis=-1, name='concat_grads_loss')([grads_input, loss_input])
    lstm = LSTM(2, stateful=True, return_state=True, name='lstm')
    lstm.trainable = False
    lstm_full_output = lstm(full_lstm_input)
    lstm_output = lstm_full_output[0]
    states_outputs = lstm_full_output[1:]

    lr = Dense(1, name='learning_rate_dense', kernel_initializer=Constant(value=0.0),
               bias_initializer=Constant(value=0.0), activation='sigmoid')(lstm_output)

    flat_grads = Flatten(name='flatten_grads_input')(grads_input)

    right = Multiply(name='output_right')([lr, flat_grads])

    output = Subtract(name='output')([params_input, right])

    return MetaPredictLearnerModel(learner=learner, train_mode=True, backpropagation_depth=backprop_depth,
                                   inputs=[grads_input, loss_input, params_input],
                                   input_tensors=[grads_tensor, loss_tensor, params_tensor],
                                   states_outputs=states_outputs, outputs=output)


# noinspection PyProtectedMember
def test_meta_learner(learner: Model,
                      initial_learning_rate: float = 0.05,
                      backpropagation_depth: int = 20) -> MetaLearnerModel:
    # initialize weights, so in the beginning model resembles SGD
    # forget rate is close to 1 and lr is set to some constant value

    lr_bias = inverse_sigmoid(initial_learning_rate)

    meta_batch_size = get_trainable_params_count(learner)

    train_meta_learner = test_train_meta_learner(backpropagation_depth, meta_batch_size, lr_bias)
    predict_meta_learner = test_predict_meta_learner(learner, backpropagation_depth, meta_batch_size)

    return MetaLearnerModel(predict_meta_learner, train_meta_learner)
