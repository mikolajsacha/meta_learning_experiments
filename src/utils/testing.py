"""
Methods for testing model correctness etc
"""
from logging import Logger

import keras.backend as K
import numpy as np
import tensorflow as tf

from src.model.meta_learner.meta_model import MetaLearnerModel
from src.model.meta_learner.meta_training_sample import MetaTrainingSample
from src.model.util import get_input_tensors, standardize_predict_inputs, get_trainable_params_count, \
    standardize_train_inputs


# noinspection PyProtectedMember
def gradient_check(meta_model: MetaLearnerModel,
                   training_sample: MetaTrainingSample,
                   logger: Logger,
                   epsilon: float = 10e-7) -> bool:
    """
    Performs gradient check on a single meta-training-sample.
    Warning: This method is very slow for big models!
    :param meta_model: MetaLearnerModel
    :param training_sample: training sample to gradient-check
    :param logger: Logger instance
    :param epsilon: epsilon factor used in gradient checking
    :return: True if gradient check passes, otherwise False
    """
    if training_sample.final_output is None:
        raise ValueError("For gradient check, 'final_output' must not be None")
    if training_sample.learner_training_batches is None:
        raise ValueError("For gradient check, 'learner_training_batches' must not be None")
    if training_sample.learner_validation_batch is None:
        raise ValueError("For gradient check, 'learner_validation_batch' must not be None")
    if training_sample.initial_learner_weights is None:
        raise ValueError("For gradient check, 'initial_learner_weights' must not be None")

    state_tensors = meta_model.predict_model.state_tensors
    input_tensors = get_input_tensors(meta_model.train_model)
    learner = meta_model.predict_model.learner

    sess = K.get_session()

    # first step is to evaluate gradients of meta-learner parameters using our method
    # to evaluate gradients, I use 'train_model' version of meta-learner

    # initialize meta-learner (train) states
    assert len(state_tensors) == len(training_sample.initial_states)
    feed_dict = dict(zip(meta_model.states_placeholder, training_sample.initial_states))
    sess.run(meta_model.init_train_states_updates, feed_dict=feed_dict)

    # standardize input for current meta-training sample
    inputs = standardize_predict_inputs(meta_model.train_model, training_sample.inputs)

    # compute gradients on current meta-learner parameters and training sample
    feed_dict = dict(zip(input_tensors, inputs))
    feed_dict[meta_model.learner_grad_placeholder] = training_sample.learner_grads

    # our method of computation of meta-learner gradients - this is what i want to check here for being correct
    evaluation = sess.run(fetches=meta_model.chained_grads, feed_dict=feed_dict)
    evaluated_meta_grads = np.concatenate([grad.flatten() for grad in evaluation])

    # gradient check for each meta-learner weight
    # for gradient checking i use 'predict_model' version of meta-learner (which is used for training Learner)
    n_meta_learner_params = get_trainable_params_count(meta_model.train_model)
    approximated_meta_grads = np.zeros(shape=n_meta_learner_params)

    valid_x, valid_y = training_sample.learner_validation_batch
    learner_valid_ins = standardize_train_inputs(learner, valid_x, valid_y)

    # tensors used for updating meta-learner weights
    trainable_meta_weights = sess.run(meta_model.predict_model.trainable_weights)
    meta_weights_placeholder = [tf.placeholder(shape=w.get_shape(), dtype=tf.float32)
                                for w in meta_model.predict_model.trainable_weights]
    meta_weights_updates = [tf.assign(w, new_w) for w, new_w in zip(meta_model.predict_model.trainable_weights,
                                                                    meta_weights_placeholder)]

    def calculate_loss(new_weights):
        # update weights of meta-learner ('predict_model')
        f_dict = dict(zip(meta_weights_placeholder, new_weights))
        sess.run(meta_weights_updates, feed_dict=f_dict)

        # initialize learner parameters
        learner.set_weights(training_sample.initial_learner_weights)

        # initialize meta-learner (predict) states
        f_dict = dict(zip(meta_model.states_placeholder, training_sample.initial_states))
        sess.run(meta_model.init_predict_states_updates, feed_dict=f_dict)

        # train learner using same batches as in the sample (meta 'predict_model' is used here)
        for x, y in training_sample.learner_training_batches:
            learner.train_on_batch(x, y)

        # calculate new learner loss on validation set after training
        f_dict = dict(zip(meta_model.predict_model.learner_inputs, learner_valid_ins))
        new_loss = sess.run(fetches=[learner.total_loss], feed_dict=f_dict)[0]

        return new_loss

    grad_ind = 0
    for i, w in enumerate(trainable_meta_weights):
        # set meta-learner ('predict_model') params to new, where only one weight is changed by some epsilon
        if w.ndim == 2:
            for j in range(w.shape[0]):
                for k in range(w.shape[1]):
                    changed_meta_learner_weights = [w.copy() for w in trainable_meta_weights]
                    changed_meta_learner_weights[i][j][k] += epsilon
                    loss1 = calculate_loss(changed_meta_learner_weights)
                    changed_meta_learner_weights[i][j][k] -= 2 * epsilon
                    loss2 = calculate_loss(changed_meta_learner_weights)
                    approximated_meta_grads[grad_ind] = (loss1 - loss2) / (2 * epsilon)
                    grad_ind += 1
        elif w.ndim == 1:
            for j in range(w.shape[0]):
                changed_meta_learner_weights = [w.copy() for w in trainable_meta_weights]
                changed_meta_learner_weights[i][j] += epsilon
                loss1 = calculate_loss(changed_meta_learner_weights)
                changed_meta_learner_weights[i][j] -= 2 * epsilon
                loss2 = calculate_loss(changed_meta_learner_weights)
                approximated_meta_grads[grad_ind] = (loss1 - loss2) / (2 * epsilon)
                grad_ind += 1
        else:
            raise ValueError("Only weights with ndim == 1 or ndim == 2 are supported in grad check")

    approximated_grad_diff = np.linalg.norm(approximated_meta_grads - evaluated_meta_grads) / \
                             (np.linalg.norm(approximated_meta_grads) + np.linalg.norm(evaluated_meta_grads))

    if approximated_grad_diff > epsilon:
        logger.error("GRAD-CHECK: (epsilon={}, dist={})!".format(epsilon, approximated_grad_diff))
        return False
    else:
        logger.debug("Grad-Check passed. (epsilon={}, dist={})".format(epsilon, approximated_grad_diff))

    return True
