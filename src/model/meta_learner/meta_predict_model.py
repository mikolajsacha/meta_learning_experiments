from typing import List, Optional, Tuple

import tensorflow as tf
from keras import Model
from keras.optimizers import Optimizer
import keras.backend as K
import numpy as np

from src.model.meta_learner.meta_training_sample import MetaTrainingSample
from src.model.util import get_trainable_params_count, get_input_tensors, standardize_train_inputs
from src.training.training_configuration import TrainingConfiguration


class MetaPredictLearnerModel(Model, Optimizer):
    """
     Model for Meta Learning used during prediction of Learner parameters.
     The model takes previous parameter value and gradient/loss as inputs and outputs the new parameter value
    """

    def __init__(self, learner: Model, configuration: TrainingConfiguration, train_mode: bool,
                 states_outputs: List[tf.Tensor], input_tensors: List[tf.Tensor], inputs, outputs, name=None):
        Model.__init__(self, inputs, outputs, name)
        self.debug_mode = configuration.debug_mode
        self.learner = learner
        self.learner_grads = K.concatenate([K.flatten(g) for g in
                                            K.gradients(self.learner.total_loss,
                                                        self.learner._collected_trainable_weights)])
        self.learner_inputs = get_input_tensors(self.learner)

        self.output_size = get_trainable_params_count(self.learner)
        self.input_tensors = input_tensors
        self.states_outputs = states_outputs

        self.backprop_depth = configuration.backpropagation_depth
        self.train_mode = train_mode

        self.state_tensors = []

        # for BPTT (many-to-one) we need to store values of last inputs together with value of last output
        # to save memory, I use tensors with constant shape in circular way, marking current index
        self.current_backprop_index = tf.Variable(0, dtype=tf.int32)

        self.initial_states = []
        self.states_history = []
        self.inputs_history = []
        self.learner_weights_history = []
        self.current_output = tf.Variable(tf.zeros(shape=(1, self.output_size)), name='current_meta_output')

    def get_updates(self, loss: tf.Tensor, params: List[tf.Tensor]) -> List[tf.Tensor]:
        """
        Returns list of tensors of updates for learner parameters based on loss and previous values
        :param loss: tensor with loss
        :param params: list of tensors of parameters (each tensor = parameters for a single layer)
        :return: list of tensors with updates for each layer
        """
        if len(self.state_tensors) == 0:
            raise ValueError("Model is probably not compiled (len(self.state_tensors) == 0)")

        # here we assume that our meta-learner inputs are Learner tensors
        updates = []

        # save history of prediction for training meta-learner later
        if self.train_mode:
            def update_hist(hist: tf.Variable, new_val: tf.Tensor) -> tf.Tensor:
                return tf.scatter_update(hist, indices=self.current_backprop_index, updates=new_val)

            for input_hist, current_input in zip(self.inputs_history, self.input_tensors):
                updates.append(update_hist(input_hist, current_input))

            for state_hist, current_state in zip(self.states_history, self.state_tensors):
                updates.append(update_hist(state_hist, current_state))

            updates.append(tf.assign(self.current_output, self.outputs, validate_shape=False))

            if self.debug_mode:
                for learner_params_hist, current_learner_params in \
                        zip(self.learner_weights_history, params):
                    updates.append(update_hist(learner_params_hist, current_learner_params))

        with tf.control_dependencies(updates):
            # update learner parameters
            # new_params is a 1D tensor, a batch. We need to split it into tensors for each layer
            new_params = K.flatten(self.outputs)
            i = 0
            for p in params:
                p_size = np.product(p.get_shape().as_list())
                new_p = tf.slice(new_params, begin=[i], size=[p_size])
                # now we need to reshape new_p so it feed shape of the layer
                new_p = tf.reshape(new_p, shape=p.get_shape())

                updates.append(tf.assign(p, new_p))

                i += p_size

            # update meta-learner states
            for old_state, new_state in zip(self.state_tensors, self.states_outputs):
                updates.append(tf.assign(old_state, new_state))

        if self.train_mode:
            with tf.control_dependencies(updates):
                new_backprop_index = tf.mod(tf.add(self.current_backprop_index, 1), self.backprop_depth)
                updates += [
                    tf.assign(self.current_backprop_index, new_backprop_index)
                ]

        return updates

    def _init_history_tensors(self, tensors_to_save: List[tf.Tensor]) -> List[tf.Tensor]:
        return [tf.zeros(shape=(self.backprop_depth,) + tuple(t.get_shape())) for t in tensors_to_save]

    def _initialize_tensors(self):
        self.state_tensors = []
        for layer in self.layers:
            if layer.stateful:
                self.state_tensors += layer.states

        self.states_history = [tf.Variable(t, name='states_history_{}'.format(i))
                               for i, t in enumerate(self._init_history_tensors(self.state_tensors))]

        self.inputs_history = [tf.Variable(t, name='inputs_history_{}'.format(i))
                               for i, t in enumerate(self._init_history_tensors(self.input_tensors))]

        if self.debug_mode:
            self.learner_weights_history = [tf.Variable(t, name='learner_weights_history_'.format(i)) for i, t in
                                            enumerate(self._init_history_tensors(
                                                self.learner._collected_trainable_weights))]

    def compile(self, optimizer, loss=None, metrics=None, loss_weights=None,
                sample_weight_mode=None, weighted_metrics=None,
                target_tensors=None, **kwargs):
        super(MetaPredictLearnerModel, self).compile(optimizer, loss, metrics, loss_weights, sample_weight_mode,
                                                     weighted_metrics, target_tensors, **kwargs)
        self._initialize_tensors()

    def reset_states(self):
        super(MetaPredictLearnerModel, self).reset_states()

        updates = [
            K.update(self.current_backprop_index, 0)
        ]

        K.get_session().run(updates)

    def roll_and_squeeze(self, arr: np.ndarray, backprop_ind: int):
        rolled = np.roll(arr, shift=self.backprop_depth - backprop_ind, axis=0)
        squeezed = np.squeeze(rolled)
        return squeezed

    def _retrieve_debug_training_sample(
            self,
            valid_x: np.ndarray,
            valid_y: np.ndarray,
            learner_training_batches: Optional[List[Tuple[np.ndarray, np.ndarray]]]) -> MetaTrainingSample:

        ins = standardize_train_inputs(self.learner, valid_x, valid_y)
        feed_dict = dict(zip(self.learner_inputs, ins))

        fetches = [self.current_backprop_index, self.learner_grads, self.current_output] \
                  + self.inputs_history + self.states_history + self.learner_weights_history

        evaluated = K.get_session().run(fetches, feed_dict=feed_dict)

        backprop_ind = evaluated[0]
        learner_grads = evaluated[1]
        final_output = evaluated[2].flatten()

        i = 3
        j = i + len(self.inputs_history)
        inputs_history = [self.roll_and_squeeze(e, backprop_ind) for e in evaluated[i: j]]

        i = j
        j = i + len(self.states_history)
        states_history = [self.roll_and_squeeze(e, backprop_ind) for e in evaluated[i: j]]

        learner_params_history = [self.roll_and_squeeze(e, backprop_ind) for e in evaluated[j:]]

        if self.backprop_depth == 1:
            initial_learner_weights = learner_params_history
            initial_states = states_history
        else:
            initial_learner_weights = [hist[0] for hist in learner_params_history]
            initial_states = [hist[0] for hist in states_history]

        # reshape inputs so they fit our training model
        inputs_history[0] = np.expand_dims(np.transpose(inputs_history[0]), axis=-1)
        inputs_history[1] = np.expand_dims(np.transpose(inputs_history[1]), axis=-1)
        if len(inputs_history) > 3:
            inputs_history[3] = np.expand_dims(np.transpose(inputs_history[3]), axis=-1)

        # we need only final value of params_input for BPTT
        if self.backprop_depth != 1:
            inputs_history[2] = np.expand_dims(inputs_history[2][-1], axis=-1)

        if self.backprop_depth == 1:
            for i in range(len(inputs_history)):
                inputs_history[i] = np.expand_dims(inputs_history[i], axis=-1)

        return MetaTrainingSample(
            inputs=inputs_history,
            initial_states=initial_states,
            learner_grads=learner_grads,
            final_output=final_output,
            learner_training_batches=learner_training_batches,
            learner_validation_batch=(valid_x, valid_y),
            initial_learner_weights=initial_learner_weights
        )

    def _retrieve_no_debug_training_sample(
            self,
            valid_x: np.ndarray,
            valid_y: np.ndarray) -> MetaTrainingSample:

        ins = standardize_train_inputs(self.learner, valid_x, valid_y)
        feed_dict = dict(zip(self.learner_inputs, ins))

        fetches = [self.current_backprop_index, self.learner_grads] + self.inputs_history \
                  + self.states_history + self.learner_weights_history

        evaluated = K.get_session().run(fetches, feed_dict=feed_dict)

        backprop_ind = evaluated[0]
        learner_grads = evaluated[1]

        i = 2
        j = i + len(self.inputs_history)
        inputs_history = [self.roll_and_squeeze(e, backprop_ind) for e in evaluated[i: j]]

        states_history = [self.roll_and_squeeze(e, backprop_ind) for e in evaluated[j:]]

        if self.backprop_depth == 1:
            initial_states = states_history
        else:
            initial_states = [hist[0] for hist in states_history]

        # reshape inputs so they fit our training model
        inputs_history[0] = np.expand_dims(np.transpose(inputs_history[0]), axis=-1)
        inputs_history[1] = np.expand_dims(np.transpose(inputs_history[1]), axis=-1)
        if len(inputs_history) > 3:
            inputs_history[3] = np.expand_dims(np.transpose(inputs_history[3]), axis=-1)

        # we need only final value of params_input for BPTT
        if self.backprop_depth == 1:
            for i in range(len(inputs_history)):
                inputs_history[i] = np.expand_dims(inputs_history[i], axis=-1)
        else:
            inputs_history[2] = np.expand_dims(inputs_history[2][-1], axis=-1)

        return MetaTrainingSample(
            inputs=inputs_history,
            initial_states=initial_states,
            learner_grads=learner_grads,
        )

    def retrieve_training_sample(
            self,
            valid_x: np.ndarray,
            valid_y: np.ndarray,
            learner_training_batches: Optional[List[Tuple[np.ndarray, np.ndarray]]]) -> MetaTrainingSample:
        if self.debug_mode:
            return self._retrieve_debug_training_sample(valid_x, valid_y, learner_training_batches)
        else:
            return self._retrieve_no_debug_training_sample(valid_x, valid_y)
