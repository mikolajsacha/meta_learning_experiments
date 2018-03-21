from typing import List

from keras import Model
import keras.backend as K
import tensorflow as tf

from src.model.custom_optimizers import CustomOptimizer
from src.model.meta_learner.meta_predict_model import MetaPredictLearnerModel
from src.model.meta_learner.meta_training_sample import MetaTrainingSample
from src.model.util import get_input_tensors, standardize_predict_inputs


# noinspection PyProtectedMember
class MetaLearnerModel(object):
    """
     Class for training and predicting Learner parameters using Meta Learning.
     Contains two Meta-Learner model: one used for prediction, and other used for training.
    """

    def __init__(self, predict_model: MetaPredictLearnerModel, train_model: Model, debug_mode: bool):
        self.predict_model = predict_model
        self.train_model = train_model
        self.debug_mode = debug_mode

        self.train_state_tensors = None
        self.predict_state_tensors = None
        self.states_placeholder = None
        self.init_train_states_updates = None
        self.init_predict_states_updates = None
        self.chained_grads = None
        self.train_updates = None
        self.copy_weights_updates = None
        self.learner_grad_placeholder = None
        self.batch_grads_placeholder = None

    def load_weights(self, weights_path: str):
        self.predict_model.load_weights(weights_path)
        self.train_model.load_weights(weights_path)

    def reset_states(self):
        self.predict_model.reset_states()
        self.train_model.reset_states()

    @property
    def train_mode(self):
        return self.predict_model.train_mode

    @train_mode.setter
    def train_mode(self, value):
        self.predict_model.train_mode = value
        # we need to reset learner train function so it uses new updates
        self.predict_model.learner.train_function = None

    def compile(self):
        if not isinstance(self.train_model.optimizer, CustomOptimizer):
            raise ValueError("CustomOptimizer is needed for training Meta-Learner")

        self.train_state_tensors = []
        for layer in self.train_model.layers:
            if layer.stateful:
                self.train_state_tensors += layer.states

        self.states_placeholder = [tf.placeholder(shape=t.get_shape(), dtype=tf.float32, name='state_ph_{}'.format(i))
                                   for i, t in enumerate(self.predict_model.state_tensors)]

        self.init_train_states_updates = [tf.assign(t, new_t, name='init_train_state_{}'.format(i))
                                          for i, (t, new_t) in
                                          enumerate(zip(self.train_state_tensors, self.states_placeholder))]

        self.learner_grad_placeholder = tf.placeholder(shape=(self.predict_model.output_size,), dtype=tf.float32,
                                                       name='learner_grad_ph')

        with tf.control_dependencies(self.init_train_states_updates):
            self.chained_grads = tf.gradients(self.train_model.output, self.train_model._collected_trainable_weights,
                                              grad_ys=self.learner_grad_placeholder)

        self.batch_grads_placeholder = [tf.placeholder(shape=t.get_shape(), dtype=tf.float32,
                                                       name='batch_grad_ph_{}'.format(i))
                                        for i, t in enumerate(self.train_model._collected_trainable_weights)]

        self.train_updates = self.train_model.optimizer.get_updates_by_grads(
            self.batch_grads_placeholder, self.train_model._collected_trainable_weights)

        self.copy_weights_updates = [tf.assign(w, train_w, name='copy_weights_train_to_pred_{}'.format(i))
                                     for i, (w, train_w) in
                                     enumerate(zip(self.predict_model._collected_trainable_weights,
                                                   self.train_model._collected_trainable_weights))]
        self.copy_train_weights()

        # tensors used only in debug mode
        if self.debug_mode:
            self.predict_state_tensors = []
            for layer in self.predict_model.layers:
                if layer.stateful:
                    self.predict_state_tensors += layer.states

            self.init_predict_states_updates = [tf.assign(t, new_t, name='init_pred_state_{}'.format(i))
                                                for i, (t, new_t) in
                                                enumerate(zip(self.predict_state_tensors, self.states_placeholder))]

    def copy_train_weights(self):
        K.get_session().run(self.copy_weights_updates)

    def train_on_batch(self, samples: List[MetaTrainingSample]):
        state_tensors = self.predict_model.state_tensors
        input_tensors = get_input_tensors(self.train_model)

        sess = K.get_session()

        batch_grads = []

        for sample in samples:
            # initialize states and train meta-learner using chained gradients of learner loss and meta-learner output
            inputs = standardize_predict_inputs(self.train_model, sample.inputs)

            assert len(state_tensors) == len(sample.initial_states)
            feed_dict = {}

            for state_ph, state_val in zip(self.states_placeholder, sample.initial_states):
                feed_dict[state_ph] = state_val

            for input_tens, input_val in zip(input_tensors, inputs):
                feed_dict[input_tens] = input_val

            feed_dict[self.learner_grad_placeholder] = sample.learner_grads

            evaluation = sess.run(self.chained_grads + self.init_train_states_updates, feed_dict=feed_dict)
            sample_grads = evaluation[:len(self.chained_grads)]

            # batch_grads = mean of sample grads
            if len(batch_grads) == 0:
                batch_grads = sample_grads
            else:
                for i, g in enumerate(sample_grads):
                    batch_grads[i] += g

        for i in range(len(batch_grads)):
            batch_grads[i] /= len(samples)

        # train on batch_grads using custom optimizer
        feed_dict = dict(zip(self.batch_grads_placeholder, batch_grads))
        sess.run(self.train_updates, feed_dict)

        # copy weights from 'train' meta_learner to 'predict' meta_learner
        self.copy_train_weights()
