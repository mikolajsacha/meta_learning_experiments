import os
import random
from typing import Callable, Optional, List, Tuple

import time

from itertools import islice

from keras.optimizers import SGD
from tqdm import tqdm

from keras import Model
import keras.backend as K

from src.datasets.metadataset import MetaLearnerDataset
from src.isotropy.lanczos import TopKEigenvaluesBatched
from src.model.custom_optimizers import CustomAdam
from src.model.meta_learner.meta_model import MetaLearnerModel
from src.model.meta_learner.meta_training_sample import MetaTrainingSample
from src.model.util import get_trainable_params_count
from src.training.training_configuration import TrainingConfiguration
from src.utils.testing import gradient_check
from shutil import copyfile


def reset_weights(model: Model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)


class MetaLearningTask(object):
    """ Represents a single meta-learning tasks, which contains a meta-learner, meta-dataset and learner model. """

    def __init__(
            self,
            meta_dataset: MetaLearnerDataset,
            learner_factory: Callable[[], Model],
            meta_learner_factory: Callable[[Model, TopKEigenvaluesBatched], MetaLearnerModel],
            configuration: TrainingConfiguration,
            training_history_path: str,
            meta_learner_weights_path: str,
            meta_learner_weights_history_dir: str,
            best_meta_learner_weights_path: str,
            task_checkpoint_path: Optional[str] = None):
        """
        :param meta_dataset: meta-dataset containing learner-datasets for the task
        :param learner_factory: method that returns instances of Learner Model
        :param meta_learner_factory: method that returns instances of meta-learner Model
        :param configuration: TrainingConfiguration
        :param training_history_path: path to output file with training history
        :param meta_learner_weights_path: path to output file with current MetaLearner weights
        :param meta_learner_weights_history_dir: path to directory with history of MetaLearner weights
        :param best_meta_learner_weights_path: path to output file with MetaLearner weights that had best valid. loss
        :param task_checkpoint_path: optional path to file with checkpoint data to save task/continue interrupted task
        """
        self.meta_dataset = meta_dataset

        self.training_history_path = training_history_path
        self.meta_learner_weights_path = meta_learner_weights_path
        self.meta_learner_weights_history_dir = meta_learner_weights_history_dir
        self.best_meta_learner_weights_path = best_meta_learner_weights_path
        self.task_checkpoint_path = task_checkpoint_path

        if not os.path.exists(self.meta_learner_weights_history_dir):
            os.makedirs(self.meta_learner_weights_history_dir)

        self.learner = None

        self.meta_learner = None
        self.optimizer_weights = None
        self.learner_factory = learner_factory
        self.meta_learner_factory = meta_learner_factory
        self.best_loss = None

        self.configuration = configuration
        self.debug_mode = configuration.debug_mode

        self.logger = configuration.logger

        if not configuration.continue_task or not task_checkpoint_path or not os.path.isfile(task_checkpoint_path):
            self.logger.info('Initializing new MetaLearningTask')

            if task_checkpoint_path:
                with open(self.task_checkpoint_path, 'w') as f:
                    f.write("0")

            self.starting_epoch = 0
        else:
            for i, line in enumerate(open(self.task_checkpoint_path, 'r')):
                if i == 0:
                    self.starting_epoch = int(line)
                elif i == 1:
                    self.best_loss = float(line)

            self.logger.info('Continuing MetaLearningTask after {} epochs'.format(self.starting_epoch))

        self.backpropagation_depth = configuration.backpropagation_depth
        self.backpropagation_padding = configuration.backpropagation_padding
        self.random_seed = configuration.random_seed

        self.eigenvals_callback = None

    def meta_train_step(
            self,
            learner_ind: int,
            learner_batch_size: int,
            n_learner_batches: int) -> Tuple[List[MetaTrainingSample], List[float], List[float]]:
        """
        Runs a single step of meta-learning, by training a random Learner model for a few batches
        :param learner_ind: index of meta-train Learner to use for meta-learning step
        :param learner_batch_size: size of a training batch for Learner
        :param n_learner_batches: number of training batches
        :return tuple of [MetaTrainingSamples for BPTT on Meta-Learner, train and valid. metrics on the batch]
        """
        if self.backpropagation_depth > n_learner_batches:
            raise ValueError("backpropagation depth can't be greater than number of training batches!")

        self.logger.debug("Training meta-optimizer using train set {:d}".format(learner_ind))

        learner_dataset = self.meta_dataset.meta_test_set[learner_ind]

        self.meta_learner.reset_states()
        self.learner.load_weights(os.path.join(os.environ['CONF_DIR'], 'initial_learner_weights.h5'))

        train_batch_x, train_batch_y = learner_dataset.train_set.x, learner_dataset.train_set.y
        valid_batch_x, valid_batch_y = learner_dataset.test_set.x, learner_dataset.test_set.y

        self.eigenvals_callback.X = train_batch_x
        self.eigenvals_callback.y = train_batch_y

        training_samples = []
        batch_generator = learner_dataset.train_set.batch_generator(batch_size=learner_batch_size, randomize=True)
        learner_training_batches = list(islice(batch_generator, n_learner_batches))

        for i, (x, y) in enumerate(learner_training_batches[:self.backpropagation_depth]):
            self.learner.train_on_batch(x, y)

        training_sample = self.meta_learner.predict_model.retrieve_training_sample(
            valid_batch_x, valid_batch_y, learner_training_batches[:self.backpropagation_depth])
        training_samples.append(training_sample)

        for i, (x, y) in enumerate(learner_training_batches[self.backpropagation_depth:]):
            self.learner.train_on_batch(x, y)
            if (i + 1) % self.backpropagation_padding == 0:
                current_training_batches = learner_training_batches[i + 1:i + 1 + self.backpropagation_depth]
                training_sample = self.meta_learner.predict_model.retrieve_training_sample(
                    valid_batch_x, valid_batch_y, current_training_batches)
                training_samples.append(training_sample)

        meta_evaluation = (self.learner.evaluate(train_batch_x, train_batch_y, verbose=0),
                           self.learner.evaluate(valid_batch_x, valid_batch_y, verbose=0))

        if self.debug_mode:
            for sample in training_samples:
                gradient_check(self.meta_learner, sample, self.logger)

        return (training_samples,) + meta_evaluation

    def meta_valid_step(
            self,
            epoch: int,
            step: int,
            learner_ind: int,
            learner_batch_size: int,
            n_learner_batches: int) -> Tuple[List[float], List[float]]:
        """
        Evaluates meta-learning by using meta-optimizer to train a selected Learner (from meta-test Learners)
        and measuring its performance on its test dataset
        :param epoch: number of current epoch
        :param step: number of current step in epoch
        :param learner_ind: index of meta-test Learner to use for meta-learning step
        :param learner_batch_size: size of a training batch for Learner
        :param n_learner_batches: number of training batches
        :return metrics of meta-optimized model
        """
        self.logger.debug("Evaluating meta-optimizer using validation set {:d}".format(learner_ind))

        learner_dataset = self.meta_dataset.meta_test_set[learner_ind]

        train_batch_x, train_batch_y = learner_dataset.train_set.x, learner_dataset.train_set.y
        valid_batch_x, valid_batch_y = learner_dataset.test_set.x, learner_dataset.test_set.y

        self.meta_learner.reset_states()
        self.learner.load_weights(os.path.join(os.environ['CONF_DIR'], 'initial_learner_weights.h5'))

        self.eigenvals_callback.X = train_batch_x
        self.eigenvals_callback.y = train_batch_y

        self.learner.fit_generator(
            generator=learner_dataset.train_set.batch_generator(batch_size=learner_batch_size, randomize=True),
            steps_per_epoch=n_learner_batches,
            epochs=1,
            verbose=0
        )

        # after each meta-valid step, save Hessian eigenvalues and some training statistics
        eigen_save_path = os.path.join(os.environ['LOG_DIR'],
                                       'eigenvals/epoch_{}/step_{}_top_K_ev.npz'.format(epoch+1, step))
        if not os.path.exists(os.path.dirname(eigen_save_path)):
            os.makedirs(os.path.dirname(eigen_save_path))
        self.eigenvals_callback.save(eigen_save_path, with_vectors=False)

        stats_path = os.path.join(os.environ['LOG_DIR'], 'stats/epoch_{}/step_{}_stats.npz'.format(epoch+1, step))
        if not os.path.exists(os.path.dirname(stats_path)):
            os.makedirs(os.path.dirname(stats_path))
        self.meta_learner.predict_model.save_statistics(stats_path)

        # evaluate trained learner on train and valid sets
        meta_evaluation = (self.learner.evaluate(train_batch_x, train_batch_y, verbose=0),
                           self.learner.evaluate(valid_batch_x, valid_batch_y, verbose=0))

        return meta_evaluation

    def meta_validate_epoch(
            self,
            epoch_number: int,
            n_meta_valid_steps: int,
            n_learner_batches: int,
            learner_batch_size: int,
    ) -> List[float]:
        """
        Runs a single "meta-epoch' of validating meta-learning model
        :param epoch_number: epoch number for displaying (optional)
        :param n_meta_valid_steps: number of validation Learner trainings
        :param n_learner_batches: number of training batches
        :param learner_batch_size: size of a training batch for Learner
        :return Lists of average values of valid. metrics
        """
        # set train_mode = False for faster prediction on meta-validation set
        self.meta_learner.train_mode = False

        # get loss/metrics estimation on meta-validation set
        batch_train_metrics = [0.0 for _ in range(len(self.learner.metrics_names))]
        batch_valid_metrics = [0.0 for _ in range(len(self.learner.metrics_names))]
        batch_metrics_ratios = [0.0 for _ in range(len(self.learner.metrics_names))]

        desc = 'Meta-Validating' if epoch_number is None else 'Meta-Validating (meta-epoch {})'.format(epoch_number + 1)
        random_learners_ind = random.sample(range(len(self.meta_dataset.meta_test_set)), n_meta_valid_steps)
        for i, ind in enumerate(tqdm(random_learners_ind, desc=desc)):
            train_metrics, valid_metrics = self.meta_valid_step(
                epoch=epoch_number,
                step=i,
                learner_ind=ind,
                learner_batch_size=learner_batch_size,
                n_learner_batches=n_learner_batches
            )
            for j, (train_val, valid_val) in enumerate(zip(train_metrics, valid_metrics)):
                batch_train_metrics[j] += train_val
                batch_valid_metrics[j] += valid_val
                batch_metrics_ratios[j] += valid_val / train_val

        for i in range(len(self.learner.metrics_names)):
            batch_train_metrics[i] /= n_meta_valid_steps
            batch_valid_metrics[i] /= n_meta_valid_steps
            batch_metrics_ratios[i] /= n_meta_valid_steps

        metrics_names = self.learner.metrics_names

        with open(self.training_history_path, 'a') as f:
            for name, metrics in [('TRAIN', batch_train_metrics),
                                  ('VALID', batch_valid_metrics),
                                  ('RATIO', batch_metrics_ratios)]:
                msg = "META_VALID_{}: ".format(name) + ",".join(map(str, metrics))
                self.logger.debug("META " + msg)
                f.write(msg)
                f.write('\n')

        self.logger.info("Valid. metrics: " + ", ".join("{}: {}".format(name, round(float(value), 5))
                                                        for name, value in
                                                        zip(metrics_names, batch_valid_metrics)))

        return batch_valid_metrics

    def meta_train_epoch(
            self,
            meta_batch_size: int,
            n_meta_train_steps: int,
            n_learner_batches: int,
            learner_batch_size: int,
            epoch_number: Optional[int] = None):

        """
        Runs a single "meta-epoch' of training meta-learning model
        :param meta_batch_size: size of meta-batch of Learners per one meta-optimizer weight update
        :param n_meta_train_steps: number of meta-training steps
        :param n_learner_batches: number of training batches
        :param learner_batch_size: size of a training batch for Learner
        :param epoch_number: epoch number for displaying (optional)
        """
        self.meta_learner.train_mode = True

        desc = 'Meta-Training' if epoch_number is None else 'Meta-Training (meta-epoch {})'.format(epoch_number + 1)
        random_learners_ind = random.sample(range(len(self.meta_dataset.meta_train_set)),
                                            n_meta_train_steps * meta_batch_size)
        learner_ind = 0
        for i in tqdm(range(n_meta_train_steps), desc=desc):
            training_samples = []

            batch_train_metrics = [0.0 for _ in range(len(self.learner.metrics_names))]
            batch_valid_metrics = [0.0 for _ in range(len(self.learner.metrics_names))]
            batch_metrics_ratios = [0.0 for _ in range(len(self.learner.metrics_names))]

            for _ in tqdm(range(meta_batch_size), desc='batch {} / epoch {}'.format(i + 1, epoch_number + 1)):
                meta_training_samples, train_metrics, valid_metrics = self.meta_train_step(
                    learner_ind=random_learners_ind[learner_ind],
                    n_learner_batches=n_learner_batches,
                    learner_batch_size=learner_batch_size)

                training_samples += meta_training_samples

                for j, (train_metric, valid_metric) in enumerate(zip(train_metrics, valid_metrics)):
                    batch_train_metrics[j] += train_metric
                    batch_valid_metrics[j] += valid_metric
                    batch_metrics_ratios[j] += valid_metric / train_metric

                learner_ind += 1

            for j in range(len(batch_valid_metrics)):
                batch_valid_metrics[j] /= meta_batch_size
                batch_train_metrics[j] /= meta_batch_size
                batch_metrics_ratios[j] /= meta_batch_size

            self.meta_learner.train_on_batch(training_samples)
            self.meta_learner.train_model.save(self.meta_learner_weights_path)
            self.optimizer_weights = self.meta_learner.train_model.optimizer.get_weights()

            with open(self.training_history_path, 'a') as f:
                for name, metrics in [('TRAIN', batch_train_metrics),
                                      ('VALID', batch_valid_metrics),
                                      ('RATIO', batch_metrics_ratios)]:
                    msg = "META_TRAIN_{}: ".format(name) + ",".join(map(str, metrics))
                    self.logger.debug("META " + msg)
                    f.write(msg)
                    f.write('\n')

    def _compile(self, meta_lr, epoch, learner_batch_size):
        K.clear_session()

        self.learner = self.learner_factory()

        # we need to compile learner before constructing meta_learner, and then set optimizer to meta_learner
        self.learner.compile(loss='categorical_crossentropy',
                             optimizer=SGD(lr=0.0),  # dummy optimizer
                             metrics=['accuracy'])

        eigenval_features = self.configuration.hessian_eigenvalue_features
        self.eigenvals_callback = TopKEigenvaluesBatched(K=4,
                                                         feature_K=eigenval_features,
                                                         batch_size=learner_batch_size, logger=self.logger,
                                                         save_dir="", save_eigenv=1)
        self.eigenvals_callback.model = self.learner
        self.eigenvals_callback.compile()

        if epoch == 0:
            self.eigenvals_callback.save_param_shapes(os.path.join(os.environ['LOG_DIR'], "top_K_ev_mapping.pkl"))

        self.meta_learner = self.meta_learner_factory(self.learner, self.eigenvals_callback)

        self.meta_learner.predict_model.compile(loss='mae',  # we don't use loss here anyway
                                                optimizer=SGD(lr=0.0),  # dummy optimizer
                                                metrics=[])

        self.meta_learner.train_model.compile(loss='mae',  # we don't use loss here anyway
                                              optimizer=CustomAdam(lr=meta_lr, clipvalue=0.25),
                                              metrics=[])

        # now set real optimizer to be our meta-learner predict model
        self.learner.optimizer = self.meta_learner.predict_model

        # initialize some additional tensors used for meta-training
        self.meta_learner.compile()

        if epoch > 0 and os.path.isfile(self.meta_learner_weights_path):
            self.meta_learner.load_weights(self.meta_learner_weights_path)

    def meta_train(
            self,
            lr_scheduler: Callable[[int], float],
            n_meta_epochs: int,
            meta_batch_size: int,
            n_learner_batches: int,
            meta_early_stopping: int,
            n_meta_train_steps: int,
            n_meta_valid_steps: int,
            learner_batch_size: int):
        """
        Trains meta-learning model for a few epochs
        :param lr_scheduler: function that takes number of meta-epoch and returns meta-learning rate
        :param n_meta_epochs: number of meta-epochs
        :param meta_batch_size: size of meta-batch of Learners per one meta-optimizer weight update
        :param n_learner_batches: number of training batches
        :param meta_early_stopping: early stopping patience for Learner training
        :param n_meta_train_steps: number of meta-training batches per epoch
        :param n_meta_valid_steps: number of meta-validation batches per epoch
        :param learner_batch_size: batch size when training Learner
        """
        lr = lr_scheduler(self.starting_epoch)
        self.logger.info("Starting with meta-learning rate: {}".format(lr))

        epochs_with_no_gain = 0

        self._compile(lr, self.starting_epoch, learner_batch_size)

        if self.starting_epoch == 0:
            n_params = get_trainable_params_count(self.meta_learner.train_model)

            self.logger.info("Using Meta-Learner with {} parameters".format(n_params))
            self.meta_learner.train_model.summary()

            # save initial meta-learner weights
            self.meta_learner.train_model.save(os.path.join(self.meta_learner_weights_history_dir,
                                                            'meta_weights_epoch_0.h5'))

            # copy training configuration to log dir
            copyfile(os.path.join(os.environ['CONF_DIR'], 'training_configuration.yml'),
                     os.path.join(os.environ['LOG_DIR'], 'training_configuration.yml'))
            self.logger.info("Validating Meta-Learner on start...")
            self.meta_validate_epoch(
                n_meta_valid_steps=n_meta_valid_steps,
                n_learner_batches=n_learner_batches,
                learner_batch_size=learner_batch_size,
                epoch_number=-1)

        for i in tqdm(range(n_meta_epochs - self.starting_epoch), desc='Running Meta-Training'):
            epoch = self.starting_epoch + i
            epoch_start = time.time()

            # reset backend session each epoch to avoid memory leaks etc
            self._compile(lr, epoch, learner_batch_size)

            if self.best_loss is None:
                self.logger.info("Starting meta-epoch {:d}".format(epoch + 1))
            else:
                self.logger.info("Starting meta-epoch {:d} (best loss: {:.5f})".format(epoch + 1, self.best_loss))

            if self.optimizer_weights is not None:
                self.meta_learner.train_model.optimizer.set_weights(self.optimizer_weights)

            prev_lr = lr
            lr = lr_scheduler(epoch)

            if prev_lr != lr:
                self.logger.info("Changing meta-learning rate from {} to {} (meta-epoch {})"
                                 .format(prev_lr, lr, epoch + 1))
                self.meta_learner.train_model.optimizer.update_lr(lr)

            self.meta_train_epoch(
                meta_batch_size=meta_batch_size,
                n_learner_batches=n_learner_batches,
                n_meta_train_steps=n_meta_train_steps,
                learner_batch_size=learner_batch_size,
                epoch_number=epoch)

            self.meta_learner.train_model.save(os.path.join(self.meta_learner_weights_history_dir,
                                                            'meta_weights_epoch_{}.h5'.format(epoch+1)))

            valid_metrics = self.meta_validate_epoch(
                n_learner_batches=n_learner_batches,
                n_meta_valid_steps=n_meta_valid_steps,
                learner_batch_size=learner_batch_size,
                epoch_number=epoch)

            epoch_duration = round(time.time() - epoch_start, 2)
            self.logger.info("Duration of epoch {}: {} s".format(epoch + 1, epoch_duration))
            self.logger.info("*" * 50)

            loss_ind = self.learner.metrics_names.index('loss')
            if self.best_loss is None or valid_metrics[loss_ind] < self.best_loss:
                self.best_loss = valid_metrics[loss_ind]
                epochs_with_no_gain = 0
                self.meta_learner.train_model.save(self.best_meta_learner_weights_path)
            else:
                epochs_with_no_gain += 1

            with open(self.task_checkpoint_path, 'w') as f:
                f.write(str(epoch + 1))
                f.write('\n')
                f.write(str(self.best_loss))

            if epochs_with_no_gain >= meta_early_stopping:
                self.logger.info("Early stopping after {} meta-epochs".format(epoch + 1))
                self.logger.info("*" * 30)
                break
