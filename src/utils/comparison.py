""" Comparing trained meta-learner to a normal optimizer """
from typing import List, Callable, Optional

from itertools import islice
from keras import Model
from tqdm import tqdm
import numpy as np
import math
from keras.optimizers import Optimizer

from src.datasets.metadataset import MetaLearnerDataset
from src.isotropy.lanczos import TopKEigenvaluesBatched
from src.training.meta_learning_task import reset_weights


def learning_rate_grid_search(optimizer_factory: Callable[[float], Optimizer],
                              meta_dataset: MetaLearnerDataset,
                              lr_values: List[float],
                              n_learner_batches: int,
                              learner_batch_size: int,
                              learner: Model,
                              trainings_per_dataset: int,
                              initial_learner_weights: Optional[List[np.ndarray]] = None) -> float:
    """
    Performs grid-search on meta-train set to find best learning rate of optimizer and saves results on valid. set
    :param optimizer_factory: method that returns Optimizer for a given learning rate
    :param meta_dataset: MetaLearnerDataset to get data from
    :param lr_values: list of all values of learning rate to be tested
    :param n_learner_batches: number of training batches for a single Learner
    :param learner_batch_size: batch size of Learner
    :param learner: model for Learner
    :param trainings_per_dataset: number of trainings per single dataset per lr value
    :param initial_learner_weights: initial weights for training Learner
    :return: best lr value
    """
    assert len(lr_values) > 0

    best_lr = 0.0
    best_loss = math.inf

    prg_bar = tqdm(total=len(lr_values) * len(meta_dataset.meta_train_set) * trainings_per_dataset,
                   desc='Grid-Search of learning rate')

    for lr in lr_values:
        total_loss = 0.0

        for learner_dataset in meta_dataset.meta_train_set:
            valid_batch_x, valid_batch_y = learner_dataset.test_set.x, learner_dataset.test_set.y
            for _ in range(trainings_per_dataset):
                learner.optimizer = optimizer_factory(lr)
                if initial_learner_weights is None:
                    reset_weights(learner)
                else:
                    learner.set_weights(initial_learner_weights)
                if isinstance(learner.optimizer, Model):
                    learner.optimizer.reset_states()
                learner.fit_generator(
                    generator=learner_dataset.train_set.batch_generator(batch_size=learner_batch_size, randomize=True),
                    steps_per_epoch=n_learner_batches,
                    epochs=1,
                    verbose=0
                )
                evaluation = learner.evaluate(valid_batch_x, valid_batch_y, verbose=0)
                if isinstance(evaluation, list):
                    evaluation = evaluation[0]
                total_loss += evaluation
                prg_bar.update(1)

        if total_loss < best_loss:
            best_loss = total_loss
            best_lr = lr

    prg_bar.close()

    return best_lr


def compare_optimizers(meta_dataset: MetaLearnerDataset,
                       optimizer_factories: List[Callable[[np.array, np.array], Optimizer]],
                       n_learner_batches: int,
                       learner_batch_size: int,
                       learner: Model,
                       trainings_per_dataset: int,
                       initial_learner_weights: Optional[List[np.ndarray]] = None) -> List[List[float]]:
    """
    Compares performance of two or more optimizers on meta-valid set
    :param meta_dataset: MetaLearnerDataset to get data from
    :param optimizer_factories: list of functions that generate Optimizers to compare
    :param n_learner_batches: number of training batches for a single Learner
    :param learner_batch_size: batch size of Learner
    :param learner: model for Learner
    :param trainings_per_dataset: number of trainings per single dataset per lr value
    :param initial_learner_weights: initial weights for training Learner
    :return: List of Lists of all acquired valid. losses using optimizers on meta-valid tasks
    """
    losses = [[] for _ in optimizer_factories]

    prg_bar = tqdm(total=len(meta_dataset.meta_test_set * trainings_per_dataset * len(optimizer_factories)),
                   desc='Evaluating optimizers...')

    for learner_dataset in meta_dataset.meta_test_set:
        valid_batch_x, valid_batch_y = learner_dataset.test_set.x, learner_dataset.test_set.y
        train_generator = learner_dataset.train_set.batch_generator(batch_size=learner_batch_size, randomize=True)

        for _ in range(trainings_per_dataset):
            training_batches = list(islice(train_generator, n_learner_batches))
            if initial_learner_weights is None:
                reset_weights(learner)
                current_initial_learner_weights = learner.get_weights()
            for i, optimizer_factory in enumerate(optimizer_factories):
                # use same batches and initial weights for all optimizers
                learner.optimizer = optimizer_factory(learner_dataset.train_set.x, learner_dataset.train_set.y)
                if initial_learner_weights is None:
                    # noinspection PyUnboundLocalVariable
                    learner.set_weights(current_initial_learner_weights)
                else:
                    learner.set_weights(initial_learner_weights)
                learner.fit_generator(
                    generator=(b for b in training_batches),
                    steps_per_epoch=n_learner_batches,
                    epochs=1,
                    verbose=0
                )
                evaluation = learner.evaluate(valid_batch_x, valid_batch_y, verbose=0)
                if isinstance(evaluation, list):
                    evaluation = evaluation[0]
                losses[i].append(evaluation)
                prg_bar.update(1)

    prg_bar.close()

    return losses


def analyze_training(meta_dataset: MetaLearnerDataset,
                     optimizer_factory: Callable[[np.array, np.array], Optimizer],
                     n_learner_batches: int,
                     learner_batch_size: int,
                     learner: Model,
                     trainings_per_dataset: int,
                     initial_learner_weights: Optional[List[np.ndarray]] = None):
    """
    Analyze statistics during training Learner using Meta-Learner
    :param meta_dataset: MetaLearnerDataset to get data from
    :param optimizer_factory: functions that generates Optimizer
    :param n_learner_batches: number of training batches for a single Learner
    :param learner_batch_size: batch size of Learner
    :param learner: model for Learner
    :param trainings_per_dataset: number of trainings per single dataset per lr value
    :param initial_learner_weights: initial weights for training Learner
    :return: Tuple of lists of statistics
    """
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
    hessian_eigen = []

    eigenvals_callback = TopKEigenvaluesBatched(K=4, batch_size=learner_batch_size, logger=None,
                                                save_dir="", save_eigenv=1)
    eigenvals_callback.model = learner
    eigenvals_callback.compile()

    prg_bar = tqdm(total=len(meta_dataset.meta_test_set * trainings_per_dataset), desc='Analyzing Learner trainings...')

    for learner_dataset in meta_dataset.meta_test_set:
        train_x, train_y = learner_dataset.train_set.x, learner_dataset.train_set.y
        valid_x, valid_y = learner_dataset.test_set.x, learner_dataset.test_set.y

        eigenvals_callback.X = train_x
        eigenvals_callback.y = train_y

        for _ in range(trainings_per_dataset):
            current_train_losses, current_train_accuracies = [], []
            current_valid_losses, current_valid_accuracies = [], []
            current_eigen = []

            if initial_learner_weights is None:
                reset_weights(learner)
            else:
                learner.set_weights(initial_learner_weights)

            learner.optimizer = optimizer_factory(train_x, train_y)

            for i, training_batch in enumerate(learner_dataset.train_set.batch_generator(
                    batch_size=learner_batch_size,
                    randomize=True)):
                if i >= n_learner_batches:
                    break
                learner.train_on_batch(training_batch[0], training_batch[1])

                train_evaluation = learner.evaluate(train_x, train_y, verbose=0)
                assert isinstance(train_evaluation, list)
                current_train_losses.append(train_evaluation[0])
                current_train_accuracies.append(train_evaluation[1])

                valid_evaluation = learner.evaluate(valid_x, valid_y, verbose=0)
                assert isinstance(valid_evaluation, list)
                current_valid_losses.append(valid_evaluation[0])
                current_valid_accuracies.append(valid_evaluation[1])

                eigen = np.mean(eigenvals_callback.compute_top_K(with_vectors=False)[:-1])
                current_eigen.append(eigen)

            train_losses.append(current_train_losses)
            train_accuracies.append(current_train_accuracies)
            valid_losses.append(current_valid_losses)
            valid_accuracies.append(current_valid_accuracies)
            hessian_eigen.append(current_eigen)

            prg_bar.update(1)

    prg_bar.close()
    print()

    return train_losses, train_accuracies, valid_losses, valid_accuracies, hessian_eigen
