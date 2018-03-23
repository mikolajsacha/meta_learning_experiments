""" Comparing trained meta-learner to a normal optimizer """
from typing import List, Callable

from itertools import islice
from keras import Model
from tqdm import tqdm
import numpy as np
import math
from keras.optimizers import Optimizer

from src.datasets.metadataset import MetaLearnerDataset


def learning_rate_grid_search(optimizer_factory: Callable[[float], Optimizer],
                              meta_dataset: MetaLearnerDataset,
                              lr_values: List[float],
                              n_learner_batches: int,
                              learner_batch_size: int,
                              learner: Model,
                              trainings_per_dataset: int,
                              initial_learner_weights: List[np.ndarray]) -> float:
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
                learner.set_weights(initial_learner_weights)
                if isinstance(learner.optimizer, Model):
                    learner.optimizer.reset_states()
                learner.fit_generator(
                    generator=learner_dataset.train_set.batch_generator(batch_size=learner_batch_size, randomize=True),
                    steps_per_epoch=n_learner_batches,
                    epochs=1,
                    verbose=0
                )
                total_loss += learner.evaluate(valid_batch_x, valid_batch_y, verbose=0)
                prg_bar.update(1)

        if total_loss < best_loss:
            best_loss = total_loss
            best_lr = lr

    prg_bar.close()

    return best_lr


def compare_optimizers(meta_dataset: MetaLearnerDataset,
                       optimizer_factories: List[Callable[[], Optimizer]],
                       n_learner_batches: int,
                       learner_batch_size: int,
                       learner: Model,
                       trainings_per_dataset: int,
                       initial_learner_weights: List[np.ndarray]) -> List[List[float]]:
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
            for i, optimizer_factory in enumerate(optimizer_factories):
                # use same batches for all optimizers
                learner.optimizer = optimizer_factory()
                learner.set_weights(initial_learner_weights)
                learner.fit_generator(
                    generator=(i for i in training_batches),
                    steps_per_epoch=n_learner_batches,
                    epochs=1,
                    verbose=0
                )
                loss = learner.evaluate(valid_batch_x, valid_batch_y, verbose=0)
                losses[i].append(loss)
                prg_bar.update(1)

    prg_bar.close()

    return losses
