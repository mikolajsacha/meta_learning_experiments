"""
Classes for storing data for Meta-Learning
"""
import logging
from itertools import cycle, islice
from typing import List, Tuple, Generator, Optional
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
import pandas as pd
import scipy.stats


def entropy(data: np.ndarray):
    counts = np.sum(data, axis=0)
    return scipy.stats.entropy(counts)


class Dataset(object):
    """
    Represents a single labeled dataset
    """

    def __init__(
            self,
            x: np.ndarray,
            x_ind: np.ndarray,
            y: np.ndarray):
        """
        :param x: matrix of inputs x
        :param x_ind: vector of indices of inputs in the whole input meta dataset
        :param y: matrix of outputs y (as one hot vectors)
        """
        assert y.ndim == 2 and x.ndim > 1
        assert y.shape[1] > 1
        assert x_ind.ndim == 1
        assert x.shape[0] == y.shape[0]

        self.n_samples = x.shape[0]
        self.n_classes = y.shape[1]
        self.x = x
        self.x_ind = x_ind
        self.y = y

    def random_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param batch_size: size of the batch (> 0)
        :return: a random batch of samples as tuple (x, y)
        """
        if batch_size > self.n_samples:
            raise ValueError("Batch size is greater than size of the dataset!")
        random_indices = np.random.choice(self.n_samples, size=batch_size, replace=False)
        return self.x[random_indices], self.y[random_indices]

    def batch_generator(self, batch_size: int, randomize: bool=False) -> \
            Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if randomize:
            while True:
                yield self.random_batch(batch_size)
        else:
            cycle_ind = cycle(range(self.n_samples))
            while True:
                batch_ind = list(islice(cycle_ind, batch_size))
                yield self.x[batch_ind], self.y[batch_ind]

    def __repr__(self):
        return "Dataset: {:4d} (x,y) samples, {:2d} classes, {:4d} features". \
            format(self.n_samples, self.n_classes, self.x.shape[1])


class LearnerDataset(object):
    """
    Dataset for a single learner, with train/test datasets.
    """

    def __init__(
            self,
            train_set: Dataset,
            test_set: Dataset,
            labels_mapping: Optional[List[int]] = None):  # not sure if labels_mapping will be needed, so Optional
        """
        :param train_set: train Dataset
        :param test_set: test Dataset
        :param labels_mapping: mapping from labels of Learner [0,1,...] to real labels
        """
        self.train_set = train_set
        self.test_set = test_set
        self.labels_mapping = labels_mapping

    def __repr__(self):
        if self.labels_mapping is None:
            labels_mapping = ""
        else:
            labels_mapping = "    labels mapping: " + \
                             ", ".join("{}->{}".format(i, l) for i, l in enumerate(self.labels_mapping))

        return "LearnerDataset:\n    train set: {:s}\n    test set:  {:s}\n" \
                   .format(str(self.train_set), str(self.test_set)) + labels_mapping


class MetaLearnerDataset(object):
    """
    Dataset for training a meta-learner. It contains several LearnerDatasets with train and test sets
    """

    def __init__(
            self,
            meta_train_set: List[LearnerDataset],
            meta_test_set: List[LearnerDataset]):
        """
        :param meta_train_set: list of LearnerDataset for meta-train set
        :param meta_test_set: list of LearnerDataset for meta-test set
        """
        self.meta_train_set = meta_train_set
        self.meta_test_set = meta_test_set

    def __repr__(self):
        def pretty_meta_datasets(datasets):
            return '\n'.join("  {:4<d}. {:s}".format(i, str(s)) for i, s in enumerate(datasets))

        return ("*" * 60 + "\nMetaLearnerDataset:\n\nmeta-train: {:4d} LearnerDatasets:\n{}\n" +
                "\nmeta-test:  {:4d} LearnerDatasets:\n{}\n" + "*" * 60 + "\n"). \
            format(len(self.meta_train_set), pretty_meta_datasets(self.meta_train_set),
                   len(self.meta_test_set), pretty_meta_datasets(self.meta_test_set))

    def save(self, path: str):
        store = pd.HDFStore(path, 'w')
        for meta_set_key, meta_set in [('train', self.meta_train_set), ('test', self.meta_test_set)]:
            for i, learner_set in enumerate(meta_set):
                for set_key in ['train', 'test']:
                    full_key = "{}_{}_{}".format(meta_set_key, set_key, i)
                    data_set = learner_set.train_set if set_key == 'train' else learner_set.test_set
                    y = np.argmax(data_set.y, axis=1)  # from categorical to 1D vector
                    d = {'x_ind': data_set.x_ind, 'y': y}
                    df = pd.DataFrame(d)
                    store[full_key] = df
        store.close()


def load_meta_dataset(path: str, full_x: np.ndarray) -> MetaLearnerDataset:
    meta_train_train, meta_train_test = [], []
    meta_test_train, meta_test_test = [], []

    store = pd.HDFStore(path, 'r')

    for h5_key in store.keys():
        h5_key = h5_key[1:]
        key_split = h5_key.split('_')
        h5_set = store[h5_key]

        data_set = Dataset(
            x=full_x[h5_set['x_ind'][:]],
            x_ind=h5_set['x_ind'][:],
            y=to_categorical(h5_set['y'][:])
        )

        if key_split[0] == 'train':
            if key_split[1] == 'train':
                meta_train_train.append(data_set)
            else:
                meta_train_test.append(data_set)
        else:
            if key_split[1] == 'train':
                meta_test_train.append(data_set)
            else:
                meta_test_test.append(data_set)

    store.close()

    assert len(meta_train_train) == len(meta_train_test)
    assert len(meta_test_train) == len(meta_test_test)

    meta_train, meta_test = [], []

    for train, test in zip(meta_train_train, meta_train_test):
        mean, std = np.mean(train.x, axis=0), np.std(train.x, axis=0)
        train.x = (train.x - mean) / std
        test.x = (test.x - mean) / std
        meta_train.append(LearnerDataset(train_set=train, test_set=test))

    for train, test in zip(meta_test_train, meta_test_test):
        mean, std = np.mean(train.x, axis=0), np.std(train.x, axis=0)
        train.x = (train.x - mean) / std
        test.x = (test.x - mean) / std
        meta_test.append(LearnerDataset(train_set=train, test_set=test))

    return MetaLearnerDataset(meta_train_set=meta_train, meta_test_set=meta_test)


class MetaLearningDatasetFactory(object):
    """
    Takes full train and test datasets and generates randomized meta-learning datasets by splitting them randomly
    into several tasks with train/test sets
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            meta_test_ratio: float,
            learner_train_size: int,
            learner_test_size: int,
            classes_per_learner_set: int,
            n_train_sets: int,
            n_test_sets: int,
            logger: logging.Logger):
        """
        :param x: X matrix (samples)
        :param y: Y matrix (labels/classes)
        :param meta_test_ratio: proportion of classes to put in meta-test dataset
        :param learner_train_size: size of the training set for a learner dataset, i. e. number of 'shots'
        :param learner_test_size: size of the test set for a learner dataset
        :param classes_per_learner_set: number of classes in each of learner datasets
        :param n_train_sets: number of learner datasets in meta-training collection
        :param n_test_sets: number of learner datasets in meta-test collection
        :param logger: Logger
        """

        self.logger = logger

        self.x = x
        self.y = y

        # total number of classes (assume classes are numbered 0,1,2,n-1)
        self.n_classes = int(np.max(y) + 1)
        self.logger.debug("Found {} classes".format(self.n_classes))

        smaller_set_ratio = min(meta_test_ratio, 1 - meta_test_ratio)
        smaller_set_n_classes = int(smaller_set_ratio * self.n_classes)
        if classes_per_learner_set > smaller_set_n_classes:
            raise ValueError("To many classes per learner dataset ({:d}), ".format(classes_per_learner_set) +
                             "where smaller of meta-sets has only {:d} classes!".format(smaller_set_n_classes))

        self.classes_per_learner_set = classes_per_learner_set

        self.meta_test_ratio = meta_test_ratio
        self.learner_train_size = learner_train_size
        self.learner_test_size = learner_test_size
        self.n_train_sets = n_train_sets
        self.n_test_sets = n_test_sets

    def get(self) -> MetaLearnerDataset:
        """
        Generates a randomized MetaLearningDataset, which consists of several tasks with train/test sets.
        I use method from this paper: https://openreview.net/pdf?id=rJY0-Kcll
        :return: MetaLearnerDataset
        """
        # 1) Split classes into meta-training and meta-test datasets
        randomized_classes = np.random.permutation(self.n_classes)

        n_train_classes = int((1 - self.meta_test_ratio) * self.n_classes)
        n_test_classes = self.n_classes - n_train_classes

        train_classes = list(sorted(randomized_classes[0:n_train_classes]))
        test_classes = list(sorted(randomized_classes[n_train_classes:]))

        self.logger.info("{:d} train classes, {:d} test classes".format(n_train_classes, n_test_classes))

        # 2) Generate LearnerDatasets for each of the class group
        meta_train_set = []
        meta_test_set = []

        class_indices = [np.array([i for i, y in enumerate(self.y) if y == k]) for k in range(self.n_classes)]

        # generate learner datasets
        for m_set, classes, n, name in \
                [(meta_train_set, train_classes, self.n_train_sets, "train"),
                 (meta_test_set, test_classes, self.n_test_sets, "test")]:

            for _ in tqdm(range(n), desc="Generating LearnerDatasets... ({})".format(name)):
                learner_classes = np.random.choice(classes, size=self.classes_per_learner_set, replace=False)

                total_learner_size = self.learner_train_size + self.learner_test_size
                samples_per_class = total_learner_size // self.classes_per_learner_set

                # generate samples for Learner so classes are equally represented (data set is not skewed)
                learner_ind = np.empty(total_learner_size, dtype=np.int)
                i = 0
                for cl in learner_classes[:-1]:
                    learner_ind[i:i + samples_per_class] = np.random.choice(class_indices[cl], size=samples_per_class,
                                                                            replace=False)
                    i += samples_per_class

                # if total_learner_size / self.classes_per_learner_set is not integer, fill up with last class
                learner_ind[i:] = np.random.choice(class_indices[learner_classes[-1]], size=total_learner_size - i,
                                                   replace=False)
                np.random.shuffle(learner_ind)

                learner_train_ind = learner_ind[:self.learner_train_size]
                learner_test_ind = learner_ind[self.learner_train_size:]

                # normalize features for each dataset separately, using metrics from train set
                mean = np.mean(self.x[learner_train_ind], axis=0)
                std = np.std(self.x[learner_train_ind], axis=0)

                learner_train_x = (self.x[learner_train_ind] - mean) / std
                learner_test_x = (self.x[learner_test_ind] - mean) / std

                # map labels so they have numbers from [0, self.classes_per_learner_set]
                reverse_ind_map = dict((l, i) for i, l in enumerate(learner_classes))

                # convert labels to one-hot vectors
                learner_train_y = np.zeros((self.learner_train_size, self.classes_per_learner_set))
                learner_test_y = np.zeros((self.learner_test_size, self.classes_per_learner_set))

                for y_set, ind in [(learner_train_y, learner_train_ind), (learner_test_y, learner_test_ind)]:
                    for i in range(ind.size):
                        y_orig = self.y[ind[i]]
                        y_mapped = reverse_ind_map[y_orig]
                        y_set[i][y_mapped] = 1  # convert label to one-hot vector

                ent = entropy(learner_train_y)
                min_ent = 0.5
                if ent < min_ent:
                    raise ValueError("One of the train dataset turned out to be too skewed (entropy: {})".format(ent))

                ent = entropy(learner_test_y)
                if ent < min_ent:
                    raise ValueError("One of the test dataset turned out to be too skewed (entropy: {})".format(ent))

                m_set.append(LearnerDataset(
                    train_set=Dataset(learner_train_x, learner_train_ind, learner_train_y),
                    test_set=Dataset(learner_test_x, learner_test_ind, learner_test_y),
                    labels_mapping=learner_classes
                ))

        return MetaLearnerDataset(meta_train_set=meta_train_set, meta_test_set=meta_test_set)
