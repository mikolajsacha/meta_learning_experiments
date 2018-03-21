import os
import logging
from typing import List, Tuple
import yaml

from src.utils.logging import str_to_log_level, configure_logger


class TrainingConfiguration(object):
    def __init__(
            self,
            continue_task: bool,
            debug_mode: bool,
            dataset_key: str,
            logger: logging.Logger,
            lr_schedule: List[Tuple[int, float]],
            logging_level: int,
            meta_batch_size: int,
            learner_batch_size: int,
            n_learner_batches: int,
            backpropagation_depth: int,
            backpropagation_padding: int,
            learner_train_size: int,
            learner_test_size: int,
            classes_per_learner_set: int,
            n_train_sets: int,
            n_test_sets: int,
            n_meta_epochs: int,
            meta_early_stopping: int,
            meta_test_class_ratio: float,
            initial_learner_lr: float,
            hidden_state_size: int,
            n_meta_valid_steps: int):
        self.continue_task = continue_task
        self.debug_mode = debug_mode
        self.lr_schedule = lr_schedule
        self.dataset_key = dataset_key
        self.logger = logger
        self.logging_level = logging_level
        self.meta_batch_size = meta_batch_size
        self.backpropagation_depth = backpropagation_depth
        self.backpropagation_padding = backpropagation_padding
        self.learner_batch_size = learner_batch_size
        self.n_learner_batches = n_learner_batches
        self.learner_train_size = learner_train_size
        self.learner_test_size = learner_test_size
        self.classes_per_learner_set = classes_per_learner_set
        self.n_train_sets = n_train_sets
        self.n_test_sets = n_test_sets
        self.meta_test_class_ratio = meta_test_class_ratio
        self.initial_lr = initial_learner_lr
        self.n_meta_valid_steps = n_meta_valid_steps
        self.n_meta_epochs = n_meta_epochs
        self.meta_early_stopping = meta_early_stopping
        self.hidden_state_size = hidden_state_size

    @property
    def meta_dataset_path(self):
        file_name = "{}_{}_{}_{}.h5".format(self.dataset_key, self.n_train_sets, self.n_test_sets,
                                            self.classes_per_learner_set)
        return os.path.join(os.environ['DATA_DIR'], file_name)

    def log_summary(self):
        params = {
            'continue_task': self.continue_task,
            'debug_mode': self.debug_mode,
            'backpropagation_depth': self.backpropagation_depth,
            'backpropagation_padding': self.backpropagation_padding,
            'lr_schedule': self.lr_schedule,
            'dataset_key': self.dataset_key,
            'logging_level': self.logging_level,
            'meta_batch_size': self.meta_batch_size,
            'learner_batch_size': self.learner_batch_size,
            'n_learner_batches': self.n_learner_batches,
            'learner_train_size': self.learner_train_size,
            'learner_test_size': self.learner_test_size,
            'classes_per_learner_set': self.classes_per_learner_set,
            'n_train_sets': self.n_train_sets,
            'n_test_sets': self.n_test_sets,
            'meta_test_class_ratio': self.meta_test_class_ratio,
            'initial_lr': self.initial_lr,
            'n_meta_valid_steps': self.n_meta_valid_steps,
            'n_meta_epochs': self.n_meta_epochs,
            'meta_early_stopping': self.meta_early_stopping,
            'hidden_state_size': self.hidden_state_size
        }
        msg = '\n'.join("{}: {}".format(k, v) for k, v in sorted(params.items(), key=lambda x: x[0]))
        self.logger.info("TRAINING CONFIGURATION:\n" + msg)


def read_configuration(path: str) -> TrainingConfiguration:
    with open(path, 'r') as stream:
        conf = yaml.load(stream)

    log_level = str_to_log_level(conf['logging_level'])
    logger = configure_logger(name='train-meta-model', level=log_level)
    lr_schedule = list(zip(conf['lr_schedule_epochs'], conf['lr_schedule_rates']))

    return TrainingConfiguration(
        continue_task=conf['continue_task'],
        debug_mode=conf['debug_mode'],
        dataset_key=conf['dataset_key'],
        logger=logger,
        lr_schedule=lr_schedule,
        logging_level=conf['logging_level'],
        meta_batch_size=conf['meta_batch_size'],
        learner_batch_size=conf['learner_batch_size'],
        n_learner_batches=conf['n_learner_batches'],
        backpropagation_depth=conf['backpropagation_depth'],
        backpropagation_padding=conf['backpropagation_padding'],
        learner_train_size=conf['learner_train_size'],
        learner_test_size=conf['learner_test_size'],
        classes_per_learner_set=conf['classes_per_learner_set'],
        n_train_sets=conf['n_train_sets'],
        n_test_sets=conf['n_test_sets'],
        n_meta_epochs=conf['n_meta_epochs'],
        meta_early_stopping=conf['meta_early_stopping'],
        meta_test_class_ratio=conf['meta_test_class_ratio'],
        initial_learner_lr=conf['initial_learner_lr'],
        hidden_state_size=conf['hidden_state_size'],
        n_meta_valid_steps=conf['n_meta_valid_steps'])
