import argh
import logging
import os

from src.datasets.cifar import load_cifar100
from src.training.train import run_meta_learning
from src.training.training_configuration import TrainingConfiguration
from src.utils.logging import configure_logger


def train(clear_logs: bool = False):
    if clear_logs:
        log_dir = os.environ['LOG_DIR']
        for filename in os.listdir(log_dir):
            filepath = os.path.join(log_dir, filename)
            if os.path.isfile(filepath) and (filename.endswith('.txt') or filename.endswith('.log')):
                print("Clearing log file: {}".format(filename))
                os.remove(filepath)

    log_level = logging.INFO
    logger = configure_logger(name='train-meta-model-test', level=log_level)

    lr_schedule = [
        (0, 0.001)
    ]

    training_configuration = TrainingConfiguration(
        continue_task=True,
        debug_mode=False,
        hidden_state_size=10,
        n_meta_epochs=256,
        meta_early_stopping=256,
        lr_schedule=lr_schedule,
        classes_per_learner_set=2,
        meta_batch_size=8,
        learner_batch_size=32,
        n_learner_batches=32,
        backpropagation_depth=16,
        backpropagation_padding=6,
        learner_train_size=32,
        learner_test_size=96,
        n_train_sets=64,
        n_test_sets=64,
        meta_test_class_ratio=0.3,
        initial_learner_lr=0.05,
        n_meta_valid_steps=64,
        dataset_key="cifar100",
        logger=logger,
        logging_level=log_level)

    training_configuration.log_summary()

    X_train, y_train, X_test, y_test = load_cifar100()

    run_meta_learning(conf=training_configuration, x=X_train, y=y_train)


if __name__ == '__main__':
    argh.dispatch_command(train)
