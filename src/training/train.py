import os
import numpy as np
from keras import Model
from keras.optimizers import SGD

from src.isotropy.lanczos import TopKEigenvaluesBatched
from src.model.meta_learner.lstm_model import lstm_meta_learner
from src.training.training_configuration import TrainingConfiguration
from src.datasets.metadataset import MetaLearningDatasetFactory, load_meta_dataset

from src.datasets.cifar import cifar_input_shape
from src.model.learner.simple_cnn import build_simple_cnn
from src.model.util import get_trainable_params_count
from src.training.lr_schedule import get_lr_scheduler
from src.training.meta_learning_task import MetaLearningTask


def run_meta_learning(conf: TrainingConfiguration, x: np.ndarray, y: np.ndarray):
    meta_dataset_path = conf.meta_dataset_path
    logger = conf.logger

    if not os.path.isfile(meta_dataset_path):
        factory = MetaLearningDatasetFactory(
            x=x,
            y=y,
            meta_test_ratio=conf.meta_test_class_ratio,
            learner_train_size=conf.learner_train_size,
            learner_test_size=conf.learner_test_size,
            classes_per_learner_set=conf.classes_per_learner_set,
            n_train_sets=conf.n_train_sets,
            n_test_sets=conf.n_test_sets,
            logger=logger)

        meta_dataset = factory.get()
        logger.info("Saving generated dataset to {}".format(meta_dataset_path))
        meta_dataset.save(meta_dataset_path)
    else:
        logger.info("Loading previously generated dataset from {}".format(meta_dataset_path))
        meta_dataset = load_meta_dataset(meta_dataset_path, x)

    def learner_factory():
        return build_simple_cnn(cifar_input_shape, conf.classes_per_learner_set)

    def meta_learner_factory(learner: Model, eigenvals_callback: TopKEigenvaluesBatched):
        return lstm_meta_learner(learner=learner, eigenvals_callback=eigenvals_callback, configuration=conf)

    # build dummy learner/meta-learner just to display summary
    dummy_learner = learner_factory()
    n_params = get_trainable_params_count(dummy_learner)

    dummy_learner.compile(loss='categorical_crossentropy',
                          optimizer=SGD(lr=0.0),
                          metrics=['accuracy'])

    logger.info("Using Learner with {} parameters".format(n_params))
    dummy_learner.summary()

    log_dir = os.environ['LOG_DIR']

    meta_learning_task = MetaLearningTask(
        configuration=conf,
        task_checkpoint_path=os.path.join(log_dir, 'checkpoint.txt'),
        meta_dataset=meta_dataset,
        learner_factory=learner_factory,
        meta_learner_factory=meta_learner_factory,
        training_history_path=os.path.join(log_dir, "meta_training_history.txt"),
        meta_learner_weights_path=os.path.join(log_dir, "meta_weights.h5"),
        meta_learner_weights_history_dir=os.path.join(log_dir, "meta_weights_history"),
        best_meta_learner_weights_path=os.path.join(log_dir, "meta_weights_best.h5"))

    meta_learning_task.meta_train(
        n_meta_epochs=conf.n_meta_epochs,
        meta_early_stopping=conf.meta_early_stopping,
        n_learner_batches=conf.n_learner_batches,
        meta_batch_size=conf.meta_batch_size,
        n_meta_train_steps=conf.n_train_sets // conf.meta_batch_size,
        n_meta_valid_steps=conf.n_meta_valid_steps,
        learner_batch_size=conf.learner_batch_size)
