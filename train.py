import argh
import os

from src.datasets.cifar import load_cifar100
from src.training.train import run_meta_learning
from src.training.training_configuration import read_configuration


def train(clear_logs: bool = False):
    if clear_logs:
        log_dir = os.environ['LOG_DIR']
        for filename in os.listdir(log_dir):
            filepath = os.path.join(log_dir, filename)
            if os.path.isfile(filepath) and (filename.endswith('.txt') or filename.endswith('.log')
                                             or filename.endswith('h5')):
                print("Clearing log file: {}".format(filename))
                os.remove(filepath)

    train_conf_path = os.path.join(os.environ['CONF_DIR'], 'training_configuration.yml')

    training_configuration = read_configuration(train_conf_path)
    training_configuration.log_summary()

    X_train, y_train, X_test, y_test = load_cifar100()

    run_meta_learning(conf=training_configuration, x=X_train, y=y_train)


if __name__ == '__main__':
    argh.dispatch_command(train)
