import argh
import os
from src.training.training_configuration import read_configuration

from src.datasets.cifar import load_cifar100
from src.training.train import run_meta_learning


def train(clear_logs: bool = False, description: str = ''):
    """
    Train Meta-Learner
    :param clear_logs: clear directory with logs before training
    :param description: description of experiment
    """
    log_dir = os.environ['LOG_DIR']
    if clear_logs:
        for filename in os.listdir(log_dir):
            filepath = os.path.join(log_dir, filename)
            if os.path.isfile(filepath) and (filename.endswith('.txt') or filename.endswith('.log')
                                             or filename.endswith('h5')):
                print("Clearing log file: {}".format(filename))
                os.remove(filepath)

    if description:
        with open(os.path.join(log_dir, 'description.txt'), 'w') as f:
            f.write(description)

    train_conf_path = os.path.join(os.environ['CONF_DIR'], 'training_configuration.yml')

    training_configuration = read_configuration(train_conf_path)
    training_configuration.log_summary()

    X_train, y_train, X_test, y_test = load_cifar100()

    run_meta_learning(conf=training_configuration, x=X_train, y=y_train)


if __name__ == '__main__':
    argh.dispatch_command(train)
