from typing import List, Tuple, Optional

import numpy as np


class MetaTrainingSample(object):
    """
    Stores data for a single sample for training meta-model using BPTT
    """

    def __init__(
            self,
            inputs: List[np.ndarray],
            initial_states: List[np.ndarray],
            learner_grads: np.ndarray,
            final_output: Optional[np.ndarray] = None,
            initial_learner_weights: Optional[List[np.ndarray]] = None,
            learner_training_batches: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
            learner_validation_batch: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.inputs = inputs
        self.final_output = final_output
        self.initial_states = initial_states
        self.learner_grads = learner_grads
        self.initial_learner_weights = initial_learner_weights
        self.learner_training_batches = learner_training_batches
        self.learner_validation_batch = learner_validation_batch
