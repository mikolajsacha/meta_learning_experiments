"""
Keras callback using Lanczos iteration to compute top K eigenvalues.
"""

import os
import gc
import _pickle as pickle
import keras.backend as K
from keras.callbacks import Callback
import collections
import numpy as np
from scipy.linalg import svd

from tensorflow.contrib.solvers.python.ops.lanczos import lanczos_bidiag

import tensorflow as tf

try:
    # noinspection PyProtectedMember
    from tensorflow.python.ops.gradients import _hessian_vector_product
except ImportError:
    # noinspection PyProtectedMember
    from tensorflow.python.ops.gradients_impl import _hessian_vector_product


def _gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q


# noinspection PyProtectedMember,PyMethodOverriding,SpellCheckingInspection,PyPep8Naming,PyBroadException
class TopKEigenvaluesBatched(Callback):
    """
    Computes top K eigenvalues using Lanczos algorithm and supports batching internally

    See e.g. http://people.bath.ac.uk/mamamf/talks/lanczos.pdf
    """

    def __init__(self, K, batch_size, logger, save_dir, save_eigenv=False, impl="tf", inference_mode=True,
                 feature_K=None):
        super().__init__()
        self.K = K
        self.impl = impl
        self.inference_mode = inference_mode
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.save_eigenv = save_eigenv
        self.X = None
        self.y = None
        self._lanczos_tensor = None
        self._my_model = None
        self.logger = logger
        self._this_epoch_ev = None
        self._epoch_begin_logs = None
        self.parameters = None
        self.parameter_names = None
        self.parameter_shapes = None
        self.spectral_norm_tensor = None
        self.spectral_norm_tensor_2 = None
        self.eigenvector_tensor = None
        self.feature_K = feature_K if feature_K is not None else K

    def _construct_laszlo_operator_batched(self, dtype=np.float32):
        L = self.get_my_model().total_loss
        ws = self.get_my_model().trainable_weights
        X, y, sample_weights = self.get_my_model()._feed_inputs[0], \
                               self.get_my_model()._feed_targets[0], self.get_my_model()._feed_sample_weights[0]
        bs = self.batch_size

        shapes = [K.int_shape(w) for w in ws]
        dim = np.sum([np.prod(s) for s in shapes])
        shape = (dim, dim)
        linear_operator = collections.namedtuple(
            "LinearOperator", ["shape", "dtype", "apply", "apply_adjoint"])

        v_vect = tf.placeholder(tf.float32, [dim, 1])
        v_reshaped = []
        cur = 0
        for s in shapes:
            v_reshaped.append(K.reshape(v_vect[cur:np.prod(s) + cur], s))
            cur += np.prod(s)
        Hv_vect = _hessian_vector_product(L, ws, v_reshaped)

        sess = K.get_session()

        # noinspection SpellCheckingInspection
        def apply_cpu(v):
            res = [0 for _ in ws]
            for id1 in range(self.X.shape[0] // bs):
                x_batch = self.X[id1 * bs:(id1 + 1) * bs].astype(dtype)
                y_batch = self.y[id1 * bs:(id1 + 1) * bs].astype(dtype)
                ress = sess.run(Hv_vect, feed_dict={v_vect: v,
                                                    X: x_batch,
                                                    y: y_batch,
                                                    sample_weights: np.ones(shape=(bs,))
                                                    })
                for id2 in range(len(ws)):
                    res[id2] += bs * ress[id2].reshape(-1, 1)

            return np.concatenate(res, axis=0) / self.X.shape[0]

        def apply(v):
            return tf.py_func(apply_cpu, [v], tf.float32)

        return linear_operator(
            apply=apply,
            apply_adjoint=apply,
            dtype=dtype,
            shape=shape)

    def compile(self):
        if self.logger is not None:
            self.logger.debug("Compiling TopKEignevaluesBattched Callback")
        _op = self._construct_laszlo_operator_batched()
        self._lanczos_tensor = {self.K: lanczos_bidiag(_op, self.K)}
        if self.K != self.feature_K:
            self._lanczos_tensor[self.feature_K] = lanczos_bidiag(_op, self.feature_K)

        self.parameters = self.get_my_model().trainable_weights
        self.parameter_names = [p.name for p in self.get_my_model().trainable_weights]
        self.parameter_shapes = [K.int_shape(p) for p in self.get_my_model().trainable_weights]

        def get_eigenvalue_feature():
            biggest_eigenval = self.compute_top(k=self.feature_K, with_vectors=False)
            return np.float32(biggest_eigenval)

        def get_eigenvector_features():
            biggest_eigenval, biggest_eigenvec = self.compute_top(k=self.feature_K, with_vectors=True)
            return [np.float32(biggest_eigenval), np.float32(biggest_eigenvec[:, :self.feature_K])]

        self.spectral_norm_tensor = tf.py_func(get_eigenvalue_feature, [], tf.float32)

        eigenvector_tensors = tf.py_func(get_eigenvector_features, [], [tf.float32, tf.float32])
        self.spectral_norm_tensor_2 = eigenvector_tensors[0]
        self.eigenvector_tensor = eigenvector_tensors[1]

    def set_my_model(self, model):
        model.summary()
        self._my_model = model

    def get_my_model(self):
        # Defaults to model set by keras, but can be overwritten
        if self._my_model is not None:
            return self._my_model
        else:
            return self.model

    def compute_top(self, k, with_vectors=True):
        if self.X is None or self.y is None:
            raise ValueError("Set self.X and self.y first")
        if self._lanczos_tensor is None:
            self.compile()
        if k not in self._lanczos_tensor:
            raise ValueError("No Lanczos Tensor created for K={}".format(k))

        if self.logger is not None:
            self.logger.debug("Computing bi-diagonalization using Laszlo algorithm")
        lanczos_tensor = self._lanczos_tensor[k]
        res_laszlo = K.get_session().run(lanczos_tensor)
        # According to https://en.wikipedia.org/wiki/Lanczos_algorithm
        # performing SVD on bi-diagonalized matrix returned by Lanczos
        A = np.zeros(shape=(k + 1, k))
        for i in range(k):
            A[i][i] = res_laszlo.alpha[i]
            A[i + 1][i] = res_laszlo.beta[i]

        try:
            Z = svd(A)
            V = res_laszlo.v
            # This "magic" formula is taken from wiki on Lanczos. Tested that it finds indeed eigenvectors
            eigenvalues = Z[1]
            if with_vectors:
                eigenvectors = V.dot(Z[2].T)
        except Exception:
            if self.logger is not None:
                self.logger.warning("Failed svd. Probably due to NaNs in A")
            eigenvalues = np.zeros(shape=(k,))
            if with_vectors:
                eigenvectors = np.zeros_like(res_laszlo.v)

        if self.logger is not None:
            self.logger.debug(eigenvalues[0:10])

        if with_vectors:
            # noinspection PyUnboundLocalVariable
            return eigenvalues, eigenvectors
        return eigenvalues

    def compute_top_K(self, with_vectors=True):
        return self.compute_top(self.K, with_vectors)

    def save(self, path, E=None, Ev=None, with_vectors=True):
        if E is None or Ev is None:
            if with_vectors:
                E, Ev = self.compute_top_K(with_vectors=True)
            else:
                E = self.compute_top_K(with_vectors=False)
        if self.logger is not None:
            self.logger.debug("Saving eigenvectors")
        if with_vectors:
            np.savez(os.path.join(self.save_dir, path), E=E, Ev=Ev)
        else:
            np.savez(os.path.join(self.save_dir, path), E=E)

    def save_param_shapes(self, path):
        pickle.dump({"shapes": self.parameter_shapes, "names": self.parameter_names}, open(path, 'wb'))

    def on_epoch_begin(self, epoch, logs):
        E, Ev = self.compute_top_K()
        logs['top_K_e'] = E
        if self.save_eigenv > 0:
            if epoch % self.save_eigenv == 0:
                self.save(os.path.join(self.save_dir, "top_K_ev_{}.npz".format(epoch)), E, Ev)
                self.save_param_shapes(os.path.join(self.save_dir, "top_K_ev_mapping.pkl"))
        self._this_epoch_ev = Ev
        self._epoch_begin_logs = logs
        if self.logger is not None:
            self.logger.debug(logs['top_K_e'])

        gc.collect()  # Let go of previous Ev

    def on_epoch_end(self, epoch, logs):
        # HACK!!! Seems like it is necessary, weirdly
        for k in self._epoch_begin_logs:
            logs[k] = self._epoch_begin_logs[k]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['X']
        del state['y']
        del state['parameters']
        del state['_lanczos_tensor']
        del state['_this_epoch_ev']
        del state['_my_model']
        return state
