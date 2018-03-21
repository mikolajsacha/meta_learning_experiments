from abc import ABCMeta, abstractmethod

from keras.optimizers import Adam, SGD
import keras.backend as K
import tensorflow as tf


class CustomOptimizer(metaclass=ABCMeta):
    """ We need to optimize meta-learner using raw gradients, not loss """
    @abstractmethod
    def get_updates_by_grads(self, grads, params):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def update_lr(self, lr: float):
        raise NotImplementedError('Abstract method')


class CustomAdam(Adam, CustomOptimizer):
    """
    Adam optimizer - same as in Keras library
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(CustomAdam, self).__init__(lr, beta_1, beta_2, epsilon, decay, amsgrad, **kwargs)
        self.lr_ph = tf.placeholder(shape=self.lr.get_shape(), dtype=tf.float32)
        self.lr_update = K.update(self.lr, self.lr_ph)

    def get_updates_by_grads(self, grads, params):
        # same code as in Adam in Keras
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def update_lr(self, lr: float):
        sess = K.get_session()
        sess.run(self.lr_update, {self.lr_ph: lr})


class CustomSGD(SGD, CustomOptimizer):
    """
    SGD optimizer - same as in Keras library
    """

    def __init__(self, lr=0.001):
        super(CustomSGD, self).__init__(lr=lr)
        self.lr_ph = tf.placeholder(shape=self.lr.get_shape(), dtype=tf.float32)
        self.lr_update = K.update(self.lr, self.lr_ph)

    def get_updates_by_grads(self, grads, params):
        updates = []

        for g, p in zip(grads, params):
            new_p = p - self.lr * g
            updates.append(K.update(p, new_p))

        return updates

    def update_lr(self, lr: float):
        sess = K.get_session()
        sess.run(self.lr_update, {self.lr_ph: lr})
