import tensorflow as tf
from korean_ocr.layers.text import CharCompose, PaddingEOS
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras import backend as K


class WordAccuracy(Metric):
    """
    """
    def __init__(self, compose_prediction=True, **kwargs):
        super().__init__(**kwargs)
        self.accs = self.add_weight(name='acc', initializer='zeros')
        self.nums = self.add_weight(name='count', initializer='zeros')
        self.compose_prediction = compose_prediction
        self.char_compose = CharCompose()
        self.padding_eos = PaddingEOS()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.where(tf.not_equal(tf.cast(y_true, dtype=tf.int32), tf.constant(-1, dtype=tf.int32)),
                          tf.cast(y_true, dtype=tf.int32),
                          tf.cast(tf.ones_like(y_true) * ord('\n'), dtype=tf.int32))
        if self.compose_prediction:
            y_pred = self.char_compose(y_pred)
            y_pred = self.padding_eos(y_pred)

        sparse_true = dense_to_sparse(y_true, blank_value=ord('\n'))
        sparse_pred = dense_to_sparse(y_pred, blank_value=ord('\n'))

        word_error_rate = tf.edit_distance(sparse_true, sparse_pred, normalize=True)
        word_accuracy = tf.clip_by_value(1 - word_error_rate, 0., 1.)
        self.accs.assign_add(tf.reduce_sum(word_accuracy))
        self.nums.assign_add(tf.cast(tf.shape(y_pred)[0], tf.float32))

    def result(self):
        return self.accs / (self.nums + K.epsilon())


def dense_to_sparse(tensor, blank_value=0):
    idx = tf.where(tf.not_equal(tf.cast(tensor, tf.int32),
                                tf.constant(blank_value, tf.int32)))
    sparse = tf.SparseTensor(idx,
                             tf.gather_nd(tensor, idx),
                             tf.cast(tf.shape(tensor), tf.int64))
    return sparse


__all__ = ["WordAccuracy"]