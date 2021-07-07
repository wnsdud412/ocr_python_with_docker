import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import Loss
from korean_ocr.utils.jamo import 초성, 중성, 종성
from korean_ocr.layers.text import CharDeCompose, DEFAULT_SPECIAL_CHARACTERS


class CharCategoricalCrossEntropy(Loss):
    """
    Calculate Negative Log-Likelihood per JAMO(초성, 중성, 종성)

    :param y_true:
        tensor (samples, time_steps) containing the truth labels.
    :param y_pred:
        tensor (samples, time_steps, num_categories) containing the output of the softmax.

    * caution

    """
    def __init__(self,
                 special_characters=DEFAULT_SPECIAL_CHARACTERS,
                 **kwargs):
        self.special_characters = special_characters
        self.char_decompose = CharDeCompose()
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_true = K.cast(y_true, tf.int32)
        y_pred = y_pred[:, :tf.shape(y_true)[1]] # Slicing prediction

        ignore_mask = tf.cast(
            tf.not_equal(y_true, tf.constant(-1, dtype=tf.int32)), dtype=K.floatx())

        y_true_한글, y_true_초성, y_true_중성, y_true_종성, y_true_특수 = self.char_decompose(y_true)
        y_pred_한글, y_pred_초성, y_pred_중성, y_pred_종성, y_pred_특수 = tf.split(
            y_pred, [1, len(초성) + 1, len(중성) + 1, len(종성) + 1, -1], axis=-1)

        loss_한글 = K.binary_crossentropy(
            tf.cast(y_true_한글, tf.float32), tf.squeeze(y_pred_한글, axis=-1)) * ignore_mask

        loss_초성 = K.sparse_categorical_crossentropy(y_true_초성, y_pred_초성) * ignore_mask
        loss_중성 = K.sparse_categorical_crossentropy(y_true_중성, y_pred_중성) * ignore_mask
        loss_종성 = K.sparse_categorical_crossentropy(y_true_종성, y_pred_종성) * ignore_mask
        loss_자모 = tf.cast(y_true_한글, tf.float32) * (loss_초성 + loss_중성 + loss_종성) / 3.

        loss_특수 = K.sparse_categorical_crossentropy(y_true_특수, y_pred_특수) * ignore_mask
        loss_특수 = (1. - tf.cast(y_true_한글, tf.float32)) * loss_특수

        losses = K.sum(loss_한글+loss_자모+loss_특수, axis=1)
        return losses


__all__ = ["CharCategoricalCrossEntropy"]
