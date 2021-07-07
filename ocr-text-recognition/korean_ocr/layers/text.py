from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras import backend as K
from korean_ocr.utils.jamo import 초성, 중성, 종성
import tensorflow as tf


"""
한글의 경우, 기본적으로 자모자 조합형 언어로 구성되어 있습니다.
이 때문에, 완성형 언어와 같은 방식으로 가갸거갸고교구규... 등으로 처리한다면, 11172의 클래스로 분류해야 하지만
한글 자모자로 구성할 경우, 초성(19자)+중성(21자)+종성(28자)로 나누면 됩니다.
"""
DEFAULT_SPECIAL_CHARACTERS = (
    ' ', '"', "'", '(', ')', ',', '.', '?', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


class CharEmbedding(Layer):
    """
    1. 한글의 경우 : Unicode 번호를 Jamo 별로 Decompose 후 각각 Embedding 하는 Module Class
    2. 한글이 아닌 경우 : text 순서에 따라 Embedding 하는 Module Class
    """

    def __init__(self,
                 special_characters=DEFAULT_SPECIAL_CHARACTERS,
                 num_embed=16,
                 mask_value=-1,
                 **kwargs):
        self.num_embed = num_embed
        self.special_characters = special_characters
        self.mask_value = mask_value
        # input dim : 자모자 갯수 + ["blank"]
        self.초성_layer = Embedding(input_dim=len(초성) + 1, output_dim=num_embed)
        self.중성_layer = Embedding(input_dim=len(중성) + 1, output_dim=num_embed)
        self.종성_layer = Embedding(input_dim=len(종성) + 1, output_dim=num_embed)
        # input dim : ['\n'] + special case + ["blank"]
        num_special = len(special_characters) + 2
        self.특수_layer = Embedding(input_dim=num_special, output_dim=num_embed)
        self.char_decompose = CharDeCompose()
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # (1) decompose : 자모자로 분리하기
        inputs = tf.cast(inputs, dtype=tf.int32)
        한글_arr, 초성_arr, 중성_arr, 종성_arr, 특수_arr = self.char_decompose(inputs)

        # (2) embed : Embedding Layer 통과하기
        초성_embed = self.초성_layer(초성_arr)
        중성_embed = self.중성_layer(중성_arr)
        종성_embed = self.종성_layer(종성_arr)
        특수_embed = self.특수_layer(특수_arr)

        # (3) concat : 하나의 embedding Vector로 쌓기
        embed = K.concatenate([초성_embed, 중성_embed, 종성_embed, 특수_embed], axis=-1)

        dec_mask = tf.not_equal(inputs, self.mask_value)
        return embed, dec_mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "special_characters": self.special_characters,
            "num_embed": self.num_embed,
            "mask_value": self.mask_value
        })
        return config


class CharDeCompose(Layer):
    """ 자모자 Unicode를 Decompose하여 초성/중성/종성별로 나누어주는 함수 Module Layer
    """

    def __init__(self,
                 special_characters=DEFAULT_SPECIAL_CHARACTERS,
                 **kwargs):
        self.special_characters = special_characters
        self.table = build_special_lookup_table(self.special_characters, reverse=False)
        kwargs.setdefault('trainable', False)
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        한글_mask = (inputs >= ord('가')) & (inputs <= ord('힣'))
        특수_mask = (~한글_mask) & (tf.not_equal(inputs, -1)) # Blank
        한글_arr = tf.cast(한글_mask, tf.int32)

        초성_code = ((inputs - ord('가')) // len(종성)) // len(중성)
        초성_code = tf.where(한글_mask, 초성_code, tf.ones_like(초성_code) * len(초성))

        중성_code = ((inputs - ord('가')) // len(종성)) % len(중성)
        중성_code = tf.where(한글_mask, 중성_code, tf.ones_like(중성_code) * len(중성))

        종성_code = (inputs - ord('가')) % len(종성)
        종성_code = tf.where(한글_mask, 종성_code, tf.ones_like(종성_code) * len(종성))

        특수_code = self.table.lookup(inputs)
        특수_code = tf.where(특수_mask, 특수_code,
                            tf.ones_like(특수_code) * len(self.special_characters))

        return 한글_arr, 초성_code, 중성_code, 종성_code, 특수_code

    def get_config(self):
        config = super().get_config()
        config.update({
            'special_characters': self.special_characters
        })
        return config


class CharCompose(Layer):
    """ 자모자 Compose하여 Unicode 숫자로 바꾸어주는 Module Layer
    """
    def __init__(self,
                 special_characters=DEFAULT_SPECIAL_CHARACTERS,
                 **kwargs):
        self.special_characters = special_characters
        self.table = build_special_lookup_table(self.special_characters, reverse=True)
        kwargs.setdefault('trainable', False)
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        한글_pred, 초성_pred, 중성_pred, 종성_pred, 특수_pred = tf.split(
            inputs, [1, len(초성)+1, len(중성)+1, len(종성)+1, len(self.special_characters)+2],
            axis=-1)

        초성_code = tf.cast(tf.argmax(초성_pred, axis=-1), dtype=K.floatx())
        중성_code = tf.cast(tf.argmax(중성_pred, axis=-1), dtype=K.floatx())
        종성_code = tf.cast(tf.argmax(종성_pred, axis=-1), dtype=K.floatx())
        한글_unicode = tf.cast(
            (초성_code * len(중성) + 중성_code) * len(종성) +
            종성_code + ord('가'), dtype=tf.int32)

        특수_code = tf.cast(tf.argmax(특수_pred, axis=-1), dtype=tf.int32)
        특수_unicode = self.table.lookup(특수_code)

        한글_mask = tf.squeeze(한글_pred, axis=-1) >= 0.5
        unicode_arr = tf.where(한글_mask, 한글_unicode, 특수_unicode)

        return unicode_arr

    def get_config(self):
        config = super().get_config()
        config.update({
            'special_characters': self.special_characters
        })
        return config


class PaddingEOS(Layer):
    """ 예측 결과값을 <EOS> 토큰으로 Padding처리하는 Module Layer
    """
    def call(self, inputs, **kwargs):
        # 첫번째 등장한 EOS 뒤 모든 값들을 EOS로 변경해줌
        def fn(tensor):
            idx = tf.where(tf.equal(tensor, ord('\n')))
            # <EOS>가 없는 경우
            idx = tf.cond(tf.size(idx) > 0,
                          lambda: idx[0, 0],
                          lambda: tf.cast(tf.shape(tensor)[0], tf.int64))
            padded = tf.concat([tensor[:idx],
                                tf.ones_like(tensor[idx:]) * ord('\n')],
                               axis=0)
            return padded
        unicode_arr = tf.map_fn(fn, inputs)

        # 불필요한 EOS 패딩을 제거
        indices = tf.where(
            tf.reduce_all(tf.equal(unicode_arr, ord('\n')), axis=0))
        ind = tf.cond(tf.size(indices) > 0,
                      lambda: tf.cast(indices[0, 0], tf.int64),
                      lambda: tf.cast(tf.shape(unicode_arr)[1], tf.int64))
        unicode_arr = unicode_arr[:, :ind]

        return unicode_arr


class CharClassifier(Layer):
    """ 자모자 별로 분류하는 Classifier
    자모 별로 Dense Layer *2 & Softmax를 둚
    """
    def __init__(self,
                 special_characters=DEFAULT_SPECIAL_CHARACTERS,
                 num_fc=128,
                 **kwargs):
        super().__init__(**kwargs)
        self.special_characters = special_characters
        self.num_fc = num_fc

        self.한글_dense = Dense(num_fc, activation='relu')
        self.한글_classify = Dense(1, activation='sigmoid')

        self.초성_dense = Dense(num_fc, activation='relu')
        self.초성_classify = Dense(len(초성) + 1, activation='softmax')

        self.중성_dense = Dense(num_fc, activation='relu')
        self.중성_classify = Dense(len(중성) + 1, activation='softmax')

        self.종성_dense = Dense(num_fc, activation='relu')
        self.종성_classify = Dense(len(종성) + 1, activation='softmax')

        self.특수_dense = Dense(num_fc, activation='relu')
        self.특수_classify = Dense(len(self.special_characters) + 2, activation='softmax')

        self.concat = Concatenate(axis=-1)

    def call(self, inputs, **kwargs):
        한글_prediction = self.한글_classify(self.한글_dense(inputs))
        초성_prediction = self.초성_classify(self.초성_dense(inputs))
        중성_prediction = self.중성_classify(self.중성_dense(inputs))
        종성_prediction = self.종성_classify(self.종성_dense(inputs))
        특수_prediction = self.특수_classify(self.특수_dense(inputs))

        return self.concat([
            한글_prediction, 초성_prediction, 중성_prediction, 종성_prediction, 특수_prediction])

    def get_config(self):
        config = super().get_config()
        config.update({
            "special_characters": self.special_characters,
            "num_fc": self.num_fc
        })
        return config


def build_special_lookup_table(special_characters, reverse=False):
    cases = ['\n'] + list(special_characters)
    if reverse:
        kv_init = tf.lookup.KeyValueTensorInitializer(
            keys=list(range(len(cases))),
            values=[ord(case) for case in cases])
    else:
        kv_init = tf.lookup.KeyValueTensorInitializer(
            keys=[ord(case) for case in cases],
            values=list(range(len(cases))))
    table = tf.lookup.StaticHashTable(kv_init, -1)
    if int(tf.__version__.split('.')[0]) == 1:
        K.get_session().run(table.initializer)
    return table


__all__ = ["CharDeCompose",
           "CharCompose",
           "CharEmbedding",
           "CharClassifier",
           "PaddingEOS",
           "build_special_lookup_table"]
