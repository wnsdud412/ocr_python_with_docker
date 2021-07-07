from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Bidirectional, GRU, LSTM, GRUCell
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import backend as K
import tensorflow as tf


class SequenceEncoder(Layer):
    """ Squence Encoder Class,
    Text Image Sequence를 2층의 Bidirectional Recurrent Cell을 통해 순서 정보를 처리

    paper
    ----
    On the top of the convolutional layers is a two-layer BLSTM network, each LSTM has 256 hidden units.

    - robust scene text recognition with automatic rectification
    ----

    | Layer Name    | #Hidden Units |
    | ----          | ------ |
    | Bi-RNNcell1   | 256    |
    | Bi-RNNcell2   | 256    |

    """
    def __init__(self,
                 recurrent_cell='lstm',
                 num_depth=2,
                 num_states=256,
                 **kwargs):
        self.recurrent_cell = recurrent_cell
        self.num_depth = num_depth
        self.num_states = num_states
        super().__init__(**kwargs)
        self.blocks = []
        if self.recurrent_cell.lower() == 'gru':
            for i in range(num_depth):
                self.blocks.append(Bidirectional(GRU(num_states, return_sequences=True)))
        elif self.recurrent_cell.lower() == 'lstm':
            for i in range(num_depth):
                self.blocks.append(Bidirectional(LSTM(num_states, return_sequences=True)))
        else:
            raise ValueError('recurrent_cell은 gru와 lstm만 지원합니다.')

    def call(self, inputs, **kwargs):
        x = inputs[0]
        mask = inputs[1]

        for block in self.blocks:
            x = block(x, mask=mask)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_states": self.num_states,
            "num_depth": self.num_depth,
            'recurrent_cell': self.recurrent_cell
        })
        return config


class AttentionDecoder(Layer):
    """ This Class Implements Bahdanau Attention

    paper
    ----
    The generation is a T-step process, at step t, the decoder computes a vector of attention weights alpha
    via the attention process described in [8]:
    alpha_t = Attend(s_t-1, alpha_{t-1}, h)
    where s_{t-1} is the state variable of the GRU cell at the last step . For t=1,
    both s_0 and alpha_0 are zero vectors. then, a glimpse g_t is computed by linearly combining the vectors
    in h: g_t = sum^L_i=1 alpha_{ti}h_i. Since alpht_t has non-negative values that sum to one, it
    effectively controls where the decoder focuses on.
    The state s_{t-1} is updated via the recurrent process of GRU:
    s_t = GRU(l_{t-1},g_t,s_{t-1}),
    where l_{t-1} is the (t-1)-th ground-truth label in training, while in testing, it is the label
    predicted in the previous step, i.e. l_{t-1}.
    The probability distribution over the label space is estimated by:
    y_t = softmax(W^T_s_t).
    Following that, a character l_t is predicted by taking the class with the highest probability.

    - robust scene text recognition with automatic rectification
    ----

    NOTE: Luong Attention을 채택하지 않은 이유

    Luong Attention이 일반적으로 보다 연산량이 적고, 단순한 구조를 가지고 있지만,
    Bahdanau Attention이 Encoder와 Decoder의 Non-Linear 관계를 잘 학습할 수있기 때문에,
    Bahdanau Attention을 이용함
    두 Attention의 차이는 Task에 따라 달라진다고 함

    ref : https://stackoverflow.com/questions/55916133/luong-attention-and-bahdanau-when-should-we-use-luong-or-bahdanau

    """

    def __init__(self, num_states=256, **kwargs):
        self.num_states = num_states
        self.key_dense = Dense(self.num_states, use_bias=False, name='key_dense')
        self.query_dense = Dense(self.num_states, name='query_dense')
        self.score_dense = Dense(1, use_bias=False, name='score_dense')
        self.gru_cell = GRUCell(self.num_states, name='attention_gru')
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        inputs: [encoder_output_sequence, decoder_output_sequences, masking]
        """
        states_encoder = inputs[0]
        states_decoder = inputs[1]
        enc_masks = inputs[2]
        dec_masks = inputs[3]

        enc_masks = (1-tf.cast(enc_masks, tf.float32)) * (2. ** -31)
        enc_masks = tf.expand_dims(enc_masks, axis=-1)

        # >>> (batch size, length of encoder sequence, num hidden)
        key_vectors = self.key_dense(states_encoder)
        value_vectors = states_encoder

        def attention_step(inputs, states):
            curr_states = states[0]
            next_state = self.step(inputs, curr_states, key_vectors, value_vectors, enc_masks)
            return next_state, [next_state]

        batch_size = tf.shape(states_encoder)[0]
        initial_state = tf.zeros((batch_size, self.num_states))

        dec_masks = dec_masks[..., None]
        last_output, outputs, new_states = K.rnn(
            attention_step, states_decoder, [initial_state], mask=dec_masks)
        return outputs

    def step(self, inputs, states, key_vectors, value_vectors, masking):
        """ Step Function for Computing Attention for a one decoder step
        """
        # (1) Calculate Score
        query_vector = self.query_dense(states)
        score = self.score_dense(
            K.tanh(key_vectors + tf.expand_dims(query_vector, axis=1)))
        score = score - masking
        # (2) Normalize Score
        attention = K.softmax(score, axis=1)
        # (3) Calculate Glimpse Vector
        glimpse = K.sum(value_vectors * attention, axis=1)
        # (4) Concatenate Glimpse Vector and Inputs
        context = tf.concat([glimpse, inputs], axis=-1)
        # (5) Calculate Next State Vector
        next_state, _ = self.gru_cell(context, states=[states])
        return next_state

    def get_config(self):
        config = {
            "num_states": self.num_states
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdditionPositionalEncoding(Layer):
    """
    Positional Encoding 정보를 Input에 추가하는 Module Class

    Transformer Model에서 순서 정보를 입력에 주입하는 방식에서 차용함

        PE_{(pos, 2i)} = sin(pos/10000^{2i/d_model})
        PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_model})

    BLSTM 대신 feature sequence에 순서 정보를 추가할 수 있음
    """

    def call(self, inputs, **kwargs):
        b, ts, _ = tf.unstack(tf.shape(inputs)[:3])
        _, _, nh = inputs.get_shape().as_list()[:3]

        ts_range = tf.range(ts)[:, None]
        hs_range = tf.range(nh)[None, :]

        angle_rates = 1. / tf.pow(
            10000., (2. * tf.cast(hs_range // 2, tf.float32)) / tf.cast(nh, tf.float32))
        angle_rads = angle_rates * tf.cast(ts_range, tf.float32)

        # apply sin to even indices in the array; 2i
        mask = tf.tile(tf.constant([1, 0], tf.float32),
                       [nh // 2 + 1, ])[None, :nh]
        pos_encoding = (
                tf.math.sin(angle_rads) * mask
                + tf.math.cos(angle_rads) * (1. - mask))
        pos_encoding = pos_encoding[None, ...]
        pos_encoding = tf.tile(pos_encoding, [b, 1, 1])
        return inputs + tf.cast(pos_encoding, dtype=tf.float32)


__all__ = ["SequenceEncoder",
           "AttentionDecoder",
           "AdditionPositionalEncoding"]