import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from korean_ocr.layers.text import build_special_lookup_table, CharCompose
from korean_ocr.utils.jamo import 초성, 중성, 종성


class AttentionInference(Layer):
    """ Bahdanau Attention에서의 추론

    """
    def __init__(self,
                 char_embedding_layer,
                 attention_layer,
                 char_classifier,
                 max_length=30,
                 **kwargs):
        self.max_length = max_length
        self.char_embedding = char_embedding_layer

        (k, q, s, g) = attention_layer.submodules
        self.key_dense = k
        self.query_dense = q
        self.score_dense = s
        self.gru_cell = g
        self.char_classifier = char_classifier

        self.special_characters = self.char_embedding.get_config()['special_characters']
        self.num_states = attention_layer.get_config()['num_states']
        self.char_compose = CharCompose(self.special_characters)
        self.built = True
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        inputs: [encoder_output_sequence, masking]
        """
        states_encoder = inputs[0]
        enc_masks = inputs[1]

        enc_masks = (1. - tf.cast(enc_masks, tf.float32)) * (2. ** -31)
        enc_masks = tf.expand_dims(enc_masks, axis=-1)

        # >>> (batch size, length of encoder sequence, num hidden)
        key_vectors = self.key_dense(states_encoder)
        value_vectors = states_encoder

        batch_size = tf.shape(value_vectors)[0]
        curr_state = tf.zeros((batch_size, self.num_states))
        curr_unicode = tf.ones((batch_size, ), dtype=tf.int32) * ord('\n')

        outputs = []
        for i in range(self.max_length):
            curr_embed, _ = self.char_embedding(curr_unicode)
            curr_state = self.attention_step(
                curr_embed, curr_state, key_vectors, value_vectors, enc_masks)
            preds = self.char_classifier(curr_state)
            curr_unicode = self.char_compose(preds)
            outputs.append(curr_unicode)
        outputs = tf.stack(outputs, axis=-1)
        return outputs

    def attention_step(self, inputs, states, key_vectors, value_vectors, masking):
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
        # (5) Calculate Hidden Vector
        next_state, _ = self.gru_cell(context, states=[states])
        return next_state


class BeamSearchInference(Layer):
    """ Bahdanau Attention에서의 추론을 Beam Search 알고리즘 방식으로 구현한 것

    """

    def __init__(self,
                 char_embedding_layer,
                 attention_layer,
                 char_classifier,
                 beam_width=3,
                 max_length=50,
                 alpha=0.7,
                 **kwargs):
        self.beam_width = beam_width
        self.max_length = max_length
        self.alpha = alpha

        self.char_embedding = char_embedding_layer
        key_dense, query_dense, score_dense, gru_cell = attention_layer.submodules
        self.key_dense = key_dense
        self.query_dense = query_dense
        self.score_dense = score_dense
        self.gru_cell = gru_cell
        self.char_classifier = char_classifier

        self.special_characters = self.char_embedding.get_config()['special_characters']
        self.num_states = attention_layer.get_config()['num_states']
        self.table = build_special_lookup_table(self.special_characters, reverse=True)
        self.built = True
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        inputs: [encoder_output_sequence, masking]
        """
        states_encoder = inputs[0]
        masking = inputs[1]

        masking = (1. - tf.cast(masking, tf.float32)) * (2. ** -31)
        masking = tf.expand_dims(masking, axis=-1)

        # >>> (batch size, length of encoder sequence, num hidden)
        key_vectors = self.key_dense(states_encoder)
        val_vectors = states_encoder

        # >>> (batch size, beam width, length of encoder sequence, num hidden)
        beam_masking = add_beam_axis(masking, self.beam_width)
        beam_key_vectors = add_beam_axis(key_vectors, self.beam_width)
        beam_val_vectors = add_beam_axis(val_vectors, self.beam_width)

        def beam_search_step(_, states):
            """ Step Function for Computing Attention for a one decoder step
            """
            # GRU Cell 상태 벡터 (batch size, beam width, num state)
            curr_state = states[0]
            # 이전 입력 Unicode 값 (batch size, beam width)
            curr_unicode = states[1]
            # 해당 beam에 대한 log prob 합 (batch size, beam width)
            curr_log_probs = states[2]
            # 해당 beam의 이전 출력 길이 (batch size, beam width)
            curr_length = states[3]
            # 해당 빔이 출력을 끝냈는지 유무 (batch size, beam width)
            curr_finished = states[4]

            # Output Masking 용
            not_finish_mask = tf.logical_not(curr_finished)

            # 이전 입력 unicode의 Embedding Vector (batch size, beam width, num embed)
            curr_embed, _ = self.char_embedding(curr_unicode)
            # 어텐션 Vector (batch size, beam width, num state)
            beam_next_states = self.beam_attention_step(
                curr_embed, curr_state, beam_key_vectors, beam_val_vectors, beam_masking)

            # Classifier 출력 (batch size, beam width, 92)
            beam_probs = self.char_classifier(beam_next_states)
            # 첫번째 STEP 경우, Masking
            beam_probs = mask_first_step_beam_probs(curr_length, beam_probs)
            # finish인 경우, Masking
            beam_probs = beam_probs * tf.cast(not_finish_mask[..., None], tf.float32)
            # Log Scale 변경하고, 가~힣 + Special cases에 대한 softmax 값 체계로 변경
            # (batch size, beam width, num chars)
            # num chars -> num 한글 + num special cases + 1([\n])
            log_beam_probs = convert_to_log_prob(beam_probs, self.beam_width)
            prev_beam_id, next_unicode, log_probs = self.decode_log_beam_probs(log_beam_probs)
            # beam_indices : 이전 unicode의 beam id (linked list 구조) (batch size, beam width)
            # next_unicode : Beam 내 unicode 값 (batch size, beam width)
            # log_probs : 해당 unicode에 대한 log probability (batch size, beam width)

            next_unicode = (
                tf.where(not_finish_mask, next_unicode, tf.ones_like(next_unicode) * -1))

            next_state = (
                self.gather_states_from_prev_beam_id(beam_next_states, prev_beam_id))
            not_finish_state_mask = tf.tile(not_finish_mask[..., None], [1, 1, self.num_states])
            next_state = tf.where(not_finish_state_mask, next_state, curr_state)

            next_log_probs = (
                self.gather_states_from_prev_beam_id(curr_log_probs, prev_beam_id) + log_probs)
            next_log_probs = tf.where(not_finish_mask, next_log_probs, curr_log_probs)

            next_length = (
                  self.gather_states_from_prev_beam_id(curr_length, prev_beam_id)
                  + tf.ones_like(curr_length))
            next_length = tf.where(not_finish_mask, next_length, curr_length)

            next_finished = (
                self.gather_states_from_prev_beam_id(curr_finished, prev_beam_id) |
                tf.equal(next_unicode, ord('\n')) |
                tf.equal(next_unicode, -1))

            output = tf.stack([prev_beam_id, next_unicode], axis=-1)
            states = [next_state, next_unicode, next_log_probs, next_length, next_finished]
            return output, states

        batch_size = tf.shape(states_encoder)[0]
        dummy_decoder = tf.zeros([batch_size, self.max_length, 1])
        initial_unicode = (
                tf.ones((batch_size, self.beam_width), dtype=tf.int32) * ord('\n'))
        initial_state = (
            tf.zeros((batch_size, self.beam_width, self.num_states)))
        initial_log_probs = (
            tf.zeros((batch_size, self.beam_width), dtype=tf.float32))
        initial_length = (
            tf.zeros((batch_size, self.beam_width), dtype=tf.int32))
        initial_finished = (
            tf.zeros((batch_size, self.beam_width), dtype=tf.bool))

        _, outputs, stats = K.rnn(
            beam_search_step, dummy_decoder,
            [initial_state, initial_unicode, initial_log_probs,
             initial_length, initial_finished])

        log_probs = stats[2]
        length = stats[3]
        length_penalty = self.calculate_length_penalty(length)
        beam_scores = log_probs / length_penalty

        unicode_seqs = self.reorder_unicode_sequences(outputs, beam_scores)
        beam_scores = tf.sort(beam_scores, axis=-1, direction='DESCENDING')
        return unicode_seqs, beam_scores

    def beam_attention_step(self,
                            beam_inputs,
                            beam_states,
                            beam_key_vectors,
                            beam_val_vectors,
                            beam_masking):
        # (1) calculate score
        beam_query_vector = self.query_dense(beam_states)
        beam_score = self.score_dense(
            K.tanh(beam_key_vectors +
                   tf.expand_dims(beam_query_vector, axis=-2)))
        beam_score = beam_score - beam_masking
        # (2) normalize score
        beam_attention = K.softmax(beam_score, axis=-2)
        # (3) calculate glimpse vector
        beam_glimpse = K.sum(beam_val_vectors * beam_attention, axis=-2)
        # (4) concatenate glimpse vector and inputs
        beam_context = tf.concat([beam_glimpse, beam_inputs], axis=-1)
        # (5) Calculate next states vector
        beam_next_states, _ = self.gru_cell(beam_context, states=[beam_states])
        return beam_next_states

    def decode_log_beam_probs(self, log_beam_probs):
        num_chars = log_beam_probs.get_shape().as_list()[-1]
        flatten_beam_probs = K.reshape(
            log_beam_probs, (-1, self.beam_width * num_chars))
        log_probs, indices = tf.math.top_k(flatten_beam_probs, k=self.beam_width)

        beam_indices = indices // num_chars
        unicode = indices % num_chars
        num_한글 = len(초성) * len(중성) * len(종성)
        unicode = tf.where(unicode >= num_한글,
                           self.table.lookup(unicode - num_한글),
                           unicode + ord('가'))
        return beam_indices, unicode, log_probs

    def gather_states_from_prev_beam_id(self, states, prev_beam_id):
        batch_size = tf.shape(states)[0]
        batch_id = tf.reshape(
            tf.tile(
                tf.range(0, batch_size * self.beam_width, self.beam_width, dtype=tf.int32)[:, None],
                [1, self.beam_width]), (-1,))
        batch_beam_id = batch_id + tf.reshape(prev_beam_id, (-1,))
        chosen_states = tf.gather(
            tf.reshape(states,
                       (batch_size * self.beam_width, -1)), batch_beam_id)
        reshaped_states = tf.reshape(chosen_states, (batch_size, self.beam_width, -1))
        if K.ndim(states) > 2:
            return reshaped_states
        else:
            return tf.squeeze(reshaped_states, axis=-1)

    def reorder_unicode_sequences(self, outputs, beam_scores):
        beam_id_seqs, unicode_seqs = tf.unstack(outputs, axis=-1)
        beam_id_seqs = tf.roll(beam_id_seqs, -1, axis=1)

        out_seqs = tf.concat([beam_id_seqs, unicode_seqs], axis=-1)
        reshaped_out_seqs = tf.reshape(out_seqs, (-1, self.beam_width * 2))

        reshaped_unicode_seqs = tf.map_fn(
            lambda x: tf.gather(x[self.beam_width:], x[:self.beam_width]), reshaped_out_seqs)

        batch_size = tf.shape(beam_id_seqs)[0]
        reordered_unicode_seqs = tf.reshape(reshaped_unicode_seqs, (batch_size, -1, self.beam_width))

        # return shape : (batch size, beam width, time steps)
        unicode_seqs = tf.transpose(reordered_unicode_seqs, (0, 2, 1))

        def sort_by_beam_score(batch_id):
            return tf.gather(unicode_seqs[batch_id],
                             tf.argsort(beam_scores[batch_id], direction='DESCENDING'))

        return tf.map_fn(sort_by_beam_score, tf.range(0, batch_size, delta=1))

    def calculate_length_penalty(self, tensor):
        return (tf.cast(5+tensor, tf.float32) ** self.alpha /
                tf.cast(5+1, tf.float32) ** self.alpha)


def add_beam_axis(tensor, beam_width):
    ndim = K.ndim(tensor) + 1
    multipliers = (
        tf.tensor_scatter_nd_update(
            tf.ones((ndim,), tf.int32), [[1]], [beam_width])
    )
    tensor = tf.expand_dims(tensor, 1)
    return tf.tile(tensor, multipliers)


def log_with_clip(tensor):
    return tf.math.log(tf.clip_by_value(tensor, 1e-20, 1 - 1e-20))


def mask_first_step_beam_probs(initial_length, beam_probs):
    is_first_step = tf.reduce_sum(initial_length) <= 0
    mask_beam_probs = tf.concat([beam_probs[:, :1],
                                 tf.zeros_like(beam_probs[:, 1:])], axis=1)
    return tf.cond(is_first_step,
                   lambda: mask_beam_probs,
                   lambda: beam_probs)


def convert_to_log_prob(beam_probs, beam_width):
    한글_prob, 초성_prob, 중성_prob, 종성_prob, 특수_prob = tf.split(
        beam_probs, [1, len(초성) + 1, len(중성) + 1, len(종성) + 1, -1], axis=-1)
    조합_prob = (
          초성_prob[:, :, :-1, None, None]
        + 중성_prob[:, :, None, :-1, None]
        + 종성_prob[:, :, None, None, :-1])/3
    조합_prob = tf.reshape(
        조합_prob, (-1, beam_width, len(초성)*len(중성)*len(종성)))
    log_한글_probs = (log_with_clip(한글_prob) + log_with_clip(조합_prob))/2
    log_특수_probs = (log_with_clip(1-한글_prob) + log_with_clip(특수_prob[:,:,:-1]))/2

    log_beam_probs = tf.concat([log_한글_probs, log_특수_probs], axis=-1)
    return log_beam_probs


__all__ = ['BeamSearchInference', "AttentionInference"]

