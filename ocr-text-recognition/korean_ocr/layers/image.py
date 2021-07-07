from tensorflow.python.keras.layers import Conv2D, Layer
from tensorflow.python.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.python.keras import backend as K
import tensorflow as tf


class PreprocessImage(Layer):
    """ Convolution Layer의 Input Format에 맞게 형변환 해주는 모듈

    1. Resize
        (batch size, image height, image width) -> (batch size, image height, image width, 1)
    2. Type 변환
        tf.uint8 -> tf.float32
    3. Normalize
        X range : [0, 255] -> [-1., 1.]
    """
    def __init__(self, height=64, normalize=True, zero_mean=True, **kwargs):
        self.height = height
        self.normalize = normalize
        self.zero_mean = zero_mean
        kwargs.setdefault('trainable', False)
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        inputs = self.resize_inputs(inputs)

        mask = 1 - tf.cast(
            tf.reduce_all(tf.equal(inputs, 0.),
                          axis=1, keepdims=True), tf.float32)
        mask = tf.tile(mask, [1, tf.shape(inputs)[1], 1, 1])

        if self.normalize:
            if self.zero_mean:
                inputs = (inputs - 127.5) / 127.5
            else:
                inputs = inputs / 255.

        return inputs, mask

    def resize_inputs(self, inputs):
        b, h, w = tf.unstack(tf.shape(inputs)[:3])  # Dynamic Shape

        inputs = tf.cast(inputs, tf.float32)
        inputs = tf.reshape(inputs, (b, h, w, 1))

        def resize_by_height(inputs):
            new_h = self.height
            new_w = tf.cast(tf.math.ceil(w / h * new_h), tf.int32)
            inputs = tf.image.resize(inputs, (new_h, new_w))
            return inputs
        inputs = tf.cond(tf.equal(h, self.height),
                         lambda: inputs,
                         lambda: resize_by_height(inputs))
        inputs.set_shape([None, self.height, None, 1]) # Dynamic Shape to Static Shape
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "height": self.height,
            "normalize": self.normalize,
            "zero_mean": self.zero_mean
        })
        return config


class ConvFeatureExtractor(Layer):
    """ Image Encoder Class,

    Original Implementation와는 영상의 해상도가 달라, 좀 더 깊은 Feature Extractor로 구성

    변경 사항
    1. VGG Style에서 ResNet Style로 Feature Extractor 변경
    2. VGG Style로 7 layer으로 구성된 것을, ResBlock 단위로 7 Block(== 14 layer)로 변경
    3. Batch Normalization을 추가하여 빠르게 수렴할 수 있도록 함
    """
    def __init__(self,
                 filters=(32, 64, 128, 128, 256, 256),
                 strides=(2, 2, 1, 2, 1, 2),
                 **kwargs):
        assert len(filters) == len(strides), "filters의 리스트 크기와 strides의 리스트 크기는 동일해야 합니다."
        self.filters = filters
        self.strides = strides

        f = filters[0]
        s = strides[0]
        self.conv1_1 = Conv2D(f, (3, 3), padding='same', use_bias=False)
        self.norm1_1 = BatchNormalization()
        self.conv1_2 = Conv2D(f, (3, 3), padding='same', use_bias=False)
        self.norm1_2 = BatchNormalization()
        self.maxpool1 = MaxPooling2D((s, s), (s, s), padding='same')

        self.blocks = []
        for f, s in zip(self.filters[1:], self.strides[1:]):
            self.blocks.append(ResidualLayer(f, s))

        self.final_conv = Conv2D(self.filters[-1], (2, 2), padding='valid')
        self.final_norm = BatchNormalization()
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = inputs[0]
        mask = inputs[1]

        x = self.conv1_1(x)
        x = self.norm1_1(x)
        x = K.relu(x)
        x = self.conv1_2(x)
        x = self.norm1_2(x)
        x = K.relu(x)
        x = self.maxpool1(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_conv(x)
        x = self.final_norm(x)
        outputs = K.relu(x)

        mask = self.resize_mask(mask, outputs)
        return outputs, mask

    def resize_mask(self, mask, target):
        target_shape = tf.shape(target)[1:3]
        mask = tf.image.resize(mask, target_shape,
                               tf.image.ResizeMethod.BILINEAR)
        mask = tf.cast(mask > 0.5, tf.float32)
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "strides": self.strides,
            "filters": self.filters,
        })
        return config


class ResidualLayer(Layer):
    """
    Residual Connection이 포함된 ResNet-Block
    """
    def __init__(self, filters, strides=1, **kwargs):
        self.filters = f = filters
        self.strides = s = strides
        super().__init__(**kwargs)

        self.skip = Conv2D(f, (1, 1), padding='same', use_bias=False)
        self.conv1 = Conv2D(f, (3, 3), padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(f, (3, 3), padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        self.pool = MaxPooling2D((s, s), (s, s), padding='same')

    def call(self, inputs, **kwargs):
        skipped = self.skip(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = K.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = K.relu(skipped + x)
        x = self.pool(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "strides": self.strides
        })
        return config


class Map2Sequence(Layer):
    """ CNN Layer의 출력값을 RNN Layer의 입력값으로 변환하는 Module Class
    Transpose & Reshape을 거쳐서 진행

    CNN output shape  ->  RNN Input Shape

    (batch size, height, width, channels)
    -> (batch size, width, height * channels)

    """
    def call(self, inputs, **kwargs):
        x = inputs[0]
        mask = inputs[1]

        b, _, w, _ = tf.unstack(tf.shape(x))
        _, h, _, f = x.shape.as_list()

        x = K.permute_dimensions(x, (0, 2, 1, 3))
        outputs = tf.reshape(x, shape=[b, w, h * f])

        mask = K.permute_dimensions(mask, (0, 2, 1, 3))
        mask = tf.reshape(mask, shape=[b, w, h])
        mask = tf.reduce_all(mask > 0.5, axis=-1)
        return outputs, mask


class SpatialTransformer(Layer):
    """
    * CAUTION *
    STN 네트워크는 RARE 모델에서 포함시킨 것과 포함시키지 않은 것의 성능이 큰 차이가 없고,
    STN이 포함될 경우, 이미지 크기를 고정시켜야 한다는 문제가 있어서, 제거하여 구현.

    Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [4]_:
    downsample_fator : float
        A value of 1 will keep the orignal size of the image.
        Values larger than 1 will down sample the image. Values below 1 will
        upsample the image.
        example image: height= 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
    .. [4]  https://github.com/sbillburg/CRNN-with-STN/blob/master/STN/spatial_transformer.py
    """

    def __init__(self,
                 localization_net,
                 output_size,
                 **kwargs):
        self.loc_net = localization_net
        self.output_size = output_size
        super(SpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.loc_net.build(input_shape)
        self.trainable_weights = self.loc_net.trainable_weights

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]),
                int(output_size[1]),
                int(input_shape[-1]))

    def call(self, X, mask=None):
        affine_transformation = self.loc_net.call(X)
        output = self._transform(affine_transformation, X, self.output_size)
        return output

    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype=tf.int32)
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, image, x, y, output_size):
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
        num_channels = tf.shape(image)[3]

        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)

        height_float = tf.cast(height, dtype=tf.float32)
        width_float = tf.cast(width, dtype=tf.float32)

        output_height = output_size[0]
        output_width = output_size[1]

        x = .5*(x + 1.0)*width_float
        y = .5*(y + 1.0)*height_float

        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype=tf.int32)
        max_x = tf.cast(width - 1,  dtype=tf.int32)
        zero = tf.zeros([], dtype=tf.int32)

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype=tf.float32)
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        y0 = tf.cast(y0, tf.float32)
        y1 = tf.cast(y1, tf.float32)

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        output = tf.add_n([area_a*pixel_values_a,
                           area_b*pixel_values_b,
                           area_c*pixel_values_c,
                           area_d*pixel_values_d])
        return output

    def _meshgrid(self, height, width):
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, shape=(1, -1))
        y_coordinates = tf.reshape(y_coordinates, shape=(1, -1))
        ones = tf.ones_like(x_coordinates)
        indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
        return indices_grid

    def _transform(self, affine_transformation, input_shape, output_size):
        batch_size, _, _, num_channels = tf.unstack(tf.shape(input_shape))

        affine_transformation = tf.reshape(affine_transformation, shape=(batch_size,2,3))

        affine_transformation = tf.reshape(affine_transformation, (-1, 2, 3))
        affine_transformation = tf.cast(affine_transformation, tf.float32)

        output_height = output_size[0]
        output_width = output_size[1]
        indices_grid = self._meshgrid(output_height, output_width)
        indices_grid = tf.expand_dims(indices_grid, 0)
        indices_grid = tf.reshape(indices_grid, [-1]) # flatten?
        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, tf.stack([batch_size, 3, -1]))

        # transformed_grid = tf.batch_matmul(affine_transformation, indices_grid)
        transformed_grid = tf.matmul(affine_transformation, indices_grid)
        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
        x_s_flatten = tf.reshape(x_s, [-1])
        y_s_flatten = tf.reshape(y_s, [-1])

        transformed_image = self._interpolate(input_shape,
                                              x_s_flatten,
                                              y_s_flatten,
                                              output_size)

        transformed_image = tf.reshape(transformed_image, shape=(batch_size,
                                                                 output_height,
                                                                 output_width,
                                                                 num_channels))
        return transformed_image


class DecodeImageContent(Layer):
    """
    Image Format(jpg, jpeg, png)으로 압축된 이미지를 Decode하여
    Tensorflow Array를 반환
    """
    def call(self, inputs, **kwargs):
        # From Rank 1 to Rank 0
        inputs = tf.reshape(inputs,shape=())
        image = tf.io.decode_image(inputs, channels=1,
                                   expand_animations=False)
        image = tf.expand_dims(image, axis=0)
        image = tf.squeeze(image, axis=-1)
        return image


__all__ = ["PreprocessImage",
           "ConvFeatureExtractor",
           "Map2Sequence",
           "SpatialTransformer",
           "DecodeImageContent"]