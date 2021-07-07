from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Constant


def ssd384_body(x):
    source_layers = []

    # Block 1
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_1', activation='relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_2', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool1')(x)
    # Block 2
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_1', activation='relu')(x)
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_2', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool2')(x)
    # Block 3
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_2', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_3', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool3')(x)
    # Block 4
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_3', activation='relu')(x)
    source_layers.append(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool4')(x)
    # Block 5
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_3', activation='relu')(x)
    x = MaxPooling2D(pool_size=3, strides=1, padding='same', name='pool5')(x)
    # FC6
    x = Conv2D(1024, 3, strides=1, dilation_rate=(6, 6), padding='same', name='fc6', activation='relu')(x)
    # FC7
    x = Conv2D(1024, 1, strides=1, padding='same', name='fc7', activation='relu')(x)
    source_layers.append(x)
    # Block 6
    x = Conv2D(256, 1, strides=1, padding='same', name='conv6_1', activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, 3, strides=2, padding='valid', name='conv6_2', activation='relu')(x)
    source_layers.append(x)
    # Block 7
    x = Conv2D(128, 1, strides=1, padding='same', name='conv7_1', activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv7_2', activation='relu')(x)
    source_layers.append(x)
    # Block 8
    x = Conv2D(128, 1, strides=1, padding='same', name='conv8_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='valid', name='conv8_2', activation='relu')(x)
    source_layers.append(x)
    # Block 9
    x = Conv2D(128, 1, strides=1, padding='same', name='conv9_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='valid', name='conv9_2', activation='relu')(x)
    source_layers.append(x)

    return source_layers


def multibox_head(source_layers, num_priors, normalizations=None, softmax=True):
    
    num_classes = 2
    class_activation = 'softmax' if softmax else 'sigmoid'

    mbox_conf = []
    mbox_loc = []
    mbox_quad = []
    mbox_rbox = []
    for i in range(len(source_layers)):
        x = source_layers[i]
        name = x.name.split('/')[0]
        
        # normalize
        if normalizations is not None and normalizations[i] > 0:
            name = name + '_norm'
            x = Normalize(normalizations[i], name=name)(x)
            
        # confidence
        name1 = name + '_mbox_conf'
        x1 = Conv2D(num_priors[i] * num_classes, (3, 5), padding='same', name=name1)(x)
        x1 = Flatten(name=name1+'_flat')(x1)
        mbox_conf.append(x1)

        # location, Delta(x,y,w,h)
        name2 = name + '_mbox_loc'
        x2 = Conv2D(num_priors[i] * 4, (3, 5), padding='same', name=name2)(x)
        x2 = Flatten(name=name2+'_flat')(x2)
        mbox_loc.append(x2)
        
        # quadrilateral, Delta(x1,y1,x2,y2,x3,y3,x4,y4)
        name3 = name + '_mbox_quad'
        x3 = Conv2D(num_priors[i] * 8, (3, 5), padding='same', name=name3)(x)
        x3 = Flatten(name=name3+'_flat')(x3)
        mbox_quad.append(x3)

        # rotated rectangle, Delta(x1,y1,x2,y2,h)
        name4 = name + '_mbox_rbox'
        x4 = Conv2D(num_priors[i] * 5, (3, 5), padding='same', name=name4)(x)
        x4 = Flatten(name=name4+'_flat')(x4)
        mbox_rbox.append(x4)
        
    mbox_conf = concatenate(mbox_conf, axis=1, name='mbox_conf')
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation(class_activation, name='mbox_conf_final')(mbox_conf)
    
    mbox_loc = concatenate(mbox_loc, axis=1, name='mbox_loc')
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)
    
    mbox_quad = concatenate(mbox_quad, axis=1, name='mbox_quad')
    mbox_quad = Reshape((-1, 8), name='mbox_quad_final')(mbox_quad)
    
    mbox_rbox = concatenate(mbox_rbox, axis=1, name='mbox_rbox')
    mbox_rbox = Reshape((-1, 5), name='mbox_rbox_final')(mbox_rbox)

    predictions = concatenate([mbox_loc, mbox_quad, mbox_rbox, mbox_conf], axis=2, name='predictions')
    
    return predictions


def TBPP384(input_shape=(384, 384, 3), softmax=True):
    """
    TextBoxes++384 architecture.

    # Arguments
        input_shape: Shape of the input image.
    
    # References
        - [TextBoxes++: A Single-Shot Oriented Scene Text Detector](https://arxiv.org/abs/1801.02765)
    """
    
    # SSD body
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd384_body(x)
    
    num_maps = len(source_layers)
    
    # Add multibox head for classification and regression
    num_priors = [14] * num_maps
    normalizations = [1] * num_maps
    output_tensor = multibox_head(source_layers, num_priors, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    
    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    
    model.aspect_ratios = [[1,2,3,5,1/2,1/3,1/5] * 2] * num_maps
    model.shifts = [[(0.0, -0.25)] * 7 + [(0.0, 0.25)] * 7] * num_maps
    model.special_ssd_boxes = False
    model.scale = 0.5
    
    return model


class Normalize(Layer):
    """
    Normalization layer as described in ParseNet paper.
    # Arguments
        scale: Default feature scale.
    # Input shape
        4D tensor with shape: (samples, rows, cols, channels)
    # Output shape
        Same as input
    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    # TODO
        Add possibility to have one scale for all features.
    """

    def __init__(self, scale=20, **kwargs):
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name=self.name + '_gamma',
                                     shape=(input_shape[-1],),
                                     initializer=Constant(self.scale),
                                     trainable=True)
        super(Normalize, self).build(input_shape)

    def call(self, x, mask=None):
        return self.gamma * K.l2_normalize(x, axis=-1)
