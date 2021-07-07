from tensorflow.python.keras.engine.base_layer import AddMetric, AddLoss
from tensorflow.python.keras.utils import get_custom_objects
from korean_ocr.layers.image import PreprocessImage, ConvFeatureExtractor
from korean_ocr.layers.image import Map2Sequence, SpatialTransformer
from korean_ocr.layers.sequence import SequenceEncoder, AttentionDecoder
from korean_ocr.layers.sequence import AdditionPositionalEncoding
from korean_ocr.layers.text import CharDeCompose, CharCompose
from korean_ocr.layers.text import CharEmbedding, CharClassifier
from korean_ocr.layers.inference import BeamSearchInference, AttentionInference
from korean_ocr.utils.optimizer import AdamW, RectifiedAdam
from korean_ocr.utils.losses import CharCategoricalCrossEntropy
from korean_ocr.utils.metrics import WordAccuracy
import korean_ocr.data
import korean_ocr.layers
import korean_ocr.utils

# Custom Layer 구성하기
get_custom_objects().update({
    "PreprocessImage": PreprocessImage,
    "ConvFeatureExtractor": ConvFeatureExtractor,
    "Map2Sequence": Map2Sequence,
    "SpatialTransformer": SpatialTransformer,
    "SequenceEncoder": SequenceEncoder,
    "AttentionDecoder": AttentionDecoder,
    "AdditionPositionalEncoding": AdditionPositionalEncoding,
    "CharClassifier": CharClassifier,
    "BeamSearchInference": BeamSearchInference,
    "DecoderInference": AttentionInference,
    "CharCompose": CharCompose,
    "CharDeCompose": CharDeCompose,
    "CharEmbedding": CharEmbedding,
})

# Custom Optimizer 구성하기
get_custom_objects().update({'AdamW': AdamW,
                             'RectifiedAdam': RectifiedAdam})

# Custom Loss 구성하기
get_custom_objects().update({
    "CharCategoricalCrossEntropy": CharCategoricalCrossEntropy
})

# Custom Metric 구성하기
get_custom_objects().update({'WordAccuracy': WordAccuracy})

# BUGS!!!> Keras 기본 인자인데, 세팅이 안되어 있어서, save Model & Load Model에서
# 따로 지정해주어야 함
get_custom_objects().update({'AddMetric': AddMetric,
                             'AddLoss': AddLoss})
