import os
import pandas as pd
from datetime import datetime
from multiprocessing import cpu_count
import json
import time
import tensorflow as tf
import warnings
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.optimizers import Adadelta, Adam, SGD
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard

from korean_ocr.layers import PreprocessImage, ConvFeatureExtractor
from korean_ocr.layers import Map2Sequence
from korean_ocr.layers import SequenceEncoder, AdditionPositionalEncoding
from korean_ocr.layers import AttentionDecoder
from korean_ocr.layers import CharClassifier, CharEmbedding
from korean_ocr.layers.text import DEFAULT_SPECIAL_CHARACTERS

from korean_ocr.data.dataset import OCRDataset
from korean_ocr.data.dataset import read_label_dataframe
from korean_ocr.data.dataset import filter_out_dataframe
from korean_ocr.data.generator import DataGenerator
from korean_ocr.utils.metrics import WordAccuracy
from korean_ocr.utils.optimizer import RectifiedAdam, AdamW
from korean_ocr.utils.losses import CharCategoricalCrossEntropy


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))
LOG_DIR = os.path.join(ROOT_DIR, 'logs/')
curr_hour = datetime.now().strftime("%m%d-%H")
SAVE_DIR = os.path.join(LOG_DIR, curr_hour)
DATA_DIR = os.path.join(ROOT_DIR, 'datasets')

"""
해당 메소드는 3가지 단계를 거쳐서 실행되어야 합니다.

1. build_ocr_model() 
   : OCR 모델을 정의하는 부분
2. compile_ocr_model()
   : OCR 모델의 학습 방식을 정의하는 메소드
3. train_ocr_model()
   : OCR 모델을 학습하는 메소드
"""


def build_ocr_model(height=64,
                    num_embed=16,
                    filters=(32, 64, 128, 128, 256, 256),
                    strides=(2, 2, 1, 2, 1, 2),
                    num_states=256,
                    num_fc=256,
                    special_characters=DEFAULT_SPECIAL_CHARACTERS,
                    use_positional_encoding=False):
    """
    pre: height > 0

    robust scene text recognition with automatic rectification을 한글 손글씨에 적용시킨 모델.
    현재 손글씨 데이터셋의 주요 특징으로 3가지가 있음
        1. 횡방향으로 높은 해상도를 요함 ( 한글은 자모자의 조합으로 되어 있어, 영어보다 복잡한 구조를 띄기 때문)
        2. 조합형 문자로 조합의 수는 11,172자 개만큼 있어, 각 조합 당 데이터 수는 매우 적음.
        3. 문장 단위 손글씨 정보가 있어, 횡방향으로 매우 긴 정보를 가지고 있음

    1. 데이터의 높은 해상도
       영어 글자 데이터는 높이 방향으로 32 Pixel 수준으로도 충분하지만,
       한글 글자 데이터는 영어보다 복잡하게 구성되어 있어, 64 Pixel 수준으로 적용할 필요가 있음.

       해결 방향
         -> Residual Connection 추가
         -> Batch Normalization 추가
         -> 깊이를 깊게 함 (대신 Filter 수를 줄임)

    2. 한글의 조합형 언어 특징
       한글은 초성(19자), 중성(21자), 종성(27자)의 조합으로 이루어진 언어로, 조합하여 만들 수 있는 총 글자 수는
       11,172자에 달함. 단일 모델로 11,172자를 학습시키는 것은 Sparse한 라벨의 문제가 발생하고,
       마지막 Classifciation Layer에 많은 Parameter를 필요로 하기 때문에 다른 방법을 모색함.

       해결 방향
         -> 초성 / 중성 / 종성 / 특수문자를 나누어 분류기를 만듦

    3. [Optional] 글자 영상이 종방향으로 매우 김
       LSTM 모델로 종방향으로 매우 긴 모델을 학습을 시키려면, 학습이 불안정하게 이루어짐. 이를 해결하기 위해,
       Stacked BiDirectional LSTM 모델 대신 Transformer 모델에서 제안된 positional encoding 방식을
       도입하였음. LSTM 모델이 제거되었기 떄문에 메모리 사용량이 크게 줄었다.

       해결 방향
        -> Stacked Bidirectional LSTM Layer 대신 PositionalEncoding Layer 추가

    :param height: 고정된 이미지의 높이
    :param num_embed : 초성 / 중성 / 종성 / 특수 기호에 대한 Embedding Vector의 크기
    :param filters: Convolution Network에서의 각 층 별 Filter 갯수
    :param strides: Convolution Network에서의 각 층 별 Strides 갯수
    :param num_states: Recurrent & Attention Layer의 State 크기
    :param num_fc: Classification Layer의 Unit 크기
    :param special_characters: 한글 외 포함시켜야 하는 문자
    :param use_positional_encoding: [Experimental],
    SequenceEncoder 대신, Feature Map에 Positional Encoding 정보만을 추가하여, RNN 연산 부분을 제거.
    연산 시간과 메모리 사용량이 줄어들고, 정확도는 BLSTM이 있을 때와 유사한 수준으로 기록됨.

    :return: tensorflow.keras.Model
    OCR 모델로 구성된 Tensorflow 모델

    모델 Input : Gray Scale로 구성된 텍스트 이미지
        * Tensor shape : (batch size, height, width)
        * Input range : [0, 255]

    """
    images = Input(shape=(None, None), name='images')

    preps, masks = PreprocessImage(
        height=height, normalize=True, name='encoder/preprocess')(images)

    conv_maps, conv_masks = ConvFeatureExtractor(
        filters, strides, name='encoder/feature_extractor')((preps, masks))
    feat_maps, enc_masks = Map2Sequence(
        name='encoder/map_to_sequence')((conv_maps, conv_masks))

    if use_positional_encoding:
        # Revised Implementation
        states_encoder = AdditionPositionalEncoding(
            name='encoder/positional_encoder')(feat_maps)
    else:
        # Original Paper Implementation
        states_encoder = SequenceEncoder(
            recurrent_cell='lstm', num_states=num_states,
            name='encoder/blsm_encoder')((feat_maps, enc_masks))

    decoder_inputs = Input(
        shape=(None,), dtype=tf.int32, name='decoder_inputs')
    states_decoder, dec_masks = CharEmbedding(
        special_characters=special_characters, num_embed=num_embed,
        name='decoder/character_embedding')(decoder_inputs)

    states = AttentionDecoder(
        num_states=num_states, name='decoder/attention')(
        [states_encoder, states_decoder, enc_masks, dec_masks])

    prediction = CharClassifier(
        special_characters=special_characters,
        num_fc=num_fc, name='decoder/classify')(states)
    model = Model([images, decoder_inputs], prediction, name='ocr_model')
    return model


def compile_ocr_model(model:tf.keras.Model,
                      optimizer='adadelta',
                      lr=1.,
                      num_gpus=1):
    """
    OCR 모델을 학습시키기 위해 필요한 요소들을 정의하는 부분.

    1. Optimizer : RectifiedAdam, AdamW, Adadelta, Adam 중에서 선택.
    2. Loss : 목표 함수
    3. Metric : Word Accuracy. Text Recognition 모델에서 주로 이용하는 Metric으로 텍스트 간 유사도를 평가할 때 쓰임

    :param model: tensorflow.keras.Model, build_ocr_model()의 return value
    :param optimizer:  RectifiedAdam, AdamW, Adadelta, Adam 중에서 선택.
    :param lr: 학습률
    :param num_gpus: 학습에 이용할 GPU의 갯수

    :return: tensorflow.keras.Model
    OCR 모델로 구성된 Tensorflow 모델

    """
    y_true = Input(shape=(None,), dtype=tf.int32, name='labels')

    if optimizer.lower() == 'rectifiedadam':
        optim = RectifiedAdam(lr=lr)
    elif optimizer.lower() == 'adamw':
        optim = AdamW(lr=lr)
    elif optimizer.lower() == 'adadelta':
        optim = Adadelta(lr=lr)
    elif optimizer.lower() == 'adam':
        optim = Adam(lr=lr, beta_2=0.98, epsilon=1e-9)
    elif optimizer.lower() == 'momentum':
        optim = SGD(lr=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError("optimizer should be one of {rectifiedadam / adamw / adadelta / adam }")

    if num_gpus > 1:
        model = multi_gpu_model(model, num_gpus)

    special_characters = get_attr_from_model_config(model, 'special_characters')
    model.compile(optim,
                  loss=CharCategoricalCrossEntropy(special_characters),
                  metrics=[WordAccuracy()],
                  target_tensors=y_true)
    return model


def train_ocr_model(model:tf.keras.Model,
                    data_dir=(os.path.join(DATA_DIR, "handwritten"),
                              os.path.join(DATA_DIR, 'printed')),
                    types=('글자(음절)','단어(어절)',),
                    batch_size=64,
                    steps=100000,
                    save_dir=SAVE_DIR,
                    use_multiprocessing=True):
    """
    OCR 모델을 데이터셋을 통해 학습

    :param model: tensorflow.keras.Model, build_ocr_model()의 return value
    :param data_dir: data directory, data directory에는 반드시 images/와 dataset_info.json이 있어야 함
    :param types: 이용할 데이터 타입의 종류 ('글자(음절)','단어(어절)','문장')
    :param batch_size: 배치 크기
    :param steps: 총 학습 횟수 (1 epoch에 1000 steps로 고정되어 있음)
    :param save_dir: 모델의 로그 기록, weight가 담긴 .h5 파일 등을 저장할 폴더
    :param use_multiprocessing: MultiProcessing을 통해 fit_generator를 운영할지 유무

    :return: tensorflow.keras.Model
    OCR 모델로 구성된 Tensorflow 모델

    """
    print("\ntrain_ocr_model의 학습 과정 및 모델이 저장될 경로"
          "\n아래 경로에는 Tensorboard Log 파일 및 .h5 파일, SavedModel 파일이 저장"
          "\n-------------------------"
          f"\nModel Log Directory : {save_dir}"
          "\n-------------------------")

    if isinstance(data_dir, str):
        data_dirs = [data_dir]
    elif isinstance(data_dir, list) or isinstance(data_dir, tuple):
        data_dirs = data_dir
    else:
        raise ValueError(
            "data_dir은 데이터 폴더의 경로로, images/와 dataset_info.json이\n"
            "저장된 폴더 경로를 지정해주어야 합니다.\n"
            "복수개의 데이터 셋을 이용하고 싶은 경우, 담긴 데이터 폴더의 경로를 리스트에 담아 넣으시면 됩니다.")

    print("\n학습할 데이터 불러오는 중...")
    start = time.time()
    train_dfs = []
    valid_dfs = []
    test_dfs = []
    for dataset_dir in data_dirs:
        label_path = os.path.join(dataset_dir, "dataset_info.json")

        label_df = read_label_dataframe(label_path)
        label_df = filter_out_dataframe(label_df)

        if "type" in label_df.columns:
            label_df = label_df[label_df.type.isin(types)]

        try:
            with open(os.path.join(dataset_dir, "train.json"), 'r') as f:
                train_flist = json.load(f)
                train_sub = label_df[label_df.file_name.isin(train_flist)]
            with open(os.path.join(dataset_dir, "validation.json"), 'r') as f:
                valid_flist = json.load(f)
                valid_sub = label_df[label_df.file_name.isin(valid_flist)]
            with open(os.path.join(dataset_dir, "test.json"), 'r') as f:
                test_flist = json.load(f)
                test_sub = label_df[label_df.file_name.isin(test_flist)]
        except FileNotFoundError:
            warnings.warn("train / validation / test 데이터셋이 나누어져 있지 않습니다. 임의로 7:1:2로 나누도록 하겠습니다.")
            train_sub, test_sub = train_test_split(label_df, test_size=0.2, random_state=42)
            train_sub, valid_sub = train_test_split(train_sub, test_size=1/8, random_state=50)
        train_dfs.append(train_sub)
        valid_dfs.append(valid_sub)
        test_dfs.append(test_sub)

    train_df = pd.concat(train_dfs)
    valid_df = pd.concat(valid_dfs)
    test_df = pd.concat(test_dfs)

    height = get_attr_from_model_config(model, 'height')
    trainset = OCRDataset(train_df, height=height)
    validset = OCRDataset(valid_df, height=height)
    testset = OCRDataset(test_df, height=height)
    end = time.time()
    print(f"소요시간 : {end-start:.3f}s-\n")

    print("\n학습할 데이터의 모수")
    print("--------------------------")
    print(f"trainset의 갯수 : {len(trainset):,}")
    print(f"validset의 갯수 : {len(validset):,}")
    print(f"testset의 갯수 : {len(testset):,}")
    print(f"-------------------------\n")

    traingen = DataGenerator(trainset, batch_size=batch_size)
    validgen = DataGenerator(validset, batch_size=batch_size, augments=None)

    callbacks = []
    rlrp = ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7, verbose=1)
    callbacks.append(rlrp)

    ckpt = ModelCheckpoint(os.path.join(save_dir, "{val_loss:.5f}-{epoch:03d}.h5"))
    callbacks.append(ckpt)

    tb = TensorBoard(log_dir=os.path.join(save_dir, 'logs'))
    callbacks.append(tb)

    print("\n모델 학습 시작...\n")
    print("--------------------------")
    epochs = steps // 1000
    if use_multiprocessing:
        workers = cpu_count()
    else:
        workers = 1
    model.fit_generator(traingen,
                        steps_per_epoch=1000,
                        validation_data=validgen,
                        validation_steps=1000,
                        use_multiprocessing=use_multiprocessing,
                        max_queue_size=10,
                        workers=workers,
                        epochs=epochs,
                        callbacks=callbacks)
    print(f"-------------------------\n")
    return model


def get_attr_from_model_config(model, attr_name="special_characters"):
    """
    Model Configuration에서 Attribute을 가져오는 메소드

    :param model: tensorflow.keras.Model, build_ocr_model()의 return value
    :param serving_dir: Serving 모델이 저장된 공간
    :return:
    """
    for layer in model.layers:
        if 'layers' in layer.get_config():
            # Multi-GPU Model Case
            attr = get_attr_from_model_config(layer, attr_name)
            if attr is not False:
                return attr
        if attr_name in layer.get_config():
            # Single-GPU Model Case
            return layer.get_config()[attr_name]
    return False
