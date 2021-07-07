import os
import tensorflow as tf
from korean_ocr.layers import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras import backend as K
from korean_ocr.utils.jamo import compose_unicode
import docker
import time
import json
import base64
import requests
from tensorflow.python.keras.models import load_model
import glob
TAG_NAME = "tensorflow/serving:latest" # Tensorflow Serving Image NAME

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TEST_IMAGE_DIR = os.path.join(ROOT_DIR, 'tests/images/')


def to_ocr_model(model: tf.keras.Model):
    # 예외 케이스 처리 : multi-gpu인 경우
    for layer in model.layers:
        if layer.name == 'ocr_model':
            # multi-GPU case
            ocr_model = layer
            break
    else:
        # singl-GPU case
        ocr_model = model
    return ocr_model


def convert_model_to_inference_model(model: tf.keras.Model,
                                     decoder_length=30,
                                     image_format_input=False):
    ocr_model = to_ocr_model(model)

    # Decoder Layer 가져오기
    embedding_layer = ocr_model.get_layer(
        "decoder/character_embedding")
    attention_layer = ocr_model.get_layer(
        "decoder/attention")
    classify_layer = ocr_model.get_layer(
        "decoder/classify")

    # 디코더 구성하기
    inference_layer = AttentionInference(
        embedding_layer, attention_layer, classify_layer, decoder_length)

    # Input Tensor 가져오기
    seq_maps, dec_inputs, enc_masks, dec_masks = attention_layer.input
    outputs = inference_layer([seq_maps, enc_masks])
    outputs = PaddingEOS()(outputs)

    if image_format_input:
        raw_inputs = Input((), dtype=tf.string, name='image')
        inputs = DecodeImageContent()(raw_inputs)
        model = Model(ocr_model.inputs[0], outputs)
        unicode_prediction = model(inputs)
        return Model(raw_inputs, unicode_prediction, name='ocr')
    else:
        model = Model(ocr_model.inputs[0], outputs, name='ocr')
        return model


def save_ocr_model_to_savedmodel_format(weight_path, serving_dir:str, decoder_length=30):
    """
    OCR Model을 Tensorflow에서 제공하는 SavedModel format으로 변경하여 저장하는 메소드

    :param weight_path: tensorflow.keras.Model이 저장된 .h5 파일 경로
    :param serving_dir: 서빙모델이 저장된 공간
    :param decoder_length: 모델의 최대 추론 길이
    :return:
    """
    model = load_model(weight_path, compile=False)
    serving = convert_model_to_inference_model(model, decoder_length, image_format_input=True)

    version_dir = get_latest_version_dir(serving_dir)
    K.set_learning_phase(0)
    session = K.get_session()

    tf.saved_model.simple_save(session, version_dir,
                               inputs={'image': serving.input},
                               outputs={'text': serving.output},
                               legacy_init_op=tf.tables_initializer())
    print(f"ocr 모델의 serving directory >>> {serving_dir}\n"
           "해당 경로에 SaveModel Format으로 저장된 모델 정보가 있습니다. docker tensorflow serving으로 해당 폴더를 지정하시면 됩니다.")
    return serving


def get_latest_version_dir(export_dir):
    """
    현재 export_dir 내 새 version directory의 경로를 가져오기

    :param export_dir:
    :return:
    """
    os.makedirs(export_dir,exist_ok=True)
    curr_version = max([0] + [int(version) for version in os.listdir(export_dir)]) + 1
    return os.path.join(export_dir, str(curr_version))


def get_docker_client():
    """
    Docker Client 정보를 가져오는 메소드

    :return:
    """
    client = docker.from_env()
    try:
        ping_test = client.ping()
    except:
        raise ValueError("Docker의 권한에 문제가 있습니다. 현재 user에 docker에 접근할 수 있는 권한을 제공하여야 합니다.\n"
                         "참고 -> https://stackoverflow.com/questions/48568172/docker-sock-permission-denied")
    if not ping_test:
        raise ValueError("Docker에 연결되지 않습니다. 현재 환경 내 Docker의 동작이 올바른지 확인해주시길 바랍니다.")
    return client


def pull_tf_serving_docker_image():
    """
    tensorflow serving image를 가져오는 메소드

    :return:
    """
    global TAG_NAME
    client = get_docker_client()

    tags = sum([image.tags
                for image in client.images.list()], [])
    if not TAG_NAME in tags:
        client.images.pull(TAG_NAME)


def run_tf_serving_docker_container(serving_dir, grpc_port=8500, rest_port=8501):
    """
    Tensorflow serving을 위한 도커 컨테이너를 활성화시키는 메소드

    :param serving_dir:
    :param grpc_port: grpc api 통신을 위한 포트 번호
    :param rest_port: rest api 통신을 위한 포트 번호
    :return:
    """
    global TAG_NAME
    client = get_docker_client()

    # 돌아가고 있는 container가 존재하는 경우, 이를 종료
    for container in client.containers.list():
        if TAG_NAME in container.image.tags:
            container.kill()

    client.containers.run(TAG_NAME, detach=True,
                          ports={'8500/tcp': str(grpc_port),
                                 "8501/tcp": str(rest_port)},
                          environment=['MODEL_NAME=ocr'],
                          volumes={
                              os.path.abspath(serving_dir): {"bind": "/models/ocr", 'mode': 'rw'}})
    # Docker Container가 정상적으로 Launching되기 위해 wait
    time.sleep(10)
    # 올바르게 동작하는지 평가
    test_process_image_to_serving_model(f"http://localhost:{rest_port}")


def test_process_image_to_serving_model(url):
    """
    Serving 모델이 올바르게 동작하는지 파악
    """
    consumed = 0
    result = 0
    fpaths = glob.glob(os.path.join(TEST_IMAGE_DIR, "*.png"))
    for fpath in fpaths:
        with open(fpath, 'rb') as f:
            image_string = base64.b64encode(f.read()).decode('utf-8')

        data = json.dumps(
            {'instances': [{'image':{'b64':image_string}}]})

        s = time.time()
        response = requests.post(f'{url}/v1/models/ocr:predict',data=data)
        consumed += time.time() - s

        codes = response.json()['predictions']
        pred = compose_unicode(codes)[0]
        gt = os.path.split(fpath)[1].split('.')[0]
        result += int(pred == gt)

    count = len(fpaths)
    acc = result / count
    speed = consumed / count
    print(f"총 {count}개의 이미지를 대상으로 모델 동작평가")
    print("-------------------------")
    print(f"정확도 : {acc:.3%}")
    print(f"속도 : {speed*1000:.3f}ms")

