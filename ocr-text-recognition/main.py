import click
import os
from korean_ocr.utils.serving import run_tf_serving_docker_container
from korean_ocr.utils.serving import save_ocr_model_to_savedmodel_format
from korean_ocr.model import build_ocr_model
from korean_ocr.model import compile_ocr_model
from korean_ocr.model import train_ocr_model
from datetime import datetime
import glob


ROOT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = os.path.join(ROOT_DIR, 'datasets')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
LOG_SAVE_DIR = os.path.join(LOG_DIR, datetime.now().strftime("%Hh%Mm"))
ARCHIEVE_DIR = os.path.join(ROOT_DIR, 'archieve')

if __name__ == "__main__":
    @click.group()
    def cli():
        pass

    @cli.command()
    @click.option("--model_dir", default=os.path.join(MODEL_DIR, "alpha"),
                  help='배포할 모델이 저장된 폴더')
    @click.option('--grpc_port', default=8500,
                  help='grpc api 통신을 위한 포트 번호')
    @click.option('--rest_port', default=8501,
                  help='rest api 통신을 위한 포트 번호')
    def launch(model_dir, grpc_port, rest_port):
        serving_dir = os.path.join(model_dir, 'serving')
        if not os.path.exists(serving_dir):
            weight_paths = glob.glob(os.path.join(model_dir, '*.h5'))
            if len(weight_paths) == 0:
                raise FileNotFoundError(f"{model_dir}내에 weight 정보가 있는 h5파일이 존재하지 않습니다.")
            weight_path = sorted(weight_paths)[0]
            save_ocr_model_to_savedmodel_format(weight_path, serving_dir)
        run_tf_serving_docker_container(serving_dir, grpc_port, rest_port)

    @cli.command()
    def train():
        model = build_ocr_model()
        model = compile_ocr_model(model)
        model = train_ocr_model(model, save_dir=LOG_SAVE_DIR,
                                use_multiprocessing=False)
    cli()