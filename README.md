# AI OCR 서버 구축 및 backend Python


## 1. tensorflow/serving

### Tensorflow-Serving 이미지 가져오기
모델을 배포하기 위해서는 Tensorflow-serving 이미지를 가져와야 합니다.

docker pull tensorflow/serving:latest

### OCR 모델을 실행시켜주는 서버 실행

아래는 `archieve/alpha/serving`에 저장되어 있는 텐서플로우 모델을 Docker Image를 통해, 서버 형태로 배포하는 코드입니다. 

docker run -d --rm -p 8500:8500 -p 8501:8501 -v "$(pwd)/ocr-text-recognition/archieve/alpha/serving/:/models/ocr" -e MODEL_NAME=ocr tensorflow/serving

도커가 서버로서 실행됨. 실행중임.

## 2. python

### (1) Python 실행환경 image 구혐

docker build -t {이미지명} .
    
### (2) Docker 실행환경 container (자동 실행까지)

docker run -it --name aidter -v "$(pwd)/tti:/volume_state" aidt

## 관련링크
https://pidongsadong.tistory.com/7
