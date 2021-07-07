FROM python:3.6

# 이미지 생성 과정에서 실행할 명령어
RUN mkdir /copy_state

RUN mkdir /volume_state

RUN pip install tensorflow==1.14

RUN apt-get install libsm6 libxext6 libxrender1 libxrender-dev libfontconfig1 

RUN apt-get update 

RUN apt-get -y install libgl1-mesa-glx

RUN pip install opencv-python

RUN pip install requests

COPY /ocr-text-recognition/requirements.txt copy_state/requirements.txt

RUN pip install -r copy_state/requirements.txt

# 이미지 내에서 명령어를 실행할(현 위치로 잡을) 디렉토리 설정

COPY aitest.py copy_state/aitest.py

# 컨테이너 실행시 실행할 명령어
CMD ["python3", "copy_state/aitest.py"]

# 이미지 생성 명령어 (현 파일과 같은 디렉토리에서)
# docker build -t {이미지명} .

# 컨테이너 생성 & 실행 명령어
#  docker run -it --name {컨테이너명} -v "$(pwd)/aitest:/volume_state" {이미지명}