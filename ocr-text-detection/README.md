# TextBoxes++ Tutorial
---

## 개요

본 페이지는 https://github.com/mvoelk/ssd_detectors 의 코드를 차용해 만든 [TextBoxes++](https://arxiv.org/pdf/1801.02765.pdf)논문 튜토리얼입니다. scripts 폴더 내의 notebook을 실행시키면 튜토리얼을 시작하실 수 있습니다.

## 환경 설정

튜토리얼을 진행하기 전에 버전 설정및 기타 라이브러리 설치를 위해 아래 단계를 수행해주세요. <br>
* 파이썬 버전 == 3.6
```python
# python 버전 확인하기
import platform
print(platform.python_version())
```

* 텐서플로우 버전 설정
```python
# Textboxes++은 tensorflow 2.0대 버전을 지원하지 않습니다.
$ pip install tensorflow==1.14 # GPU환경일 경우 tensorflow-gpu==1.14
```

* 영상 처리 라이브러리
```python
# opencv-python 설치
$ apt install libfontconfig1 libxrender1 libsm6
$ pip install opencv-python
```

* 기타 필수 라이브러리 설치
```python
$ pip install -r requirements.txt
```

## 데이터 셋

튜토리얼에 사용된 데이터 셋은 [Synthtext](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)데이터 셋으로, 7,266,866개의 단어들로 이루어진 858,750장의 합성 이미지 파일이 200개의 폴더에 나누어져 담겨있으며 Ground-truth annotation은 gt.mat파일에 담겨있습니다.<br><br>
데이터셋은 다운받으신 후 압축을 풀어 `data` 폴더에 넣어주세요.

## Expected results

본 튜토리얼을 마쳤을 때 얻을 수 있는 결과입니다.

<img src="https://i.imgur.com/YgA4EJe.png" width="800">