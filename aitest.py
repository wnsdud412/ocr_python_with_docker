import base64
import requests
import cv2
import json

path = '/volume_state/개구리.png'
def ocr_rec(path=path):
    img=cv2.imread(path)
    png_img = cv2.imencode('.png', img)
    b64_string = base64.b64encode(png_img[1]).decode('utf-8')
    url = 'http://172.17.0.1:8501/v1/models/ocr:predict' 
    payload = {"instances": [{"image": {"b64": b64_string }}]}
    res = requests.post(url, data=json.dumps(payload))
    print(res.text)
    return res.text

ocr_rec()