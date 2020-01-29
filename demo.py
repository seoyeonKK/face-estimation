from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from emotion_api import get_emotion
import tensorflow as tf
from socketIO_client_nexus import SocketIO, LoggingNamespace
import os
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

html = urlopen("https://weather.naver.com/rgn/townWetr.nhn?naverRgnCd=09320105")
bsObject = BeautifulSoup(html, "html.parser")

from time import sleep

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

pretrained_model = "https://s3.ap-northeast-2.amazonaws.com/sopt-seminar/weights.78-3.51.hdf5"
modhash = "306e44200d3f632a5dccac153c2966f2"
font = cv2.FONT_HERSHEY_SIMPLEX
imgNum = 0
flag = False

def on_finish(*args):
    print("on finish")
    global flag
    flag = True

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", type=str, default="ResNet50",
                        help="model name: 'ResNet50' or 'InceptionResNetV2'")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. age_only_weights.029-4.027-5.250.hdf5)")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()

def yield_images():
    # detectedFace 코드
    face_cascade = cv2.CascadeClassifier("haarcascade_frontface.xml")
    cnt = 0

    # 웹캠 활성화시키는 코드
    try:
        cap = cv2.VideoCapture(0)
    except:
        print('카메라 로딩 실패')
        return
    while True:
        ret, img = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 2, 0, (30, 30))
        faces = list(faces)
        if len(faces) >= 1:
            _faces = np.array(faces)
            findFaces = list(_faces[:, 2])
            maxIdx = findFaces.index(max(findFaces))
            x = _faces[:][maxIdx][0]
            y = _faces[:][maxIdx][1]
            w = _faces[:][maxIdx][2]
            h = _faces[:][maxIdx][3]

            # 얼굴을 인식하는 사각형에 대한 소스, 텍스트 소스
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3, 4, 0)
            cv2.putText(img, 'Detected Face', (x - 5, y - 5), font, 0.9, (255, 255, 0), 2)
            cnt += 1

            if cnt >= 15 and h > 100 and w > 100:
                global imgNum
                r = max(w, h) / 2
                centerx = x + w / 2
                centery = y + h / 2
                nx = int(centerx - r)
                ny = int(centery - r)
                nr = int(r * 2)
                cropped = img[ny:ny + nr, nx:nx + nr]
                # 이미지를 저장
                cv2.imwrite("thumbnail" + "0" + ".jpg", cropped)
                # emotion = get_emotion(imgNum)#emotion_api.py에서 가져온다.
                cnt = 0
                # 영상을 출력하는 소스
                cv2.imshow('frame', img)
                key = cv2.waitKey(30)
                if key == 27:  # ESC
                    break
                yield cropped

def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)
        print(image_path)
        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))

def main():
        socket = SocketIO('localhost', 3002, LoggingNamespace)
        print("connect")

        args = get_args()
        depth = args.depth
        k = args.width
        weight_file = args.weight_file
        margin = args.margin
        image_dir = args.image_dir
        if not weight_file:
            weight_file = get_file("checkpoints/weights.78-3.51.hdf5", pretrained_model,
                                   cache_subdir="pretrained_models",
                                   file_hash=modhash, cache_dir=Path(__file__).resolve().parent)
        # for face detection
        detector = dlib.get_frontal_face_detector()

        # age and gender
        # load model and weights
        img_size = 64
        model = WideResNet(img_size, depth=depth, k=k)()
        model.load_weights(weight_file)
        age_list = []

        image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()
        for img in image_generator:
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            # detect faces using dlib detector
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), img_size, img_size, 3))

            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2 )
                    faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

                # predict ages and genders of the detected faces
                results = model.predict(faces)
                predicted_genders = results[0]
                emotion = get_emotion(imgNum)
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()
                print(int(predicted_ages))
                print(predicted_genders)
                print(emotion)
                print()
                file = './thumbnail0.jpg'
                if os.path.isfile(file):
                    os.remove(file)
                age_list.append(int(predicted_ages))
                if len(age_list) == 2:
                    predicted_ages_final = (age_list[0] + age_list[1])/2
                    print(int(predicted_ages_final))
                    age_list = []
                    crawling = bsObject.body.find_all("em")[2].get_text()
                    crawling_num = re.findall("\d+", crawling)
                    crawling_dust = bsObject.body.find_all("strong")[3].get_text()
                    crawling_text = bsObject.body.find_all("em")[3].get_text()

                    for i, d in enumerate(detected):
                        label = "{},{},{},{},{},{}".format(int(predicted_ages_final),
                                                    "f" if predicted_genders[i][0] < 0.6 else "m",
                                                    "neutral" if emotion is None else emotion[2][0], crawling_num[0], crawling_dust, crawling_text)
                        listA = label.split(",")
                        print(listA)
                        socket.emit('client1', listA)

                        while True:
                            # sleep(0.5)

                            # Listen
                            socket.on('finish', on_finish)
                            socket.wait(seconds=1)

                            global flag
                            if flag is True:
                                flag = False
                                break

            # 웹캠 실행 시
            key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

            if key == 27:  # ESC
                break

if __name__ == '__main__':
    main()
