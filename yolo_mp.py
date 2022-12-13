import cv2
import numpy as np
import mediapipe as mp
from numba import jit, cuda 
from keyXYZ import keyXYZ
import joblib
# import keyXYZ
import time # -- 프레임 계산을 위해 사용

pose_knn = joblib.load('Model/PoseKeypoint.joblib')
res_point = []
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

vedio_path = './Fall_Trim.mp4' #-- 사용할 영상 경로
min_confidence = 0.5


def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    global res_point
    label = str(classes[class_id])

    if label == 'person':
        # cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)
        # cv2.putText(img, label, (x-10, y-10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if x > 0 and y > 0:
            crop_img = img[y:y_plus_h, x:x_plus_w]

            imgRGB = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                for index, landmarks in enumerate(results.pose_landmarks.landmark):
                    res_point.append(landmarks.x)
                    res_point.append(landmarks.y)
                    res_point.append(landmarks.z)
                shape1 = int(len(res_point) / len(keyXYZ))
                res_point = np.array(res_point).reshape(shape1, len(keyXYZ))
                pred = pose_knn.predict(res_point)
                res_point = []
                print(pred)
                mp_drawing.draw_landmarks(crop_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # cv2.imshow('Video', img)

# @jit(target_backend='cuda')
def detectAndDisplay(frame):
    start_time = time.time()
    img = cv2.resize(frame, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape
    #cv2.imshow("Original Image", img)

    #-- 창 크기 설정
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    #-- 탐지한 객체의 클래스 예측 
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #-- 원하는 class id 입력 / coco.names의 id에서 -1 할 것 
            if class_id == 0 and confidence > min_confidence:
                #-- 탐지한 객체 박싱
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}".format(classes[class_ids[i]], confidences[i]*100)
            # print(i, label)
            # color = colors[i] #-- 경계 상자 컬러 설정 / 단일 생상 사용시 (255,255,255)사용(B,G,R)
            # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
            draw_prediction(img, class_ids[i], round(x), round(y), round(x+w), round(y+h))

    # end_time = time.time()
    # process_time = end_time - start_time
    # print("=== A frame took {:.3f} seconds".format(process_time))
    cv2.imshow("YOLO Video", img)
#-- yolo 포맷 및 클래스명 불러오기
model_file = './yolov3.weights' #-- 본인 개발 환경에 맞게 변경할 것
config_file = './yolov3.cfg' #-- 본인 개발 환경에 맞게 변경할 것
net = cv2.dnn.readNet(model_file, config_file)

# GPU 사용
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#-- 클래스(names파일) 오픈 / 본인 개발 환경에 맞게 변경할 것
classes = []
with open("./coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#-- 비디오 활성화
cap = cv2.VideoCapture(vedio_path) #-- 웹캠 사용시 vedio_path를 0 으로 변경
# cap = cv2.VideoCapture(0) #-- 웹캠 사용시 vedio_path를 0 으로 변경
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    #-- q 입력시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()