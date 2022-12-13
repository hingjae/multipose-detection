import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['violence']

#윈도우 사이즈
seq_length = 30

#녹화 초 (30초 동안 학습동작을 녹화) 조정
secs_for_action = 30

# 미디어파이프 init
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#포즈 모델을 load함 (몸의 33개의 랜드마크를 추출)
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1 # 모델의 복잡도 (디폴트는 1인데 좀더 복잡한 모델을 사용해서 정확도를 향상 시킴)
    )

#웹캠
cap = cv2.VideoCapture(0)
created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions): #첫번째 ~ 세번째 actions순으로 녹화

        data = []
        
        ret, img = cap.read()

        cv2.imshow('img', img)
        cv2.waitKey(3000) #3초 대기

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #오픈cv에서는 bgr사용 mp는 rgb사용
            result = pose.process(img) # rgb로 변환한 이미지를 mp에 넣어줌
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.pose_landmarks is not None:
                for res in result.pose_landmarks:
                    joint = np.zeros(33, 4)
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    v1 = joint[[], 3]
                    v2 = joint