import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow
from tensorflow import keras
import cv2
import json
import mediapipe as mp


class pipeline:

    def __init__(self,imgcls_path,source_path):

        #self.objdet_path = objdet_path # path to object detection algorithm
        self.imgcls_path = imgcls_path # path to image classification algorithm
        self.source_path = source_path # path to the source (video)


    # loading object detection model
    def load_objectdetection_model(self):
        mp_hands = mp.solutions.hands
        hand = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7)
        print("object detection model has been loaded successfully")
        return hand
    
    # loading image classificaiton model
    def load_imageclassification_model(self):
        model = keras.models.load_model(self.imgcls_path)
        print("image classification model has been loaded successfully")
        return model
    
    
    # process the video
    def process_video(self,objectdetmodel):
        video = cv2.VideoCapture(self.source_path)
        while True:
            ret,frame = video.read()
            if not ret:
                print("Ignoring empty frame.")
                continue
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = objectdetmodel.process(img_rgb)
            h, w, _ = frame.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Find bounding box from landmark coordinates
                    x_min = w
                    y_min = h
                    x_max = y_max = 0

                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)

                    # Draw bounding box
                    cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
            cv2.imshow("window",frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        video.release()

    # run
    def run(self,model1):
        self.process_video(objectdetmodel=model1)