import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow
from tensorflow import keras
import cv2
import json
import mediapipe as mp
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore
from tensorflow.keras.models import load_model # type: ignore


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
        print(tensorflow.__version__)
        return hand
    
    # loading image classificaiton model
    def load_imageclassification_model(self):
        model = load_model(self.imgcls_path,
                            custom_objects={'preprocess_input': preprocess_input}
                        )
        print("image classification model has been loaded successfully")
        return model
    
    def load_database(self):
        with open("C:/Users/dell/OneDrive/Desktop/ASL_PROJECT/db.json","r") as file:
            db = json.load(file)
            file.close()
        db = {int(key):value for key,value in db.items()}
        return db
    
    def process_video(self,objectdetmodel,imgclsmodel,db):
        video = cv2.VideoCapture(self.source_path)
        video.set(cv2.CAP_PROP_FPS, 15)
        while True:
            ret,frame = video.read()
            if not ret:
                print("Ignoring empty frame.")
                continue
            resize_frame = cv2.resize(frame,dsize=None,fx=0.5,fy=0.5)
            #gray = cv2.cvtColor(resize_frame,cv2.COLOR_BGR2GRAY)
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

                    hand_crop = frame[y_min:y_max, x_min:x_max]
                    resized = cv2.resize(hand_crop, (224, 224))
                    normalized = resized / 255.0
                    input_img = np.expand_dims(normalized, axis=0)

                    prediction = imgclsmodel.predict(input_img)
                    pred_index = np.argmax(prediction)
                    pred_char = db.get(pred_index)
                    print(pred_index,pred_char)
                    # Draw bounding box
                    cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
                    cv2.putText(frame, f"{pred_char}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            cv2.imshow("window",frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        video.release()
    
    
    # run
    def run(self,model1,model2,db):
        self.process_video(model1,model2,db)