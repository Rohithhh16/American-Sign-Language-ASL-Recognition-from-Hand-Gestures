import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow import keras
import cv2
import mediapipe as mp


class pipeline:

    def __init__(self,objdet_path,imgcls_path,source_path):

        self.objdet_path = objdet_path # path to object detection algorithm
        self.imgcls_path = imgcls_path # path to image classification algorithm
        self.source_path = source_path # path to the source (video)


    # loading object detection model
    def load_objectdetection_model(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7)
        print("object detection model has been loaded successfully")
        return hands
    
    # loading image classificaiton model
    def load_imageclassification_model(self):
        model = keras.models.load_model(self.imgcls_path)
        print("image classification model has been loaded successfully")
        return model
    
    
    # process the video
    def process_video(self):
        video = cv2.VideoCapture(self.source_path)
        while True:
            ret,frame = video.read()
            if ret:
                cv2.imshow("window",frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            else:
                break
        cv2.destroyAllWindows()
        video.release()

    # run
    def run(self):
        self.process_video()