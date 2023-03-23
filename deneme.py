
import torch
import cv2
import numpy as np
from ssl import _create_unverified_context
from time import time
from ultralytics import YOLO


from tracker import Tracker as Xalil_Tracker

def main():
    start_time = time()

    _create_default_https_context = _create_unverified_context

    ################TRACKER CONF ####################
    tracker = Xalil_Tracker()

    source_video_path = "demovideo.webm"
    #video_saving_path = "output/out1.mp4"

    ################# MODEL CONF ##################
    model = YOLO("yolov8s.pt")  # load an official model


    video_cap=cv2.VideoCapture(source_video_path)

    #fps = video_cap.get(cv2.CAP_PROP_FPS)
    width, height = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #desired_fps = 35
    #result = cv2.VideoWriter(video_saving_path, cv2.VideoWriter_fourcc(*'mp4v') ,desired_fps, (width,height))


    count=0
    while video_cap.isOpened():
        count += 1    
        ret,frame=video_cap.read()
        if not ret:
            break
        #ADJUST FPS
        if count % 1 != 0:
            continue

        if count >100:
            break

        results = model.predict(source=frame,classes=2,device="mps") #same as model(frame) but in yolov8 style 

        #for result in results: #that will only return sinle value so even results[0] is same as this code 
        detections = []
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = box
            x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id) 

            detections.append([x1,y1,x2,y2,conf,class_id])

        tracker.update(frame,detections)

        for track in tracker.tracks:
            bbox = track.bbox
            track_id = track.track_id
        
        print(track_id)



        cv2.imshow("ROI",frame)
        print(f"frame {count} writing")
        if cv2.waitKey(10) == ord('q'):
            break


    video_cap.release()
    #result.release()
    cv2.destroyAllWindows()
    print("process done")        
    print("Execution time:", round(time() - start_time,2), "seconds")


if __name__ == "__main__":
    main()
