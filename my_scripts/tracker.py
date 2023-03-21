from ultralytics import YOLO


#from trackers.bytetrack import byte_tracker

import torch
import cv2
import numpy as np
from ssl import _create_unverified_context
from time import time
from ultralytics import YOLO

def main():
    start_time = time()
    _create_default_https_context = _create_unverified_context

    ################# PATH ##################
    source_video_path = "demovideo.webm"
    #video_saving_path = "output/out1.mp4"

    ################# MODEL ##################
    model = YOLO("yolov8s.pt")  # load an official model

    ################# VIDEO SOURCE ##################
    video_cap=cv2.VideoCapture(source_video_path)
    #fps = video_cap.get(cv2.CAP_PROP_FPS)
    width, height = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #desired_fps = 35
    #result = cv2.VideoWriter(video_saving_path, cv2.VideoWriter_fourcc(*'mp4v') ,desired_fps, (width,height))

    ################# TRACKER ##################
    #tracker = byte_tracker.BYTETracker()

    ################# MAIN LOOP ##################
    count=0
    while video_cap.isOpened():
        count += 1    
        ret,frame=video_cap.read()
        if not ret:
            break
        #ADJUST FPS
        if count % 1 != 0:
            continue
        if count >1:
            break

        ################# OBJECT DETECTION ##################
    
        ################# OBJECT TRACKING ##################
        """bbox_list = []
        for result in results:
            result = result.cpu().numpy()
            for box in result.boxes.xyxyn:
                x1,y1,x2,y2 = box
                x1,y1,x2,y2 = int(x1*width),int(y1*height),int(x2*width), int(y2*height)
                bbox_list.append([x1, y1, x2 - x1, y2 - y1])"""
        
        results = model.predict(source=frame,classes=2,device="mps")
        
        for i, det in enumerate(results):  # detections per image
            print("deto")
            print(det.cpu()[0])

                
                #print()
                #x1,y1,x2,y2 = box
                #tracker.update(listy,2)
            #print(listy[:, 0:4])
        ################# DRAW TRACKED OBJECTS ##################
        """tracked_objects = tracker.get_objects()
        for obj_id, bbox in tracked_objects.items():
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Object {obj_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
"""








        ################# DISPLAY FRAME ##################
        cv2.imshow("ROI",frame)
        print(f"frame {count} writing")
        if cv2.waitKey(10) == ord('q'):
            break


    video_cap.release()
    #result.release()
    cv2.destroyAllWindows()
    print("process done")        
    print("Execution time:", time() - start_time, "seconds")


if __name__ == "__main__":
    main()