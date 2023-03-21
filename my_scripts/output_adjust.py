import argparse
import cv2
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from ssl import _create_unverified_context
from time import time
from ultralytics import YOLO


import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker


@torch.no_grad()
def main(
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,

        device='mps',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        line_thickness=2,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
):




########################################### HALO HALO #######################
    source_video_path = "demovideo.webm"
    xalil_model = YOLO("yolov8s.pt")  # load an official model

    start_time = time()
    _create_default_https_context = _create_unverified_context

    video_cap=cv2.VideoCapture(source_video_path)
    width, height = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    counter = -1

    while video_cap.isOpened():
        count += 1    
        ret,frame=video_cap.read()
        if not ret:
            break
        #ADJUST FPS
        if count % 1 != 0:
            continue
        if count >50:
            break

        bs = 1

        # Create as many strong sort instances as there are video sources
        tracker_list = []
        for i in range(bs):
            tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
            tracker_list.append(tracker, )
        outputs = [None] * bs

        detections = xalil_model.predict(source=frame,classes=2,device="mps")

        for i, det in enumerate(detections):  # detections per image
            seen += 1
            annotator = Annotator(im0, line_width=line_thickness, example=str("srede"))

            if det is not None and len(det):

                outputs[i] = tracker_list[i].update(det.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output) in enumerate(outputs[i]):
                        
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]

                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        color = colors(c, True)
                        annotator.box_label(bbox, "car", color=color)

            else:
                pass
                #tracker_list[i].tracker.pred_n_update_all_tracks()
                
            # Stream results
            im0 = annotator.result()

            cv2.imshow(str(p), im0)
            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                exit()
            


if __name__ == "__main__":
    main()