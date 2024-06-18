"""
trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""

import os
import time
import argparse
import multiprocessing as mp

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

# from utils.yolo_classes import get_cls_dict
from utils.yolo_lnt_classes import get_cls_dict
from utils.camera_multi import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from paddleocr import PaddleOCR
from ocr_detection import ocr_recgn
import easyocr

WINDOW_NAME = 'TrtYOLODemo'

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=2,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-mt', '--multi_trial', type=float, default=1,  # 1 means false 0 means true
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=False,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument(
        '-n', '--num_cams', type=int, default=1,
        help='number of camera streams to process simultaneously')
    parser.add_argument(
        '-p', '--camera_paths', type=str, nargs='+', required=True,
        help='list of camera paths')
    parser.add_argument('-mp','--model_path', type=str, nargs='+', required=True,
        help='list of model path and should be came in the order of camera')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis, window_name, ocr_model = None, frame=None):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      window_name: name of the display window.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()

    while True:
        if cv2.getWindowProperty(window_name, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        if frame is not None and frame.size > 0:
            for box, cls_id in zip(boxes, clss):
                if cls_id == 1:  
                    x1, y1, x2, y2 = map(int, box)
                    dy = int((0.65*y2-y1))
                    dx = int(0.65*(x2-x1))

                    H, W = safe_frame.shape[:2]
                    beam_crop = safe_frame.copy()[max(0, y1-dy):min(y2+dy, H), max(0, x1-dx):min(x2+dx, W)]
                    beam = ocr_recgn.perform_ocr(beam_crop)
                    print(beam, "---")

        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        cv2.imshow(window_name, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(window_name, full_scrn)

def process_camera(args, cam_index):
    """Process a single camera stream."""
    cam = Camera(args, cam_index)

    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera %d!' % cam_index)

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model_path[cam_index], args.category_num, args.letter_box)

    window_name = f"{WINDOW_NAME}_{cam_index}"
    open_window(window_name, f'Camera TensorRT YOLO Demo {cam_index}',
                cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, args.conf_thresh, vis=vis, window_name=window_name, frame=cam.img_handle)

    cam.release()
    cv2.destroyAllWindows()

def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    # if not os.path.isfile('yolo/%s.trt' % args.model_path):
    #     raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)
    if args.num_cams < 1:
        raise SystemExit('ERROR: num_cams must be at least 1!')
    if len(args.camera_paths) != args.num_cams:
        raise SystemExit('ERROR: Number of camera paths must match num_cams!')

    processes = []
    for i in range(args.num_cams):
        p = mp.Process(target=process_camera, args=(args, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()