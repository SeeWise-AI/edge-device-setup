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
    parser.add_argument('-ocr','--ocr_model', type=str, nargs='+', required=False,
        help='OCR model needs to shared if you need to perform ocr')
    args = parser.parse_args()
    return args

def loop_and_detect(cam, trt_yolo, conf_th, vis, window_name, ocr_model = None):
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

    if ocr_model is not None:
        if ocr_model[0] == 'easyocr':
            print("Initializing easyocr....")
            ocr = easyocr.Reader(['en'], gpu=True)
        else:
            print("Initializing paddle ocr....")
            ocr = PaddleOCR(use_angle_cls=True, lang='en')

    while True:
        if cv2.getWindowProperty(window_name, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        
        boxes, confs, clss = trt_yolo.detect(img, conf_th)

        if ocr_model is not None:
            for box, cls_id in zip(boxes, clss):
                print(box, "----box---")
                print(cls_id, "------cls----")
                if cls_id == 1:  
                    x1, y1, x2, y2 = map(int, box)
                    cropped_img = img[y1:y2, x1:x2]
                    cv2.imshow("check", cropped_img)
                    if ocr_model[0] == 'easyocr':
                        result = ocr.readtext(cropped_img, allowlist='0123456789')
                        for out in easy_ocu_out:
                            if len(out) >= 2:
                                beam_val, beam_val_conf = out[1], out[2]
                        print(beam_val, "---")
                        # if result[0] is not None:
                        #     print("Checking wait ...")
                        #     text = " ".join([res[1] for res in result])
                        #     print(text, "---")
                        #     cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # else:
                        #     print("beam id not found")
                    else:
                        result = ocr.ocr(cropped_img)
                        print(result, "---")
                        if result[0] is not None:
                            text = " "
                            for line in result:
                                for word_info in line:
                                    if isinstance(word_info, list) and len(word_info) > 1:
                                        text += word_info[1][0] + " "
                            
                            text = text.strip()

                            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            print("Beam id is not found..")

                    

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
    if args.ocr_model:
        trt_yolo = TrtYOLO(args.model_path[cam_index], args.category_num, args.letter_box)
    else:
        trt_yolo = TrtYOLO(args.model_path[cam_index], args.category_num, args.letter_box)

    window_name = f"{WINDOW_NAME}_{cam_index}"
    open_window(window_name, f'Camera TensorRT YOLO Demo {cam_index}',
                cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, args.conf_thresh, vis=vis, window_name=window_name, ocr_model=args.ocr_model)

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