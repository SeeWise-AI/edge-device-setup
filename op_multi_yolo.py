import os
import time
import argparse
import logging
import multiprocessing as mp

import cv2
from utils.yolo_lnt_classes import get_cls_dict
from utils.camera_multi import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from paddleocr import PaddleOCR
from ocr_detection import ocr_recgn
from PIL import Image
import numpy as np

from openvino.runtime import Core

WINDOW_NAME = 'YOLODemo'

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with YOLO model')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=2,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='path to YOLO model XML file')
    parser.add_argument(
        '-w', '--weights', type=str, required=True,
        help='path to YOLO model BIN file')
    parser.add_argument(
        '-cl', '--classes', type=str, required=True,
        help='path to file containing class names')
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument(
        '-n', '--num_cams', type=int, default=1,
        help='number of camera streams to process simultaneously')
    parser.add_argument(
        '-p', '--camera_paths', type=str, nargs='+', required=True,
        help='list of camera paths')
    args = parser.parse_args()
    return args

def load_model(model_xml, model_bin):
    """Load YOLO model using OpenVINO."""
    ie = Core()
    net = ie.read_model(model=model_xml, weights=model_bin)
    compiled_model = ie.compile_model(model=net, device_name="CPU")
    return compiled_model

def get_output_layers(compiled_model):
    """Get the output layer names."""
    return compiled_model.outputs

def detect_objects(compiled_model, output_layers, img, conf_threshold):
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    outputs = compiled_model([blob])[output_layers[0]]

    boxes = []
    confidences = []
    class_ids = []

    h, w = img.shape[:2]
    for detection in outputs:
        for obj in detection:
            scores = obj[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]

            if confidence > conf_threshold:
                box = obj[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
    if len(indices) > 0:
        indices = indices.flatten()

    filtered_boxes = [boxes[i] for i in indices]
    filtered_confs = [confidences[i] for i in indices]
    filtered_class_ids = [class_ids[i] for i in indices]

    return filtered_boxes, filtered_confs, filtered_class_ids

def add_padding(box, img_shape, padding_factor=0.1):
    """Add padding to the bounding box."""
    x, y, w, h = box
    img_h, img_w = img_shape[:2]
    
    # Compute padding
    pad_x = int(w * padding_factor)
    pad_y = int(h * padding_factor)
    
    # Adjust coordinates with padding
    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, img_w)
    y2 = min(y + h + pad_y, img_h)
    
    return [x1, y1, x2 - x1, y2 - y1]

def loop_and_detect(cam, compiled_model, output_layers, classes, conf_th, window_name, ocr_model=None):
    """Continuously capture images from camera and do object detection."""
    full_scrn = False
    fps = 0.0
    tic = time.time()

    ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')
    logging.getLogger("ppocr").setLevel(logging.CRITICAL)

    while True:
        if cv2.getWindowProperty(window_name, 0) < 0:
            break
        img = cam.read()
        
        boxes, confs, clss = detect_objects(compiled_model, output_layers, img, conf_th)
       
        for box, conf, cls_id in zip(boxes, confs, clss):
            x, y, w, h = box
            label = str(classes[cls_id])
            if cls_id == 1:  
                padded_box = add_padding(box, img.shape, padding_factor=0.1)  # Adjust padding factor as needed
                x1, y1, w, h = padded_box
                cropped_img = img[y1:y1+h, x1:x1+w]
                beam = ocr_recgn(ocr_reader)
                check = beam.perform_ocr(cropped_img)
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)
        tic = toc
        img = cv2.putText(img, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(window_name, img)
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
    
    compiled_model = load_model(args.model, args.weights)
    output_layers = get_output_layers(compiled_model)
    
    window_name = f"{WINDOW_NAME}_{cam_index}"
    open_window(window_name, f'Camera YOLO Demo {cam_index}', cam.img_width, cam.img_height)
    
    classes = ['job', 'beam']
    
    loop_and_detect(cam, compiled_model, output_layers, classes, args.conf_thresh, window_name=window_name)

    cam.release()
    cv2.destroyAllWindows()

def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
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
