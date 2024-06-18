import cv2
from paddleocr import PaddleOCR, draw_ocr
# import matplotlib.pyplot as plt
from PIL import Image


class ocr_recgn:
    def __init__(self):
        self.ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')
    
    def perform_ocr(frame):
        # img = cv2.imread(self.path)
        result = self.ocr_reader.ocr(frame, cls=True)
        
        highest_conf = 0
        beam_no_with_highest_conf = ""

        for line in result:
            for res in line:
                beam_no, conf = res[1]
                if conf > highest_conf:
                    highest_conf = conf
                    beam_no_with_highest_conf = beam_no

        print("Beam number with the highest confidence:", beam_no_with_highest_conf)
        print("Confidence:", highest_conf)

