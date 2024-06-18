import cv2
from paddleocr import PaddleOCR, draw_ocr
# import matplotlib.pyplot as plt
from PIL import Image


class ocr_recgn:
    def __init__(self, ocr):
        self.ocr_reader = ocr
    
    def perform_ocr(self, frame):
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
        return beam_no_with_highest_conf

