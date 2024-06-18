import cv2
from paddleocr import PaddleOCR, draw_ocr
# import matplotlib.pyplot as plt
from PIL import Image



ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Use 'ch' for Chinese and 'en' for English

# Read the image
img_path = 'image.jpg'
img = cv2.imread(img_path)

# Perform OCR on the image
result = ocr.ocr(img_path, cls=True)

# Print OCR results
for line in result:
    print(line, "----")

# Extracting the detected text and boxes for visualization
boxes = [res[0] for res in result[0]]
txts = [res[1][0] for res in result[0]]
scores = [res[1][1] for res in result[0]]

print(scores, "---")
print( "Beam no", txts)

