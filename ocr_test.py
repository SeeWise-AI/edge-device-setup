import cv2
from paddleocr import PaddleOCR, draw_ocr
# import matplotlib.pyplot as plt
from PIL import Image

# Initialize the OCR model
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

# Visualize the results
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# im_show = draw_ocr(img, boxes, txts, scores)  # No need to specify font_path
# im_show = Image.fromarray(im_show)

cv2.imshow("test", img)
