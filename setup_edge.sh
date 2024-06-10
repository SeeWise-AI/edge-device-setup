#!/bin/bash

# # Update package list and install python3-pip
# sudo apt update
# sudo apt install -y python3-pip

# # Install jetson-stats
# sudo pip3 install jetson-stats

# # Install pycuda (user installation)
# pip3 install pycuda 

# # Install ONNX
# sudo pip3 install onnx

# # Change directory to plugin and compile
# cd plugins || { echo "Directory 'plugins' not found"; exit 1; }
# make

# # Download YOLOv3-tiny config and weights
# cd ..
# echo "downloading yolo cfg...."
# wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny.cfg -q --show-progress --no-clobber

# echo "downloading yolo weights...."
# wget https://pjreddie.com/media/files/yolov3-tiny.weights -q --show-progress --no-clobber

# Create yolov3-tiny-288.cfg and yolov3-tiny-288.weights
echo
echo "Creating yolov3-tiny-288.cfg and yolov3-tiny-288.weights"
sed -e '8s/width=416/width=288/' -e '9s/height=416/height=288/' yolov3-tiny.cfg > yolov3-tiny-288.cfg
echo >> yolov3-tiny-288.cfg
ln -sf yolov3-tiny.weights yolov3-tiny-288.weights

# Create yolov3-tiny-416.cfg and yolov3-tiny-416.weights
echo "Creating yolov3-tiny-416.cfg and yolov3-tiny-416.weights"
cp yolov3-tiny.cfg yolov3-tiny-416.cfg
echo >> yolov3-tiny-416.cfg
ln -sf yolov3-tiny.weights yolov3-tiny-416.weights

# Convert YOLOv3-tiny models to ONNX and TensorRT
python3 yolo_to_onnx.py -m yolov3-tiny-416
python3 onnx_to_tensorrt.py -m yolov3-tiny-416

# Download test videos
echo "Downloading test videos"
wget -O test.mp4 https://www.pexels.com/download/video/855564/ -q --show-progress --no-clobber
wget -O test_1.mp4 https://www.pexels.com/download/video/853889/ -q --show-progress --no-clobber
wget -O test_2.mp4 https://www.pexels.com/download/video/1249406/ -q --show-progress --no-clobber
wget -O test_3.mp4 https://www.pexels.com/download/video/855565/ -q --show-progress --no-clobber
wget -O test_4.mp4 https://www.pexels.com/download/video/2273136/ -q --show-progress --no-clobber
wget -O test_5.mp4 https://www.pexels.com/download/video/17468319/ -q --show-progress --no-clobber

echo "Setup completed successfully!"
