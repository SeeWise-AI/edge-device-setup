﻿# edge-device-setup



this demo showcase how to convert pre-trained yolov3 and yolov4 models through ONNX to TensorRT engines. The code for these 2 demos has gone through some significant changes. More specifically, I have recently updated the implementation with a "yolo_layer" plugin to speed up inference time of the yolov3/yolov4 models.

My current "yolo_layer" plugin implementation is based on TensorRT's IPluginV2IOExt. It only works for TensorRT 6+. I'm thinking about updating the code to support TensorRT 5 if I have time late on.

I developed my "yolo_layer" plugin by referencing similar plugin code by wang-xinyu and dongfangduoshou123. So big thanks to both of them.

Assuming this repository has been cloned at "${HOME}/project/tensorrt_demos", follow these steps:

Install "pycuda".

$ cd ${HOME}/project/tensorrt_demos/yolo
$ ./install_pycuda.sh
Install version "1.9.0" of python3 "onnx" module. Note that the "onnx" module would depend on "protobuf" as stated in the Prerequisite section.

$ sudo pip3 install onnx==1.9.0
Go to the "plugins/" subdirectory and build the "yolo_layer" plugin. When done, a "libyolo_layer.so" would be generated.

$ cd ${HOME}/project/tensorrt_demos/plugins
$ make
Download the pre-trained yolov3/yolov4 COCO models and convert the targeted model to ONNX and then to TensorRT engine. I use "yolov4-416" as example below. (Supported models: "yolov3-tiny-288", "yolov3-tiny-416", "yolov3-288", "yolov3-416", "yolov3-608", "yolov3-spp-288", "yolov3-spp-416", "yolov3-spp-608", "yolov4-tiny-288", "yolov4-tiny-416", "yolov4-288", "yolov4-416", "yolov4-608", "yolov4-csp-256", "yolov4-csp-512", "yolov4x-mish-320", "yolov4x-mish-640", and custom models such as "yolov4-416x256".)

$ cd ${HOME}/project/tensorrt_demos/yolo
$ ./download_yolo.sh
$ python3 yolo_to_onnx.py -m yolov4-416
$ python3 onnx_to_tensorrt.py -m yolov4-416
The last step ("onnx_to_tensorrt.py") takes a little bit more than half an hour to complete on my Jetson Nano DevKit. When that is done, the optimized TensorRT engine would be saved as "yolov4-416.trt".

In case "onnx_to_tensorrt.py" fails (process "Killed" by Linux kernel), it could likely be that the Jetson platform runs out of memory during conversion of the TensorRT engine. This problem might be solved by adding a larger swap file to the system. Reference: Process killed in onnx_to_tensorrt.py Demo#5.

Test the TensorRT "yolov4-416" engine with the "dog.jpg" image.

$ cd ${HOME}/project/tensorrt_demos
$ wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg -O ${HOME}/Pictures/dog.jpg
$ python3 trt_yolo.py --image ${HOME}/Pictures/dog.jpg \
                      -m yolov4-416
This is a screenshot of the demo against JetPack-4.4, i.e. TensorRT 7.

"yolov4-416" detection result on dog.jpg

The "trt_yolo.py" demo program could also take various image inputs. Refer to step 5 in Demo #1 again.

For example, I tested my own custom trained "yolov4-crowdhuman-416x416" TensorRT engine with the "Avengers: Infinity War" movie trailer:

Testing with the Avengers: Infinity War trailer

(Optional) Test other models than "yolov4-416".
