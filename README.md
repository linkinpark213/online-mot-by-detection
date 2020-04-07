# Online MOT-by-Detection
A simple online multi-object tracking toolbox following the tracking-by-detection paradigm.


This repository is still under construction.

## Requirements
* Python 3
* numpy
* scipy
* argparse
* opencv-python
* PyTorch 1.0+ (1.3+ if Detectron2 is used)
* [Detectron2](https://github.com/facebookresearch/detectron2) (optional)
* [mmdetection](https://github.com/open-mmlab/mmdetection) (optional)
* [CenterNet](https://github.com/xingyizhou/CenterNet) (optional)
* [PySOT](https://github.com/STVIR/pysot) (optional)

Note: This repository doesn't include any object detector implementations but provides interfaces for Detectron2 and mmdetection.
You can install either or both of them in your environment.

Note: This project is developed on Ubuntu 18.04 with Python 3.7 and NVIDIA GTX 1080Ti. 
However, a Linux environment or GPU support is not indispensable.

## Get Started
### Installation
With all requirements installed, you only need to add the code directory to your PYTHONPATH. 
Edit the line below and run in your terminal:
```
export PYTHONPATH=/path/to/online-mot-by-detection:$PYTHONPATH
```
### Run the demo
A few examples for building up online multi-object trackers are in the `examples` directory.

Here is a demo that runs a customized DeepSORT tracker with a MMDetection Faster R-CNN detector, a DG-Net re-ID encoder, a Kalman filter for target motion prediction and a cascaded bipartite matcher based on Hungarian algorithm.

MMDetection will automatically download the weight files needed but you'll have to download DG-Net weight files manually [here](https://drive.google.com/file/d/1L2jQ_TV5JmH-64JxruZW1beYzmvEV1J4/view?usp=sharing) and move the wetght file `id_00100000.pt` to `mot/encode/DGNet`.
```
python tools/demo.py configs/deepsort.py --demo_path /path/to/any/video
```

## TODO List
* Add affinity metric and motion predictor using siamese network.
* Add offline post-tracking features.
