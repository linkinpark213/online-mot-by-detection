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

Note: This repository doesn't include any object detector implementations but provides interfaces for Detectron2, MMdetection, darknet and CenterNet.
You can install any of them in your environment.

Note: This project is developed on Ubuntu 18.04 with Python 3.7 and NVIDIA GTX 1080Ti.

## Get Started
### Clone the repo
```
git clone git@github.com:linkinpark213/online-mot-by-detection.git --recurse-submodules
```

### Installation
With all requirements installed, you only need to add the code directory to your PYTHONPATH. 
Edit the line below and run in your terminal:
```
export PYTHONPATH=/path/to/online-mot-by-detection:$PYTHONPATH
```

#### (Optional) If you use YOLO detector
You'll need to compile the `darknet` project in `third_party`. Please refer to [darknet readme](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-cmake).

#### (Optional) If you use CenterNet detector
You'll need to replace the DCNv2 in `third_party/CenterNet/src/lib/models/networks` with the new version with support for PyTorch 1.0+:
```
rm -rf third_party/CenterNet/src/lib/models/networks/DCNv2
mv third_party/DCNv2 third_party/CenterNet/src/lib/models/networks/DCNv2
cd third_party/CenterNet/src/lib/models/networks/DCNv2
./make.sh
```

### Download weights for components
The weights of CenterNet/DGNet/OpenReID that we used are uploaded [here](https://drive.google.com/drive/folders/1Awi_V6gSF6RSGuesMdzr0gSIcn7lR8-E?usp=sharing).
For pre-trained weights of MMDetection or Detectron2, please refer to their model zoos.

### Run the demo
A few example configs for building up online multi-object trackers are in the `configs` directory.

Here is a demo that runs a customized DeepSORT tracker with a MMDetection Faster R-CNN detector, a DG-Net re-ID encoder, a Kalman filter for target motion prediction and a cascaded bipartite matcher based on Hungarian algorithm.

MMDetection will automatically download the weight files needed but you'll have to download DG-Net weight files manually [here](https://drive.google.com/file/d/1L2jQ_TV5JmH-64JxruZW1beYzmvEV1J4/view?usp=sharing) and move the wetght file `id_00100000.pt` to `mot/encode/DGNet`.
```
python tools/demo.py configs/deepsort.py --demo_path /path/to/any/video
```
 