# How to build the reasense demo

Follow the installation intrusctions in the order they are presented here. Some preliminary requirements before starting are:

1) Verify that you have conda installed, along with the associate env paths, in your machine
2) CUDA 12.8 or 12.6
3) C++ compiler that supports the above CUDA version

## Create conda enviroment

```bash
conda create --name <name>
conda activate <name>
pip install --upgrade pip
```

## Clone sam2 repo

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
conda activate <name>
python setup.py build
python setup.py install
pip install -e .
```

## Install realsense and torch for cuda 12.8 (see https://pytorch.org/ for cuda 12.6)

```bash
conda activate <name>
pip install pyrealsense2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Open an IDE with working directory at : ```\hope\addons\onxx\scripts```, connect your realsense camera and run ```demo_realsense_segment.py``` script from the conda environment you just created.

NOTE: Upon first run, you will have to manually install (i.e., ```pip install <pckg>```) some extra python packages that are not mentioned here such as opencv. We do not use ```requirements.txt``` on purpose.