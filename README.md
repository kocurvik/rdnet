# Evaluation of radial distortion solvers

This repo contains code for paper "Are Minimal Radial Distortion Solvers Necessary for Relative Pose Estimation?" (under review in IJCV) which is an extension of our previous work at ECCVW with the same title (arxiv: http://arxiv.org/abs/2410.05984)

## Installation

Create an environment with pytorch and packages from `requirements.txt`.

Install [PoseLib fork with robust radial distortion estimators](https://github.com/kocurvik/PoseLib/tree/rdnet) into the environment:
```shell
git clone https://github.com/kocurvik/PoseLib
git cd PoseLib
git checkout rdnet
pip install .
```

Before running the python scripts make sure that the repo is in your python path (e.g. `export PYTHONPATH=/path/to/repo/rdnet`)

## Running experiments

### Datasets
We use five datasets:
* Rotunda and Cathedral - [RD Benchmark](https://drive.google.com/drive/folders/1XmCglJMU1s6jDq5KcHmo5w2fH6ZTlj6y?usp=drive_link)
* Phototourism - download from the [IMC2020 challenge website](https://www.cs.ubc.ca/~kmyi/imw2020/data.html)
* ETH3D - download the multiview undistorted train data from the [dataset website](https://www.eth3d.net/datasets#high-res-multi-view-training-data).
* PragueParks - TBA
* EuRoC MAV - TBA

You can download the all used matches and GeoCalib Predictions at: TBA.

To prepare matches for Rotunda, Cathedral datasets you can run:
```shell
# generates all matches
python prepare_bundler.py -f superpoint /path/to/stored_matches/rotunda_new /path/to/dataset/rotunda_new
# generates only matches with the same camera in both views
python prepare_bundler.py -f superpoint -e /path/to/stored_matches/rotunda_new /path/to/dataset/rotunda_new

python prepare_bundler.py -n 10000 -f superpoint /path/to/stored_matches/cathedral /path/to/dataset/cathedral
python prepare_bundler.py -n 10000 -f superpoint -e /path/to/stored_matches/cathedral /path/to/dataset/cathedral
```

### Evaluation scripts

To run the experiments you should modify `experiments.sh` as described in the script comments and run it.

### Tables and figures

Tables and figures are generated using `utils/tables.py`, and `utils/vis.py`.

## Citation
TBA
