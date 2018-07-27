# Tau Identification
Machine learning algorithms for tau particle identification.

## Setup
Add directory to python path:
```bash
cd ~/TauID
source setup.sh
```
## Convert data
To convert ROOT file to numpy structured array:
```bash
cd ~/TauID/convert
python convert.py /path/to/file.root -n [num_events]
```
## Training
To run a simple training:
```bash
cd ~/TauID/train
python train.py -s /path/to/signal_file.z \
-b /path/to/background_file.z \
-c config_onelayer.yaml
```

and evaluate the training:
```bash
python evaluate.py -l saved-data/onelayer_base.hdf5 \
-d saved-models/onelayer_base/ \
-c config_onelayer.yml \
```
