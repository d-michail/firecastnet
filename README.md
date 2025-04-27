
# FireCastNet

Uses Lightning CLI and supports multiple models. Configuration files are located in `configs/`.

## Classification

In order to run for classification:

```bash
python main.py fit --config configs/config.yaml
```

Other models can be run using:

```bash
python main.py fit --config configs/gru-config.yaml
python main.py fit --config configs/conv-gru-config.yaml
python main.py fit --config configs/conv-lstm-config.yaml
python main.py fit --config configs/utae-config.yaml
```

## Regression

Adjust the configuration file and name it with a `regr` suffix.

```bash
python main.py fit --config configs/config-regr.yaml
```

## Requirements

You need to install pytorch, DGL and lightning. Start from DGL as this will dictate the rest of the dependencies.
Example with python 3.10, DGL 2.0.0, pytorch 2.7 and cuda 12.1.

```bash
python -m venv .venv 
source .venv/bin/activate
pip install https://data.dgl.ai/wheels/cu121/dgl-2.0.0%2Bcu121-cp310-cp310-manylinux1_x86_64.whl
pip install xarray xbatcher zarr
pip install pytorch-lightning lightning
pip install einops scikit-learn
pip install -U 'jsonargparse[signatures]>=4.27.7'
```

### Download the data

Download the [SeasFire dataset](https://zenodo.org/record/8055879) from zenodo. Note it is 44GB.

Unzip the dataset to a folder of your choice. Reference the dataset from the config file.

## Acknowledgements

This work is part of the SeasFire project, which deals with
”Earth System Deep Learning for Seasonal Fire Forecasting”
and is funded by the European Space Agency (ESA) in the
context of the ESA Future EO-1 Science for Society Call.
