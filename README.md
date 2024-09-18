
# FireCastNet

Uses Lightning CLI and supports multiple models. Configuration files are located in `configs/`.

## Classification

In order to run for classification:

```
 main.py fit --model FireCastNet --config config
```

Other models can be run using:

```
python main.py fit --model GRU --config configs/gru-config.yaml
python main.py fit --model ConvGRU --config configs/conv-gru-config.yaml
python main.py fit --model ConvLSTM --config configs/conv-lstm-config.yaml
python main.py fit --model UTAE --config configs/utae-config.yaml
```

## Regression

Adjust the configuration file and name it with a `regr` suffix. 

```
python main.py fit --model FireCastNet --config configs/config-regr.yaml
```

## Requirements 

You need to install pytorch, DGL and lightning. 

DGL version 2.0.0 from 
```
curl -LO https://data.dgl.ai/wheels/cu121/dgl-2.0.0%2Bcu121-cp310-cp310-manylinux1_x86_64.whl
```

### Download the data

Download the [SeasFire dataset](https://zenodo.org/record/8055879) from zenodo. Note it is 44GB. 

Unzip the dataset to a folder of your choice. Reference the dataset from the config file.

## Acknowledgements

This work is part of the SeasFire project, which deals with
”Earth System Deep Learning for Seasonal Fire Forecasting”
and is funded by the European Space Agency (ESA) in the
context of the ESA Future EO-1 Science for Society Call.

