# ObliqueTrees

## Scope
This repo is a skeleton wrapper for running [yggdrasil decision forests](https://ydf.readthedocs.io/en/latest/) (YDF) models with a minimal working example of training, testing, evaluating, and tuning a model

## Models
The model.py creates the model classifier class with XGboost and two types of YDF models, axis aligned and oblique splitting. These wrappers can be used to:
- train a model: train a classifier on X_train and y_train
- test a model: create predictions for X_test
- evaluate a model: determine the accuracy of a model vs y_test and produce a ROC plot but overriding the evaluate function can allow any evaluation of a model
- tune a model: run a hyperparameter scan across the X_train dataset, useful for oblqiue models but time consuming

## Datasets
The dataset.py is a wrapper for loading and splitting datasets, currently works for openML and sklearn datasets, a new "from root" class method will be needed for converting from ROOT files to the h5 saving format

The input feature transformation and quantisation functions are left as templates 


## Usage
Minimal working example in the train.ipynb notebook or train.py using hls4ml jets dataset


## Installation
Use the environment.yml to generate the 'oblique' conda environment to run, alternatively use the mamba_tq docker image found here: gitlab-registry.cern.ch/cebrown/docker-images/mamba_tq:latest

