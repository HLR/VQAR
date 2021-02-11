#!/usr/bin/env bash
cd VQAR_all
cd VQAR
conda env create -f VQAR_env.yml
conda activate VQAR_env
pip uninstall PySDD
pip install git+https://github.com/wannesm/PySDD.git#egg=PySDD