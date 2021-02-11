#!/usr/bin/env bash
mkdir VQAR_all
cd VQAR_all
wget https://zenodo.org/record/4535413/files/VQAR_code.zip
unzip VQAR_code.zip
mv VQAR_code/VQAR .
mv VQAR_code/Scallop .
rm -rf VQAR_code
rm -rf VQAR_code.zip
cd VQAR
wget https://zenodo.org/record/4535747/files/data.zip
unzip data.zip
rm -rf data.zip
