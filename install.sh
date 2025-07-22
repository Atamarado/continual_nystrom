#!/bin/bash -i

# Creation and installation of all the libraries in the continual_nystrom environment

echo "Preparing conda envionrment..."
conda create -y -n continual_nystrom python=3.13.2 pip
conda activate continual_nystrom

yes | pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
yes | pip install ipdb==0.13.13 scikit-learn==1.6.1 pandas==2.2.3 gdown==5.2.0
echo "Done"

echo "Downloading CoOadTR files"
python download_CoOadTR.py
unzip CoOadTR/data/thumos_anet/OadTR_THUMOS.zip -d CoOadTR/data/thumos_anet
rm CoOadTR/data/thumos_anet/OadTR_THUMOS.zip
unzip CoOadTR/data/thumos_kin/OadTR_THUMOS_Kinetics.zip -d CoOadTR/data/thumos_anet
rm CoOadTR/data/thumos_kin/OadTR_THUMOS_Kinetics.zip
echo "Done"

echo "Downloading Electricity Load Diagrams files"
mkdir -p electricity/data
wget https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip -O electricity/data/electricityloaddiagrams20112014.zip
unzip electricity/data/electricityloaddiagrams20112014.zip -d electricity/data
rm electricity/data/electricityloaddiagrams20112014.zip
echo "Done"


