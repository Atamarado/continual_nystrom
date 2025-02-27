# continual_nystrom

Recommended environment:
```
python==3.11.8
pytorch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
tensorflow==2.12.1
continual-inference==1.2.3
gdown==5.1.0
librosa==0.10.1
pandas==2.2.1
audioread==3.0.1
resampy==0.4.2
ipdb==0.13.8
typing-extensions==4.5.0
ipython==8.18.1
```

## Download the data
Download the Audio Classification Dataset [GTZAN dataset from Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download-directory) and unzip in the directory in ``audio_classification/data/gtzan``.
Only the files ``features_30_sec.csv`` and the folder ``genres_original`` are required for this project.

Download the Action Recognition Dataset THUMOS-14:
* [Anet features](https://drive.google.com/file/d/1Ms709_RSfT2lezPp-0TTkSJCfF-XLeOk/view)
* [Kinetics features](https://drive.google.com/file/d/1jk6eiILBISd3GvG_ZNX8kop-DNSZZPXF/view)

Unzip in ``CoOadTR/data/thumos_anet`` and ``CoOadTR/data/thumos_kin``, respectively.

## Run the scripts

If you have set up the environment and downloaded the required datasets, you can run the training and evaluation of some predefined models:
* For the Audio Classification task, run ``python audio_classification/audio_classification.py``
* For the Online Action Detection, run ``python CoOadTR/main.py``

## Cite this work

If you use or modify this code, you can cite us in LaTex by using:

```
@article{Carreto24cont_nystrom, % preprint
    author  =  {Gin\'es Carreto Pic\'on and
                Illia Oleksiienko and
                Lukas Hedegaard and
                Arian Bakhtiarnia and
                Alexandros Iosifidis
               },
    title   = {Continual Low-Rank Scaled Dot-product Attention},
    journal = {arXiv:2412.03214},
    volume  = {abs/2412.03214},
    year    = {2024}
}
```