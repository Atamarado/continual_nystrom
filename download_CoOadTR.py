import os
import gdown

os.makedirs("CoOadTR/data/thumos_anet", exist_ok=True)
os.makedirs("CoOadTR/data/thumos_kin", exist_ok=True)

gdown.download("https://drive.google.com/file/d/1Ms709_RSfT2lezPp-0TTkSJCfF-XLeOk/view?usp=sharing", "CoOadTR/data/thumos_anet/OadTR_THUMOS.zip", fuzzy=True)
gdown.download("https://drive.google.com/file/d/1jk6eiILBISd3GvG_ZNX8kop-DNSZZPXF/view?usp=sharing", "CoOadTR/data/thumos_kin/OadTR_THUMOS_Kinetics.zip", fuzzy=True)