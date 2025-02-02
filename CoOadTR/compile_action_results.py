import os
import re
import pandas as pd
import pickle

PATTERN = r'(?P<model>.+?)_(?P<num_layers>\d+)_layers_(?P<num_landmarks>\d+)_landmarks_(?P<fit_layer_epochs>-?[\d,\s]+)-_(?P<freeze_weights>\d+)_feature_(?P<feature>.+?)_seeds_(?P<seed>\d+)'

if __name__ == '__main__':
    os.chdir("/mnt/data/gines/CoOadTR/combine/")

    CONFIG_COLS = ["model", "num_layers", "num_landmarks", "fit_layer_epochs", "freeze_weights", "feature"]

    configs = []
    mAP = []
    mcAP = []

    for directory, _, _ in os.walk('.'):
        if directory == '.':
            continue

        epoch_iter = 20
        while not os.path.exists(directory + "/log_tran&test"+str(epoch_iter)+".txt") and epoch_iter >= 0:
            epoch_iter -= 1

        if epoch_iter < 0:
            continue

        path = directory + "/log_tran&test"+str(epoch_iter)+".txt"
        if not os.path.exists(path):
            continue

        try:
            with open(path, 'rb') as f:
                res_dict = pickle.load(f)
        except:
            continue

        mAP.append(res_dict['test_mAP']*100)
        mcAP.append(res_dict['test_mcAP']*100)

        config = vars(res_dict['config'])
        config = {k: config[k] for k in CONFIG_COLS+["seed"]}
        config['fit_layer_epochs'] = tuple(config['fit_layer_epochs']) if config['model'] in ['nystromformer', 'continual_nystrom'] else tuple()

        match = re.match(PATTERN, directory)
        if match:
            config['freeze_weights'] = match.group('freeze_weights')

        configs.append(config)

    res_df = pd.DataFrame.from_records(configs)
    res_df["mAP"] = mAP
    res_df["mcAP"] = mcAP

    res_df = res_df.groupby(CONFIG_COLS).agg({
        'mAP': ['mean', 'std'],
        'mcAP': ['mean', 'std'],
        'seed': ['count'],
    }).reset_index()
    pass


