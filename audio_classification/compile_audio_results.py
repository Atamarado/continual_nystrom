import os
import pickle
import pandas as pd

RESULTS_FOLDER = 'raw_results'

CONFIG_COLS = ['model', 'num_layers', 'num_landmarks', 'fit_layer_epochs', 'freeze_weights']
ACC_COLS = ['train_accuracy', 'val_accuracy', 'test_accuracy']

if __name__ == "__main__":
    file_experiment = os.listdir(RESULTS_FOLDER)

    res_list = []
    for file in file_experiment:
        with open(os.path.join(RESULTS_FOLDER, file), 'rb') as f:
            results = pickle.load(f)

        config = results["config"]

        res_dict = {}
        # Load config
        res_dict["data_seed"] = config.data_seed
        res_dict["model_seed"] = config.model_seed

        res_dict["model"] = config.model
        res_dict["num_layers"] = config.num_layers
        res_dict["num_landmarks"] = config.num_landmarks if config.model in ['nystromformer', 'continual_nystrom'] else 0
        res_dict["fit_layer_epochs"] = tuple(config.fit_layer_epochs) if config.model in ['nystromformer', 'continual_nystrom'] else tuple()
        res_dict["freeze_weights"] = results["freeze_weights"] if config.model in ['nystromformer', 'continual_nystrom'] else False

        # Load results
        res_dict["train_accuracy"] = results["train_accuracy"].item()
        res_dict["val_accuracy"] = results["val_accuracy"].item()
        res_dict["test_accuracy"] = results["test_accuracy"].item()

        res_list.append(res_dict)

    res_df = pd.DataFrame.from_records(res_list)
    res_df = res_df.groupby(CONFIG_COLS).agg({
        'train_accuracy': ['mean', 'std'],
        'val_accuracy': ['mean', 'std'],
        'test_accuracy': ['mean', 'std']
    }).reset_index()

    res_df.to_csv('results_audio_classification.csv')


