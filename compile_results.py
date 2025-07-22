import os
import pickle
import pandas as pd

task = 'electricity' # CoOadTR, audio_classification, electricity

RESULTS_FOLDER = task + '/raw_results'

def summarize_results(results_folder=RESULTS_FOLDER):
    match task:
        case 'CoOadTR':
            CONFIG_COLS = ['model', 'num_layers', 'feature', 'num_landmarks', 'freeze_weights']
        case 'audio_classification' | 'electricity':
            CONFIG_COLS = ['model', 'num_layers', 'num_landmarks', 'freeze_weights']
        case _:
            raise NotImplementedError

    file_experiment = os.listdir(results_folder)

    res_list = []
    for file in file_experiment:
        with open(os.path.join(results_folder, file), 'rb') as f:
            try:
                results = pickle.load(f)
            except:
                print("Cannot open path "+os.path.join(results_folder, file))
                continue

        config = results["config"]

        res_dict = {}

        if results["freeze_weights"] is None:
            res_dict["freeze_weights"] = "None"
        else:
            res_dict["freeze_weights"] = str(results["freeze_weights"])

        # Load config
        res_dict["data_seed"] = config.data_seed
        res_dict["model_seed"] = config.model_seed

        res_dict["model"] = config.model
        res_dict["num_layers"] = config.num_layers
        if hasattr(config, "seq_len"):
            res_dict["seq_len"] = config.seq_len
        else:
            res_dict["seq_len"] = -1

        res_dict["feature"] = config.feature
        res_dict["num_landmarks"] = config.num_landmarks

        if 'test_performance' not in results.keys():
            results['test_performance'] = results['test_accuracy']

        performance_metrics = results['test_performance']
        if isinstance(performance_metrics, tuple):
            performance_metrics = performance_metrics[0]
            results['test_performance'] = results['test_performance'][0]

        if type(performance_metrics) == dict:
            for metric in performance_metrics:
                res_dict['test_' + metric] = results['test_performance'][metric]

        res_dict["test_time"] = results["test_time"]
        res_dict["flops"] = results["flops"]
        res_dict["valley_mem_cost"] = results["valley_mem_cost"]
        res_dict["peak_mem_cost"] = results["peak_mem_cost"]

        res_list.append(res_dict)

    res_df = pd.DataFrame.from_records(res_list)

    if type(performance_metrics) == dict:
        agg_dict = {f'test_{metric}': ['mean', 'std'] for metric in performance_metrics}
    else:
        agg_dict = {}
    agg_dict['test_time'] = ['mean', 'std']
    agg_dict['flops'] = ['mean', 'std']
    agg_dict['valley_mem_cost'] = ['mean', 'std']
    agg_dict['peak_mem_cost'] = ['mean', 'std']
    agg_dict['data_seed'] = ['count']

    grouped_df = res_df.groupby(CONFIG_COLS).agg(agg_dict).reset_index()

    return grouped_df

if __name__ == "__main__":
    grouped_df = summarize_results()
    pass


