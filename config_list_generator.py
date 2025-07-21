data_seed_list = [0]
model_seed_list = range(3)
model_type_list = ["base_continual", "base", "nystromformer", "continual_nystrom"]
num_layers_list = range(1, 3)
seq_len_list = [-1]

if __name__=="__main__":
    filename = "config_list.txt"
    with open(filename, "w+") as f:
        for seq_len in seq_len_list:
            for model in model_type_list:
                for data_seed in data_seed_list:
                    for model_seed in model_seed_list:
                        for num_layers in num_layers_list:
                            fit_layer_epochs = '[5' + ',5' * (num_layers-1) + ']'
                            model_completed = False
                            for num_landmarks in [4, 8, 12, 16]:
                                if model in ['base_continual', 'base'] and model_completed:
                                    break
                                model_completed = True
                                params = "--data_seed {} --model_seed {} --model {} --num_layers {} --seq_len {} --dataset electricity --fit_layer_epochs {} --num_landmarks {}\n".format(
                                    data_seed, model_seed, model, num_layers, seq_len, fit_layer_epochs, num_landmarks)
                                f.write(params)