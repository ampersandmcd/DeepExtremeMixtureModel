import pickle
import numpy as np
import torch
import experiment
import gc
import time
import os
import names
import settings as s


if __name__ == "__main__":

    # In practice I found that I ran into numerical issues with float. Lot's of changes
    # have been made since then to make the model behave better so using double
    # may no longer be necessary.
    torch.set_default_tensor_type(torch.DoubleTensor)
    with open('../data/processed_data.pickle', 'rb') as f:
        data = pickle.load(f)
    x, y = data['x'], data['y']

    # All results will be saved within results_dir
    results_dir = os.path.join(names.results_dir(), 'all_results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    all_losses = {}
    # s.all_parms is a list of hyper-parameter settings. We iterate through this list of
    # hyper-parameters and test 10 seeds for each hyper-parameter setting
    for counter, cur_settings in enumerate(s.all_parms):
        # The first set of hyper-parameters in our list is the default hyper-parameters
        # so we will skip them.
        if counter == 0: continue
        all_losses = list()
        # setting_id will be shared across all seeds for this choice of hyper-parameters and used for file naming
        setting_id = np.random.randint(999999999)
        cur_settings['setting_id'] = setting_id
        save_dir = os.path.join(results_dir, names.settings_to_fname(s.all_parms[0], cur_settings))
        if os.path.exists(save_dir):
            print('skipping: ', save_dir)
            continue
        print(save_dir)
        os.mkdir(save_dir)
        for seed in range(10):  # iterate through 10 different seeds
            settings = cur_settings.copy()
            settings['cnn_parms'] = cur_settings['cnn_parms'].copy()
            settings['seed'] = seed
            settings['x'] = x.copy()
            settings['y'] = y.copy()
            print('##################################')
            print('counter: ', counter, '\t\tseed: ', seed)
            print('##################################')
            gc.collect()
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False
            t0 = time.time()
            e = experiment.Experiment(**settings)
            e.train()
            print(time.time() - t0)
            e.save(save_dir, save_model=False)
            all_losses.append([e.best_train_loss, e.best_val_loss, e.best_test_loss])
        with open(os.path.join(save_dir, 'all_losses.pickle'), 'wb') as f:
            pickle.dump(all_losses, f)
