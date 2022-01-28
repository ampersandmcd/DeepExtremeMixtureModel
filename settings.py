n_epoch = 50

all_parms = list()

# These are the default parameters. They work pretty well. When naming the directory for each
# experiment they are named based on how they differ from these defaults.
cnn_parms = {
    'ndim': 11,
    'hdim': 10,
    'odim': 6,
    'ksize': (3, 3, 3),
    'padding': (0, 1, 1),
    'use_bnorm': False,
    'nonlin': 'relu'
}

base_parms = {
    'seed': 0,
    'n_train': 450, 'n_val': 250, 'bsize': 10,
    'use_evt': True, 'variable_thresh': False, 'main_func': 'lognormal',
    'ymax': 250, 'mean_multiplier': 1 / 30, 'dropout_multiplier': 0, 'quantile': 0.6, 'n_epoch': 50,
    'cnn_parms': cnn_parms.copy(), 'lr': 1e-2, 'use_mc': False, 'mc_forwards': 30, 'continuous_evt': False}
all_parms.append(base_parms)

# Best performing DEMM w/ variable threshold
cnn_parms = {
    'ndim': 11,
    'hdim': 10,
    'odim': 6,
    'ksize': (3, 3, 3),
    'padding': (0, 1, 1),
    'use_bnorm': False,
    'nonlin': 'relu'
}
variable_thresh_parms = {
    'seed': 0,
    'n_train': 450, 'n_val': 250, 'bsize': 10,
    'use_evt': True, 'variable_thresh': True, 'main_func': 'lognormal',
    'ymax': 250, 'mean_multiplier': 1 / 30, 'dropout_multiplier': 0, 'quantile': 0.6, 'n_epoch': 50,
    'cnn_parms': cnn_parms.copy(), 'lr': 1e-2, 'use_mc': False, 'mc_forwards': 30, 'continuous_evt': False}
all_parms.append(variable_thresh_parms)

# Best performing DEMM w/ fixe threshold
cnn_parms = {
    'ndim': 11,
    'hdim': 10,
    'odim': 6,
    'ksize': (3, 3, 3),
    'padding': (0, 1, 1),
    'use_bnorm': False,
    'nonlin': 'tanh'
}
fixed_thresh_parms = {
    'seed': 0,
    'n_train': 450, 'n_val': 250, 'bsize': 10,
    'use_evt': True, 'variable_thresh': False, 'main_func': 'lognormal',
    'ymax': 250, 'mean_multiplier': 1 / 60, 'dropout_multiplier': 0, 'quantile': 0.6, 'n_epoch': 50,
    'cnn_parms': cnn_parms.copy(), 'lr': 1e-2, 'use_mc': False, 'mc_forwards': 30, 'continuous_evt': False}
all_parms.append(fixed_thresh_parms)

# Best performing Hurdle model
cnn_parms = {
    'ndim': 11,
    'hdim': 10,
    'odim': 6,
    'ksize': (3, 3, 3),
    'padding': (0, 1, 1),
    'use_bnorm': False,
    'nonlin': 'tanh'
}

hurdle_parms = {
    'seed': 0,
    'n_train': 450, 'n_val': 250, 'bsize': 10,
    'use_evt': False, 'variable_thresh': False, 'main_func': 'lognormal',
    'ymax': 250, 'mean_multiplier': 1 / 60, 'dropout_multiplier': 0, 'quantile': 0.6, 'n_epoch': 50,
    'cnn_parms': cnn_parms.copy(), 'lr': 10 ** (-2.5), 'use_mc': False, 'mc_forwards': 30, 'continuous_evt': False}
all_parms.append(hurdle_parms)

# Basic Vandal et al implementation
cnn_parms = {
    'ndim': 11,
    'hdim': 10,
    'odim': 6,
    'ksize': (3, 3, 3),
    'padding': (0, 1, 1),
    'use_bnorm': False,
    'nonlin': 'relu'
}

vandal_parms = {
    'seed': 0,
    'n_train': 450, 'n_val': 250, 'bsize': 10,
    'use_evt': False, 'variable_thresh': False, 'main_func': 'lognormal',
    'ymax': 250, 'mean_multiplier': 0., 'dropout_multiplier': 1e-2, 'quantile': 0.6, 'n_epoch': 50,
    'cnn_parms': cnn_parms.copy(), 'lr': 1e-2, 'use_mc': True, 'mc_forwards': 30, 'continuous_evt': False}
all_parms.append(vandal_parms)
