def results_dir():
    """
    Returns name of directory where results are stored
    """
    return '.'


def settings_to_fname(base_dic, dic):
    """
    Constructs a name for experiment based on its hyper-parameter
    settings. The name describes how the experiment's hyper-parameter
    settings differ from the 'default' hyper-parameter settings.
    Parameters:
    base_dic - dictionary of hyper-parameter settings, this should
               be the dictionary with the 'default' settings
    dic - dictionary of hyper-parameter settings, this is the the
          dictionary that you would like to construct a name for
    
    Returns:
    string with the name of the experiment
    """
    fname = 'differences@'
    for k in dic.keys():
        if k not in base_dic.keys(): continue
        if k == 'setting_id' or k == 'seed': continue
        if base_dic[k] == dic[k]: continue
        if k == 'cnn_parms':
            if base_dic[k]['hdim'] != dic[k]['hdim']:
                fname += 'hid~' + str(dic[k]['hdim']) + '@'
            if base_dic[k]['use_bnorm'] != dic[k]['use_bnorm']:
                fname += 'bnorm~' + str(dic[k]['use_bnorm']) + '@'
            if base_dic[k]['nonlin'] != dic[k]['nonlin']:
                fname += 'nonlin~' + str(dic[k]['nonlin']) + '@'
        else:
            fname += str(k) + '~' + str(dic[k]) + '@'
    if fname == 'differences@': fname = 'base'
    return sort_name(fname)


def sort_name(d):
    """
    Ensures names are always construced in a consistent order
    """
    if d == 'base': return d
    return 'differences' + '@'.join(sorted(d.split('@')[1:])) + '@'
