# DeepExtremeMixtureModel/archive

This subdirectory contains first-iteration source code which is no longer used in experiments, but which is worth keeping around.

## Overview

The code in this subdirectory implements four kinds of models:
1. DEMM w/ variable threshold
2. DEMM w/ fixed threshold
3. lognormal hurdle model
4. Vandal et al (currently returns nan after performing monte carlo dropout)
run_st.py will train one version of each of these models based on the hyperparameters saved in settings.py

A few notes about how the baselines are implemented:
- The lognormal Hurdle model is implemented by setting the threshold to an arbitrarily large value thus bypassing the EVT component of the mixture model.
- Vandal et al is essentially the Hurdle model, but after training is complete it performs monte carlo dropout and estimates the statistical parameters as described in Vandal et al.

Training a model on one seed (i.e. one data split) takes ~4.5 minutes for a GTX 1080 Ti.

## Files

- `experiment.py` - This file contains the main training loop and stores experimental results (e.g. predictions, loss)
- `model.py` - This helps make and evaluate predictions
- `modelling.py` - This has the functions for calculating the log-likelihood and mean among many other functions
- `names.py` - A file for saving important file names.
- `run_st.py` - This runs experiments based on the hyperparameters saved in `settings.py`
- `settings.py` - This contains the optimal hyperparameters for the proposed method and baselines. This is loaded by run_st.py which can then re-run the experiments.
- `util.py` - Miscellaneous utility functions including some evaluation metrics.
- `concrete_dropout.py` - Important functions for concrete dropout
- `show_results.ipynb` - Demonstrates how to load results after running run_st.py
