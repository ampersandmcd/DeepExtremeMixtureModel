# DeepExtremeMixtureModel
Official code release for Deep Extreme Mixture Model by Wilson, McDonald, Galib, Tan, and Luo.

## Overview

This code implements four kinds of models:
1. DEMM w/ variable threshold
2. DEMM w/ fixed threshold
3. lognormal hurdle model
4. Vandal et al (currently returns nan after performing monte carlo dropout)
run_st.py will train one version of each of these models based on the hyper-parameters saved in settings.py

A couple notes about how the baselines are implemented:
The lognormal Hurdle model is implemented by setting the threshold to an arbitrarily large value thus bypassing the EVT component of the mixture model.
Vandal et al is basically the Hurdle model but after training is complete it performs monte carlo dropout and estimates the statistical parameters as described in Vandal et al.

Training a model on one seed (i.e. one data split) takes ~4.5 minutes for a GTX 1080 Ti.

The following files are included:
- experiment.py - This file contains the main training loop and stores experimental results (e.g. predictions, loss)
- model.py - This helps make and evaluate predictions
- modelling.py - This has the functions for calculating the log-likelihood and mean among many other functions
- names.py - A file for saving important file names.
- *processed_data.pickle - This is the processed data.
- *rand_thresholds.pickle - Generating a large number of random thresholds to train the variable threshold DEMM turns out to be relatively slow so I did once and then randomly shuffle it during training to simulate generating new random thresholds.
- run_st.py - This runs experiments based on the hyper-parameters saved in settings.py
- settings.py - This contains the optimal hyper-parameters for the proposed method and baselines. This is loaded by run_st.py which can then re-run the experiments.
- utils.py - Miscallaneous utility functions including some evaluation metrics.
- concrete_dropout.py - Important functions for concrete dropout
- show_results.ipynb - Demonstrates how to load results after running run_st.py

The two .pickle files marked with an * are not included in this repository due to file size, but are available for download via Google Drive in a 400MB zip folder [here](https://drive.google.com/file/d/1vy9h3uiarpwrFCGFgr9Ex81q2_1v0f34/view?usp=sharing).
