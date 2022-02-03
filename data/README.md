# DeepExtremeMixtureModel/data

This subdirectory contains data used in experiments and associated files.

## Overview

The data used in our experiments comes from the [SubX](http://cola.gmu.edu/subx/) project, and consists of a spatiotemporal ensemble rainfall forecast predictor set with spatiotemporal ground truth rainfall as target.
- X is the 3-day (overlapping) average ensemble rainfall predictions.
  - Before use in pipeline, we compute its log+1 transform (i.e. x = log(1 + x))
  - Data in `all_data.pickle` has not been log-transformed or standardized.
  - Data in `processed_data.pickle` has been log-transformed and standardized.
  - Dimensions of x are:
      - \# of weekly model simulations (939)
      - \# of models in ensemble (11)
      - numerical model rainfall predictions for 7 day averages
      - spatial height (29)
      - spatial width (59)
- Y is the observed 3-day average rainfall 10 days in advance. No further transformations were made before use w/ proposed method. Dimensions of y are:
    - \# of weekly model simulations (939)
    - ignore this axis, added for consistency
    - spatial height (29)
    - spatial width (59)

## Files

- *`processed_data.pickle` - This is the processed data.
- *`rand_thresholds.pickle` - Generating a large number of random thresholds to train the variable threshold DEMM turns out to be relatively slow, so we run the generation step once in advance, fix the output, and then randomly shuffle it during training to simulate generating new random thresholds.

The two .pickle files marked with an * are not included in this repository due to file size, but are available for download via Google Drive in a 400MB zip folder [here](https://drive.google.com/file/d/1vy9h3uiarpwrFCGFgr9Ex81q2_1v0f34/view?usp=sharing).