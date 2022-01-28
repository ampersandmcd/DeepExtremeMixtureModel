import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import model
import modelling as m
import utils


class Experiment:
    def __init__(self, setting_id, seed, x, y, n_train, n_val, bsize, use_evt, variable_thresh, main_func, ymax,
                 mean_multiplier, dropout_multiplier, quantile, n_epoch, cnn_parms, lr, use_mc, mc_forwards,
                 continuous_evt, sgd_momentum=None):
        """
        This class has the main training loop and stores the best predictions and saves results.
        
        Parameters:
        setting_id - scalar, Unique identifier for these hyper-parameter settings. I typically randomly select this.
                     Just used for file naming.
        seed - int, random seed for splitting data into train, validation and test
        x - tensor, predictors
        y - tensor, target
        n_train - int, number of samples to use for training
        n_val - int, number of samples to use for validation
        bsize - int, batch size
        use_evt - boolean, whether or not to incoporate EVT into mixture model. If false becomes Hurdle baseline
        variable_thresh - boolean, randomly chooses thresholds for each location and window each batch
        main_func - string, density function for non-zero non-execss values. Must be lognormal
        ymax - scalar, max value of y that must be assigned non-zero density by GPD
        mean_multiplier - scalar, weight to assign MSE loss term (the other loss term is NLK)
        dropout_multiplier - scalar, weight for dropout regularization in Vandal et al implementation
        quantile - scalar, what quantile to use to define the excess threshold. If variable_thresh is True then the threshold
                   determined by quantile will only be used for evaluation purposes while the mixture model's threshold
                   will be random.
        n_epoch - int, number of epochs to train
        cnn_parms - dictionary, contains parameters for CNN
        lr - scalar, learning rate
        use_mc - boolean, whether or not to use monte carlo dropout. Setting to True is necessary for Vandal et al
        mc_forwards - int, number of monte carlo forward passes to use for Vandal et al
        continuous_evt - boolean, whether to force the mixture model to be continuous -- doesn't work well
        """
        self.cnn_parms = cnn_parms
        self.setting_id = setting_id
        dtype = m.set_default_tensor_type()
        if not seed is None:  # Randomly shuffle data based on the seed
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            rand_inds = np.random.choice(np.arange(x.shape[0]), size=x.shape[0], replace=False)
            self.x, self.y = torch.tensor(x[rand_inds], device=self.device, dtype=dtype), torch.tensor(y[rand_inds],
                                                                                                       device=self.device,
                                                                                                       dtype=dtype)
        else:
            self.x, self.y = torch.tensor(x, device=self.device, dtype=dtype), torch.tensor(y, device=self.device,
                                                                                            dtype=dtype)
        self.n_train, self.n_val = n_train, n_val
        self.bsize = bsize
        self.use_evt = use_evt
        self.variable_thresh = variable_thresh
        self.mean_multiplier = mean_multiplier
        self.main_func = main_func
        self.seed = seed
        self.use_mcdropout = use_mc
        self.continuous_evt = continuous_evt
        assert not (use_evt and self.use_mcdropout)  # Vandal et al don't use EVT
        if variable_thresh: assert use_evt

        cnn_parms['variable_thresh'] = variable_thresh
        cnn_parms['use_mc'] = use_mc

        self.ymax = ymax
        self.quantile = quantile
        self.n_epoch = n_epoch
        self.lr = lr
        self.sgd_momentum = sgd_momentum
        self.cnn_parms = cnn_parms
        self.mc_forwards = mc_forwards

        # Variables for storing losses/predictions when val loss is lowest
        self.best_epoch = None
        self.best_train_loss = None
        self.best_val_loss = None
        self.best_test_loss = None
        self.best_train_pred = None
        self.best_val_pred = None
        self.best_test_pred = None

        self.final_train_pred = None
        self.final_val_pred = None
        self.final_test_pred = None

        # Set the threshold
        thresh = np.nanquantile(m.tonp(self.y), q=quantile)
        self.threshes = torch.ones_like(self.y, device=self.device) * thresh

        # Make CNN
        torch_model = model.make_cnn(**cnn_parms).to(self.device)
        # Make model object. Model object will be used as a wrapper around pytorch model for e.g. making predictions
        self.model_object = model.STModel(torch_model, use_evt, main_func, ymax, mean_multiplier, dropout_multiplier,
                                          continuous_evt)
        if self.sgd_momentum is None:
            self.optim = torch.optim.Adam(self.model_object.model.parameters(), lr=self.lr)
        else:
            self.optim = torch.optim.SGD(self.model_object.model.parameters(), lr=self.lr, momentum=self.sgd_momentum)

        self.train_losses = list()
        self.val_losses = list()
        self.test_losses = list()
        self.train_pred = None
        self.val_pred = None
        self.test_pred = None
        self.rand_threshes = None

    @property
    def device(self):
        """
        Returns device
        """
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def update_random_threshes(self, lower=0.5, upper=0.95):
        """
        Re-randomizes the saved random thresholds. Should be done after each training epoch
        """
        if lower != 0.5 or upper != 0.95:
            """
            when providing variable thresh as an input to the model we have the
            standardization hard-coded assuming lower=0.5 and upper = 0.95').
            Changing this isn't a big deal just requires changing the hard coded
            standardization in the model class. See the 'forward' method of the
            CNN class in model.py
            """
            assert False
        if self.variable_thresh:
            if self.rand_threshes is None:
                # Saving an array of random thresholds and shuffling it periodically is much
                # faster than generating a new array of random thresholds.
                with open('rand_thresholds.pickle', 'rb') as f:
                    self.rand_threshes = pickle.load(f).reshape(
                        self.y.shape)  # random threshes have same shape as target
            self.rand_threshes = np.random.permutation(self.rand_threshes)
        else:
            assert False

    def _get_data(self, start, end, threshes):
        """
        Forms data set given start and end batch.
        Intended as utility function for the functions: train_data, val_data, test_data
        """
        cur_x = self.x[start:end]
        cur_y = self.y[start:end]
        cur_threshes = threshes[start:end]
        if self.variable_thresh:
            cur_x = torch.cat([cur_x, cur_threshes[:, np.newaxis].repeat(1, 1, cur_x.shape[2], 1, 1)], axis=1)
        return [cur_x, cur_y], cur_threshes

    def train_data(self, threshes):
        """
        Returns traning data
        """
        return self._get_data(start=0, end=self.n_train, threshes=threshes)

    def val_data(self, threshes):
        """
        Returns validation data
        """
        return self._get_data(start=self.n_train, end=self.n_train + self.n_val, threshes=threshes)

    def test_data(self, threshes):
        """
        Returns test data
        """
        return self._get_data(self.n_train + self.n_val, self.x.shape[0], threshes=threshes)

    def get_training_threshes(self):
        """
        Returns the thresholds used for training. If variable_thresh == True then returns the
        random thresholds otherwise it returns the thresholds defined by quantile. This means
        that if variable_thresh is True the model will be trained w/ random thresholds but
        evaluation (i.e. validation and test) will still used fixed thresholds determined by
        quantile.
        """
        if self.variable_thresh:
            return m.totensor(self.rand_threshes)
        else:
            return self.threshes

    def get_batch(self, data, threshes, bnum):
        """
        Gets the next batch of data and thresholds
        Parameters:
        data - list of tensors, data[0] is x and data[1] is y
        threshes - tensor, tensor of thresholds
        bnum - int, batch number
        """
        n_batches = data[0].shape[0] // self.bsize
        if bnum > n_batches:
            return None
        else:
            b_start = self.bsize * bnum
            b_end = self.bsize * (bnum + 1)
            return data[0][b_start:b_end], data[1][b_start:b_end], threshes[b_start:b_end]

    def batch_forward(self, data, threshes, do_mc):
        """
        Performs a forward pass with a single batch and computes losses
        Parameters:
        data - list of tensors, data[0] is the x batch and data[1] is y batch
        threshes - tensor, tensor of this batch's thresholds.
        do_mc - whetehr or not to do monte-carlo dropout (used for Vandal et al)
        """
        x, y = self.split_data(data)
        if not do_mc:
            pred = self.model_object.pred_stats(x, threshes)
        elif do_mc:
            pred = self.model_object.compute_mc_stats(x, threshes, self.mc_forwards)
        losses = self.model_object.compute_losses(pred, y, threshes)
        return pred, losses

    def data_forward(self, data, threshes, train, do_mc):
        """
        Performs a forward pass and computes losses for an entire data set (e.g. for all training data)
        Parameters:
        data - list of tensors, data[0] is the x data we will forward pass and data[1] is y data
        threshes - tensor, thresholds
        train - boolean, whether or not we're training. If we are training we need to do gradient steps
        do_mc - boolean, whether or not to do monte carlo dropout as part of Vandal et al
        """
        n_batches = data[0].shape[0] // self.bsize
        preds, losses = list(), list()
        for batch in range(n_batches):
            x, y, cur_threshes = self.get_batch(data, threshes, batch)
            pred, loss = self.batch_forward([x, y], cur_threshes, do_mc)
            loss = list(loss)
            if train:
                loss[0].backward()  # loss[0] is the loss used for training other elements of loss are other
                # evaluation metrics not intended to be used for training.

                if np.isnan(m.tonp(loss[0])) or np.isinf(m.tonp(loss[0])):
                    assert False
                self.optim.step()
                self.optim.zero_grad()
            loss[0] = m.tonp(loss[0])
            preds.append(np.concatenate(m.tonp(pred), axis=1)), losses.append(loss)
        preds = np.concatenate(preds, axis=0)
        losses = Experiment._avg_losses(losses)
        return preds, losses

    def evaluate(self, y, pred, threshes):
        """
        Evaluate our predictions on some target data
        Parameters:
        y - tensor, target
        pred - tensor, tensor of mixture model parameters
        threshes - tensor, thresholds
        """
        pred = utils.to_stats(m.totensor(pred))
        return np.array(self.model_object.compute_metrics(y, pred, threshes))

    def forward_eval(self, data, threshes, train, do_mc):
        """
        Performs a forward pass with some data to compute predictions then computes some evaluation metrics of those
        predictions.
        Parameters:
        data - list of tensors, data[0] is the x data we will forward pass and data[1] is y data
        threshes - tensor, thresholds
        train - boolean, whether or not we need to perform gradient descent
        do_mc - boolean, whether or not to do monte carlo dropout. Necessary for Vandal et al
        """
        if train:
            self.model_object.model.train()
        else:
            self.model_object.model.eval()
        pred, _ = self.data_forward(data, threshes, train, do_mc)
        losses = self.evaluate(data[1][:pred.shape[0]], pred, threshes[:pred.shape[0]])
        return pred, losses

    def train(self):
        """
        This is the training loop
        """
        for epoch in range(self.n_epoch):
            # If using random thresholds we need to randomize them each epoch
            if self.variable_thresh: self.update_random_threshes()
            # Train
            train_pred, train_losses = self.forward_eval(*self.train_data(self.get_training_threshes()), train=True,
                                                         do_mc=False)
            # Compute validation and test loss
            val_pred, val_losses = self.forward_eval(*self.val_data(self.threshes), train=False, do_mc=False)
            test_pred, test_losses = self.forward_eval(*self.test_data(self.threshes), train=False, do_mc=False)

            self.train_losses.append(train_losses)
            self.val_losses.append(val_losses)
            self.test_losses.append(test_losses)
            self.final_train_pred = m.tonp(train_pred)
            self.final_val_pred = m.tonp(val_pred)
            self.final_test_pred = m.tonp(test_pred)

            if self.best_val_loss is None or val_losses[0] < self.best_val_loss[0]:
                print('best results updated')
                self.best_epoch = epoch
                self.best_train_loss = train_losses
                self.best_val_loss = val_losses
                self.best_test_loss = test_losses
                self.best_train_pred = m.tonp(train_pred)
                self.best_val_pred = m.tonp(val_pred)
                self.best_test_pred = m.tonp(test_pred)

            print(epoch)
            print('train losses: ', train_losses)
            print('val losses: ', val_losses)
            print('test losses: ', test_losses)
            print()
        print('training complete')
        # If using Vandal et al then after training we have to do the monte carlo dropout stuff
        if self.use_mcdropout:
            self.mc_eval()
            print()
            print()
            print()
            print('mc losses: ')
            print('train losses: ', self.mc_train_losses)
            print('val losses: ', self.mc_val_losses)
            print('test losses: ', self.mc_test_losses)

    def split_data(self, data):
        """
        This probably didn't need to be its own method
        """
        return data[0], data[1]

    def _avg_losses(losses):
        """
        Averages a list of losses
        Parameters:
        losses - list of arrays
        """
        losses = np.stack([l for l in losses], axis=0)
        return np.mean(losses, axis=0)

    @property
    def fname(self):
        """
        Returns a file name for this experiment
        """
        return str(self.seed) + '-' + str(self.setting_id) + '.pickle'

    def save(self, directory, save_model=False):
        """
        Saves the experiment. Thresholds, optimizer, x, and y aren't saved
        directory - string, where to save
        save_model - boolean, whether or not to save the pytorch model
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        model = self.model_object.model
        optim = self.optim
        threshes = self.threshes
        x, y = self.x, self.y

        if save_model:
            torch.save(self.model_object.model, os.path.join(directory, str(self.seed) + '.pth'))

        self.model_object.model = None
        self.optim = None
        self.threshes = None
        self.x, self.y = None, None

        with open(os.path.join(directory, self.fname), 'wb') as f:
            pickle.dump(self, f)
        self.model_object.model = model
        self.optim = optim
        self.threshes = threshes
        self.x, self.y = x, y

    def mc_eval(self):
        """
        This does the monte carlo dropout stuff necessary for Vandal et al 
        """
        self.mc_train_pred, self.mc_train_losses = self.forward_eval(*self.train_data(self.get_training_threshes()),
                                                                     train=False, do_mc=True)
        self.mc_val_pred, self.mc_val_losses = self.forward_eval(*self.val_data(self.threshes), train=False, do_mc=True)
        self.test_pred, self.mc_test_losses = self.forward_eval(*self.test_data(self.threshes), train=False, do_mc=True)
