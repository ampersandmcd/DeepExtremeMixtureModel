import math
import numpy as np
import torch

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from scipy.stats import genpareto

import util


class SpatiotemporalLightningModule(pl.LightningModule):

    def __init__(self, st_params, cnn_params, seed, lr, n_epoch):
        super().__init__()
        self.save_hyperparameters()
        self.st_params = st_params
        self.cnn_params = cnn_params
        self.seed = seed
        self.lr = lr
        self.n_epoch = n_epoch

        # build 3D CNN backbone and spatiotemporal model
        cnn = make_cnn(**cnn_params).to(get_device())
        st_model = SpatiotemporalModel(model=cnn, **st_params).to(get_device())
        self.st_model = st_model

    def training_step(self, batch, batch_idx):
        self.train()
        x = batch["x"].type(torch.FloatTensor).to(self.device)
        y = batch["y"].type(torch.FloatTensor).to(self.device)
        if self.st_model.variable_thresh:
            # generate random thresholds in [0.5, 0.95] and augment predictors
            threshes = 0.45 * torch.rand_like(y) + 0.5
            x = torch.cat([x, threshes[:, np.newaxis].repeat(1, 1, x.shape[2], 1, 1)], axis=1)
        else:
            # generate fixed threshold but do not augment predictors
            t = np.nanquantile(to_np(y), self.st_model.quantile)
            threshes = torch.ones_like(y) * t
        if self.st_model.use_mc:
            # perform monte-carlo forward pass for Vandal et al.
            pred = self.st_model.compute_mc_stats(x, threshes, self.st_model.mc_forwards)
        else:
            # perform standard forward pass
            pred = self.st_model.compute_stats(x, threshes)
        loss, nll_loss, rmse_loss = self.st_model.compute_losses(pred, y, threshes)
        self.log("t_loss", loss)
        self.log("t_nll_loss", nll_loss)
        self.log("t_rmse_loss", rmse_loss)    # t for train
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        x = batch["x"].type(torch.FloatTensor).to(self.device)
        y = batch["y"].type(torch.FloatTensor).to(self.device)
        if self.st_model.variable_thresh:
            # generate fixed threshold at test time and augment predictors
            t = np.nanquantile(to_np(y), self.st_model.quantile)
            threshes = torch.ones_like(y) * t
            x = torch.cat([x, threshes[:, np.newaxis].repeat(1, 1, x.shape[2], 1, 1)], axis=1)
        else:
            # generate fixed threshold at test time but do not augment predictors
            t = np.nanquantile(to_np(y), self.st_model.quantile)
            threshes = torch.ones_like(y) * t
        if self.st_model.use_mc:
            # perform monte-carlo forward pass for Vandal et al.
            pred = self.st_model.compute_mc_stats(x, threshes, self.st_model.mc_forwards, test=True)
        else:
            # perform standard forward pass
            pred = self.st_model.compute_stats(x, threshes, test=True)
        metrics = to_item(self.st_model.compute_metrics(y, pred, threshes))
        return {
            "loss": metrics[0],
            "nll_loss": metrics[1],
            "rmse_loss": metrics[2],
            "zero_brier": metrics[3],
            "moderate_brier": metrics[4],
            "excess_brier": metrics[5],
            "acc": metrics[6],
            "f1_micro": metrics[7],
            "f1_macro": metrics[8],
            "auc_macro_ovo": metrics[9],
            "auc_macro_ovr": metrics[10]
        }

    def validation_epoch_end(self, outputs):
        metric_names = outputs[0].keys()
        for metric_name in metric_names:
            metric_mean = np.mean([o[metric_name] for o in outputs])
            if "loss" in metric_name:
                self.log(f"v_{metric_name}", metric_mean, prog_bar=True)
            else:
                self.log(f"v_{metric_name}", metric_mean)   # v for validation

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        metric_names = outputs[0].keys()
        for metric_name in metric_names:
            metric_mean = np.mean([o[metric_name] for o in outputs])
            self.log(f"f_{metric_name}", metric_mean)   # f for final

    def configure_optimizers(self):
        return torch.optim.Adam(self.st_model.parameters(), lr=self.lr)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpatiotemporalModel(nn.Module):

    def __init__(self, model, use_evt, moderate_func, ymax, mean_multiplier, dropout_multiplier, continuous_evt, variable_thresh, quantile, use_mc, mc_forwards):
        """
        Initialize spatiotemporal model.
        Parameters:
        model - pytorch model, the pytorch model
        use_evt - boolean, whether or not to use extreme value theory
        moderate_func - string, what density function to use for non-zero non-excess values. Must be 'lognormal'
        ymax - scalar, the max y value that must be assigned non-zero probability by the mixture model
        mean_multiplier - scalar, the weight assigned to the RMSE component of the loss function (the other component is NLK)
        dropout_multiplier - scalar, the weight for dropout regularization in Vandal et al
        continuous_evt - boolean, whether or not the mixture model's density must be continuous at 0. Setting to True
                         doesn't work well.
        variable_thresh - bool, whether or not the model is trained at various threshold values
        quantile - float in [0, 1], quantile used for EVT
        use_mc - bool, whether or not to use montecarlo dropout evaluation
        mc_forwards - int, number of monte carlo forward passes to use for Vandal et al
        """
        super().__init__()
        set_default_tensor_type()
        self.model = model
        self.mean_multiplier = mean_multiplier
        self.dropout_multiplier = dropout_multiplier
        self.moderate_func = moderate_func
        self.use_evt = use_evt
        self.continuous_evt = continuous_evt
        self.variable_thresh = variable_thresh
        self.quantile = quantile
        self.use_mc = use_mc
        self.mc_forwards = mc_forwards
        if self.continuous_evt:
            assert use_evt
        self.ymax = ymax

    def forward(self, x, threshes):
        return self.compute_stats(x, threshes)

    def effective_thresh(self, threshes):
        """
        The effective threshold is the threshold to be used by the mixture model. If we're not using EVT
        then the threshold is infinite (all non-zero values are modeled by lognormal). Note that we cannot actually
        set it to infinity, however, as that breaks the computation graph during the backward pass.
        We find setting the value to 999999999. works well, as precipitation values are much smaller than this in practice.
        Parameters:
        threshes - tensor, the actual thresholds
        """
        if self.use_evt:
            return threshes
        else:
            return torch.ones_like(threshes, device=get_device()) * 999999999.

    def _to_stats(self, cur_raw, threshes):
        """
        Takes a tensor of raw neural net output and constrains it.
        Parameters:
        cur_raw - tensor, raw neural net output
        threshes - tensor, thresholds

        Returns:
        bin_pred - tensor, bin_pred[:, 0] is probability of 0 rainfall and bin_pred[:, 1] is probability of excess rainfall
                   given non-zero rainfall
        gpd_pred - tensor, GPD parameters -- gpd_pred[:, 0] is xi and gpd_pred[:, 1] is sigma
        norm_pred - tensor, lognormal parameters -- norm_pred[:, 0] is mu, norm_pred[:, 1] is variance
        """
        bin_pred_, gpd_pred_, norm_pred_ = util.to_stats(cur_raw)  # This is just splitting the data
        bin_pred, gpd_pred, norm_pred = all_constraints(
            bin_pred_, gpd_pred_, norm_pred_, torch.ones_like(gpd_pred_[:, 0], device=get_device()) * self.ymax,
            self.continuous_evt, main_func=self.moderate_func, thresholds=threshes)
        # If we're not using evt then set probability of excess to 0. The weird stuff w/ relu was necessary
        # to make sure gradients weren't broken.
        if not self.use_evt:
            bin_pred = torch.cat([bin_pred[:, 0:1], torch.relu(-1 * torch.abs(bin_pred[:, 1:2]))], dim=1)
        return bin_pred, gpd_pred, norm_pred

    def compute_stats(self, x, threshes, test=False):
        """
        Makes predictions then converts raw predictions to constrained mixture model parameters.
        Parameters:
        x - tensor, predictors
        threshes - tensor, threshold
        test - boolean, set to True for Vandal et al at test time so we do true dropout not approximate dropout

        Returns:
        bin_pred - tensor, bin_pred[:, 0] is probability of 0 rainfall and bin_pred[:, 1] is probability of excess rainfall
                   given non-zero rainfall
        gpd_pred - tensor, GPD parameters -- gpd_pred[:, 0] is xi and gpd_pred[:, 1] is sigma
        norm_pred - tensor, lognormal parameters -- norm_pred[:, 0] is mu, norm_pred[:, 1] is variance
        """
        cur_raw = self.model(x, test)
        if np.isnan(to_np(cur_raw)).any():
            print('nans encountered')
        bin_pred, gpd_pred, norm_pred = self._to_stats(cur_raw, threshes)
        return bin_pred, gpd_pred, norm_pred

    def split_pred(self, pred):
        """
        Just splits a list of predictions up into the constituent parts

        Returns:
        bin_pred - tensor, bin_pred[:, 0] is probability of 0 rainfall and bin_pred[:, 1] is probability of excess rainfall
                   given non-zero rainfall
        gpd_pred - tensor, GPD parameters -- gpd_pred[:, 0] is xi and gpd_pred[:, 1] is sigma
        norm_pred - tensor, lognormal parameters -- norm_pred[:, 0] is mu, norm_pred[:, 1] is variance
        """
        bin_pred, gpd_pred, norm_pred = pred
        return bin_pred, gpd_pred, norm_pred

    def compute_losses(self, pred, y, threshes):
        """
        Computes NLK, RMSE, and their weighted sum.
        Parameters:
        pred - list of tensors, list of mixture model parameters
        y - tensor, target
        threshes - tensor, thresholds

        Returns:
        loss - tensor, this is the weighted average of NLK and RMSE. This is the loss used for training
               so it can be back-propogated
        predicted_logliks - array, negative log-likelihood
        rmse_loss - array, MSE of point predictions
        """
        bin_pred, gpd_pred, norm_pred = self.split_pred(pred)
        # I'm pretty sure this line was necessary to gpd_pred later on for debugging purposes. I don't think
        # its needed now but decided to keep it just in case.
        gpd_pred = util.split_var(gpd_pred)
        nll_loss = -torch_nanmean(
            loglik(
                y, gpd_pred, norm_pred, bin_pred[:, 0], bin_pred[:, 1], self.effective_thresh(threshes), self.moderate_func
            )
        )
        rmse_loss = self.compute_rmse(y, pred, threshes)

        loss = (1 - self.mean_multiplier) * nll_loss + self.mean_multiplier * rmse_loss + \
               (0 if (self.dropout_multiplier == 0) else (self.dropout_multiplier * self.model.regularisation()))
        return loss, nll_loss, rmse_loss

    def compute_metrics(self, y, pred, threshes):
        """
        Computes a wide range of evaluation metrics
        Parameters:
        y - tensor, target variable
        pred - list of tensors, list of mixture model parameters
        threshes - tensor, thresholds

        Returns:
        loss - scalar, this is the weighted average of NLK and MSE.
        predicted_logliks - scalar, negative log-likelihood
        rmse_loss - scalar, MSE of point predictions
        zero_brier - scalar, brier score of 0 rainfall class
        moderate_brier - scalar, brier score for non-zero non-excess class
        excess_brier - scalar, brier score for excess class
        acc - scalar, accuracy
        f1_micro - scalar, f1 micro of all classes
        f1_macro - scalar, f1 macro of all classes
        auc_macro_ovo - scalar, auc macro one versus one
        auc_macro_ovr - scalar, auc macro one versus all
        """
        bin_pred, gpd_pred, norm_pred = self.split_pred(pred)
        predicted_loglik = -torch_nanmean(
            loglik(
                y, gpd_pred, norm_pred, bin_pred[:, 0], bin_pred[:, 1], self.effective_thresh(threshes), self.moderate_func
            )
        )
        rmse_loss = self.compute_rmse(y, pred, threshes)

        zero_brier, moderate_brier, excess_brier, acc, f1_micro, f1_macro, auc_macro_ovo, auc_macro_ovr = \
            self.compute_class_metrics(y, pred, threshes)

        loss = predicted_loglik + self.mean_multiplier * rmse_loss + \
               (0 if (self.dropout_multiplier == 0) else (self.dropout_multiplier * self.model.regularisation()))
        return to_np(loss), to_np(predicted_loglik), to_np(
            rmse_loss), zero_brier, moderate_brier, excess_brier, acc, f1_micro, f1_macro, auc_macro_ovo, auc_macro_ovr

    def compute_brier_scores(self, y, pred, threshes):
        """
        Computes brier scores for each class.
        Parameters:
        y - tensor, target variable
        pred - list of tensors, list of mixture model parameters
        threshes - tensor, thresholds

        Returns:
        zero_brier - scalar, brier score of 0 rainfall class
        moderate_brier - scalar, brier score for non-zero non-excess class
        excess_brier - scalar, brier score for excess class
        """
        # Compute true class labels
        true_zero = to_np((y == 0) * 1.)
        true_excess = to_np((y > threshes) * 1.)
        true_moderate = to_np(1 - true_zero - true_excess)

        # Compute predicted probabilities
        pred_zero, pred_moderate, pred_excess = self.compute_all_probs(pred, threshes, aslist=True)

        zero_brier = util.brier_score(true_zero, pred_zero)
        moderate_brier = util.brier_score(true_moderate, pred_moderate)
        excess_brier = util.brier_score(true_excess, pred_excess)
        return zero_brier, moderate_brier, excess_brier

    def compute_all_probs(self, pred, threshes, aslist):
        """
        Compute the predicted probability of each of the 3 classes (zero, non-zero non-excess, and excess)
        Parameters:
        pred - list of tensors, list of mixture model parameters
        threshes - tensor, thresholds
        aslist - boolean, if True returns results as a list of arrays if False stacks the arrays into one large array
        """
        pred_zero = self.compute_zero_prob(pred)
        pred_excess = self.compute_excess_prob(pred, threshes)
        pred_moderate = self.compute_moderate_prob(pred, threshes)
        if aslist:
            return pred_zero, pred_moderate, pred_excess
        else:
            return np.stack([pred_zero, pred_moderate, pred_excess], axis=0)

    def compute_class(self, pred, threshes):
        """
        Determines the predicted class (zero, non-zero non-excess, or excess)
        Parameters:
        pred - list of tensors, list of mixture model parameters
        threshes - tensor, thresholds

        Returns:
        labels - array, predicted class labels
        """
        probs = self.compute_all_probs(pred, threshes, aslist=False)
        labels = np.argmax(probs, axis=0)
        return labels

    def compute_class_metrics(self, y, pred, threshes):
        """
        Computes various metrics related to the predicted class labels
        Parameters:
        y - tensor, target variable
        pred - list of tensors, list of mixture model parameters
        threshes - tensor, thresholds

        Returns:
        zero_brier - scalar, brier score of 0 rainfall class
        moderate_brier - scalar, brier score for non-zero non-excess class
        excess_brier - scalar, brier score for excess class
        acc - scalar, accuracy
        f1_micro - scalar, f1 micro of all classes
        f1_macro - scalar, f1 macro of all classes
        auc_macro_ovo - scalar, auc macro one versus one
        auc_macro_ovr - scalar, auc macro one versus all
        """
        pred_probs = self.compute_all_probs(pred, threshes, aslist=True)
        pred_labels = self.compute_class(pred, threshes)
        true_labels = util.compute_class_labels(y, threshes)

        acc = util.accuracy(true_labels, pred_labels)
        f1_micro, f1_macro = util.f1(true_labels, pred_labels)
        auc_macro_ovo, auc_macro_ovr = util.auc(true_labels, np.stack(pred_probs, axis=0))
        zero_brier, moderate_brier, excess_brier = self.compute_brier_scores(y, pred, threshes)
        return zero_brier, moderate_brier, excess_brier, acc, f1_micro, f1_macro, auc_macro_ovo, auc_macro_ovr

    def compute_zero_prob(self, pred):
        """
        Computes predicted probability of 0 rainfall
        Parameters:
        pred - list of tensors, list of mixture model parameters

        Returns:
        bin_pred - array, probability of 0 rainfall
        """
        bin_pred, gpd_pred, norm_pred = self.split_pred(pred)
        return to_np(bin_pred[:, 0])

    def compute_nonzero_prob(self, pred):
        """
        Computes predicted probability of non-zero rainfall
        Parameters:
        pred - list of tensors, list of mixture model parameters

        Returns:
        bin_pred - array, probability of non-zero rainfall
        """
        return to_np(1 - self.compute_zero_prob(pred))

    def compute_moderate_prob(self, pred, threshes):
        """
        Computes predicted probability of non-zero non-excess rainfall
        Parameters:
        pred - list of tensors, list of mixture model parameters

        Returns:
        bin_pred - array, probability of non-zero non-excess rainfall
        """
        return to_np(1 - self.compute_zero_prob(pred) - self.compute_excess_prob(pred, threshes))

    def compute_excess_prob(self, pred, threshes):
        """
        Computes predicted probability of excess rainfall
        Parameters:
        pred - list of tensors, list of mixture model parameters

        Returns:
        bin_pred - array, probability of excess rainfall
        """
        bin_pred, gpd_pred, norm_pred = self.split_pred(pred)
        return to_np((1 - all_cdf(threshes, gpd_pred, norm_pred, bin_pred[:, 0], bin_pred[:, 1],
                                  self.effective_thresh(threshes), threshes, self.moderate_func)))

    def compute_loglik(self, y, pred, threshes):
        """
        Computes log-likelihood
        Parameters:
        y - tensor, target variable
        pred - list of tensors, list of mixture model parameters
        threshes - tensor, thresholds
        """
        bin_pred, gpd_pred, norm_pred = self.split_pred(pred)
        return loglik(
            y[:, 0], gpd_pred, norm_pred, bin_pred[:, 0], bin_pred[:, 1], self.effective_thresh(threshes), self.moderate_func
        )

    def compute_rmse(self, y, pred, threshes):
        """
        Computes MSE of point predictions
        Parameters:
        y - tensor, target variable
        pred - list of tensors, list of mixture model parameters
        threshes - tensor, thresholds
        """
        bin_pred, gpd_pred, moderate_pred = self.split_pred(pred)
        point_pred = all_mean(gpd_pred, moderate_pred, bin_pred[:, 0], bin_pred[:, 1],
                              self.effective_thresh(threshes), self.moderate_func)
        if torch.isnan(point_pred).any():
            print("nan here")
        return torch_rmse(y, point_pred)

    def _mc_forwards(self, x, threshes, n_forwards, test=False):
        """
        Compute n_forwards forward passes through network with dropout and returns stacked predictions.
        Used for Vandal et al
        """
        results = list()
        for _ in range(n_forwards):
            bin_pred, gpd_pred, lognorm_pred = self.compute_stats(x, threshes, test)
            results.append(torch.cat([bin_pred, lognorm_pred], axis=1))
        preds = torch.stack(results, axis=0)
        return preds

    def _first_moment(self, non_zeros, mus, sigs):
        """
        Computes first moment for mc dropout w/ zero inflated lognormal
        Used for Vandal et al
        """
        return torch.mean(non_zeros * torch.exp(mus + 0.5 * sigs ** 2), axis=0)

    def _second_moment(self, non_zeros, mus, sigs):
        """
        Computes second moment for mc dropout w/ zero inflated lognormal
        Used for Vandal et al
        """
        return torch.mean(non_zeros ** 2 * torch.exp(2 * mus + 2 * sigs ** 2), axis=0)

    def _get_sig(self, first_moment, second_moment):
        """
        Computes sigma of zero inflated lognormal from first two moments
        Used for Vandal et al
        """
        return torch.log(1 + 0.5 * (4 * second_moment / first_moment ** 2 + 1) ** 0.5)

    def _get_mu(self, first_moment, sig):
        """
        Computes mu of zero inflated lognormal from first two moments
        Used for Vandal et al
        """
        return first_moment - sig ** 2 / 2

    def compute_mc_stats(self, x, threshes, n_forwards, test=False):
        """
        Computes mus and sigmas of zero-inflated lognormal
        Used for Vandal et al
        """
        preds = self._mc_forwards(x, threshes, n_forwards, test)
        non_zeros = 1 - preds[:, :, 0]
        mus = preds[:, :, 2]
        sigs = preds[:, :, 3] ** 0.5
        first_m = self._first_moment(non_zeros, mus, sigs)
        second_m = self._second_moment(non_zeros, mus, sigs)
        final_sigs = self._get_sig(first_m, second_m)
        final_mus = self._get_mu(first_m, final_sigs)

        bin_pred_avgs = torch.mean(preds[:, :, 0:2], axis=0)
        lognormal_stats = torch.stack([final_mus, final_sigs ** 2], axis=1)
        return bin_pred_avgs, torch.zeros_like(lognormal_stats) + 0.1, lognormal_stats


def make_cnn(ndim, hdim, odim, ksize, padding, use_mc, variable_thresh, bn, act=None):
    """
    Makes pytorch CNN model.
    Parameters:
    ndim - int, number of input dimensions
    hdim - int, CNN number of hidden dimensions
    odim - int, number of outputs per location
    ksize - tuple of ints, CNN kernel size
    padding - tuple of ints, CNN padding
    use_mc - boolean, if True use the Vandal et al approach
    variable_thresh - boolean, if True thresholds are randomized during training
    bn - boolean, if True use batch norm
    act - string, represents activation function. Must be either relu or tanh
    """
    if use_mc:
        assert not bn
        return CNNConcreteDropout(ndim, hdim, odim, ksize, padding)
    else:
        model = CNN(ndim, hdim, odim, ksize, padding, variable_thresh, bn, act)
    return model


def concrete_regulariser(model: nn.Module) -> nn.Module:
    """Adds ConcreteDropout regularisation functionality to a nn.Module.
    Parameters
    ----------
    model : nn.Module
        Model for which to calculate the ConcreteDropout regularisation.
    Returns
    -------
    model : nn.Module
        Model with additional functionality.
    """

    def regularisation(self) -> Tensor:
        """Calculates ConcreteDropout regularisation for each module.
        The total ConcreteDropout can be calculated by iterating through
        each module in the model and accumulating the regularisation for
        each compatible layer.
        Returns
        -------
        Tensor
            Total ConcreteDropout regularisation.
        """

        total_regularisation = 0
        for module in filter(lambda x: isinstance(x, ConcreteDropout), self.modules()):
            total_regularisation += module.regularisation

        return total_regularisation

    setattr(model, 'regularisation', regularisation)

    return model


@concrete_regulariser
class CNN(nn.Module):

    def __init__(self, ndim, hdim, odim, ksize, padding, variable_thresh, bn, act=None) -> None:
        """
        Makes pytorch CNN model.
        Parameters:
        ndim - int, number of input dimensions
        hdim - int, CNN number of hidden dimensions
        odim - int, number of outputs per location
        ksize - tuple of ints, CNN kernel size
        padding - tuple of ints, CNN padding
        variable_thresh - boolean, if True thresholds are randomized during training
        bn - boolean, if True use batch norm
        act - string, represents activation function. Must be either relu or tanh
        """
        if act is None or act == 'relu':
            non_lin = torch.nn.ReLU
        elif act == 'tanh':
            non_lin = torch.nn.Tanh
        else:
            raise ValueError("Only relu and tanh supported")
        super().__init__()
        self.variable_thresh = variable_thresh
        self.cnns = torch.nn.Sequential(
            nn.BatchNorm3d(ndim) if bn else nn.Identity(),
            torch.nn.Conv3d(ndim, hdim, ksize, padding=padding),
            non_lin(),
            nn.BatchNorm3d(hdim) if bn else nn.Identity(),
            torch.nn.Conv3d(hdim, hdim, ksize, padding=padding),
            non_lin(),
            nn.BatchNorm3d(hdim) if bn else nn.Identity(),
            torch.nn.Conv3d(hdim, hdim, ksize, padding=padding),
            non_lin(),
            nn.BatchNorm3d(hdim) if bn else nn.Identity()
        )
        self.fcs = torch.nn.Sequential(
            torch.nn.Conv3d((hdim + 1) if variable_thresh else hdim, hdim, (1, 1, 1)),
            non_lin(),
            nn.BatchNorm3d(hdim) if bn else nn.Identity(),
            torch.nn.Conv3d(hdim, hdim, (1, 1, 1)),
            non_lin(),
            nn.BatchNorm3d(hdim) if bn else nn.Identity(),
            torch.nn.Conv3d(hdim, odim, (1, 1, 1))
        )

    def forward(self, x: torch.Tensor, test=False) -> Tensor:
        """
        Parameters:
        x - tensor, predictors
        test - ignore this. only added so it would have a signature that's the same as
               CNNConcreteDropout.
        """
        if self.variable_thresh:
            # The next line is hardcoded standardization. It assumes thresholds
            # are selected from a uniform distribution ranging from 0 to 0.95
            threshes = (x[:, -1:] - 2.79) / 2.26
            x = x[:, :-1]
        else:
            threshes = None
        out = self.cnns(x)
        if threshes is not None:
            out = torch.cat([out, threshes[:, :, -1:]], axis=1)
        out = self.fcs(out)
        return out


class ConcreteDropout(nn.Module):
    """
    Concrete Dropout.

    Implementation of the Concrete Dropout module as described in the
    'Concrete Dropout' paper: https://arxiv.org/pdf/1705.07832

    Taken from:
    https://github.com/danielkelshaw/ConcreteDropout/blob/master/condrop/concrete_dropout.py
    Modified to allow true dropout with the learned probabilities at test time following:
    https://github.com/tjvandal/discrete-continuous-bdl/blob/master/bdl.py
    """

    def __init__(self,
                 weight_regulariser: float,
                 dropout_regulariser: float,
                 init_min: float = 0.1,
                 init_max: float = 0.1) -> None:

        """
        Concrete Dropout.

        Parameters
        ----------
        weight_regulariser : float
            Weight regulariser term.
        dropout_regulariser : float
            Dropout regulariser term.
        init_min : float
            Initial min value.
        init_max : float
            Initial max value.
        """

        super().__init__()

        self.weight_regulariser = weight_regulariser
        self.dropout_regulariser = dropout_regulariser

        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.p = torch.sigmoid(self.p_logit)

        self.regularisation = 0.0

    def forward(self, x: Tensor, layer: nn.Module, test: bool = False) -> Tensor:

        """
        Calculates the forward pass.

        The regularisation term for the layer is calculated and assigned to a
        class attribute - this can later be accessed to evaluate the loss.

        Parameters
        ----------
        x : Tensor
            Input to the Concrete Dropout.
        layer : nn.Module
            Layer for which to calculate the Concrete Dropout.
        test : bool
            Indicates whether to use true dropout (test=True) or to approximate it (test=False).
            Setting this to True should only be done at test/validation time hence the name.

        Returns
        -------
        Tensor
            Output from the dropout layer.
        """

        output = layer(self._concrete_dropout(x, test))

        sum_of_squares = 0
        for param in layer.parameters():
            sum_of_squares += torch.sum(torch.pow(param, 2))

        weights_reg = self.weight_regulariser * sum_of_squares / (1.0 - self.p)

        dropout_reg = self.p * torch.log(self.p)
        dropout_reg += (1.0 - self.p) * torch.log(1.0 - self.p)
        n, c, seq_len, h, w = x.shape
        dropout_reg *= self.dropout_regulariser * 27 * c / n  # not sure what to scale by here

        self.regularisation = weights_reg + dropout_reg

        return output

    def _concrete_dropout(self, x: Tensor, test: bool = False) -> Tensor:

        """
        Computes the Concrete Dropout.

        Parameters
        ----------
        x : Tensor
            Input tensor to the Concrete Dropout layer.
        test : bool
            Indicates whether to use true dropout (test=True) or to approximate it (test=False).
            Setting this to True should only be done at test/validation time hence the name.

        Returns
        -------
        Tensor
            Outputs from Concrete Dropout.
        """

        eps = 1e-7
        tmp = 0.1

        self.p = torch.sigmoid(self.p_logit)
        if test:  # At test time we perform actual dropout rather than its concrete approximation
            x = F.dropout3d(x, self.p.item(), training=True)    # keep training True to keep dropout on
        else:
            u_noise = torch.rand_like(x)

            drop_prob = (torch.log(self.p + eps) -
                         torch.log(1 - self.p + eps) +
                         torch.log(u_noise + eps) -
                         torch.log(1 - u_noise + eps))

            drop_prob = torch.sigmoid(drop_prob / tmp)

            random_tensor = 1 - drop_prob
            retain_prob = 1 - self.p

            x = torch.mul(x, random_tensor) / retain_prob

        return x


@concrete_regulariser
class CNNConcreteDropout(nn.Module):

    def __init__(self, ndim, hdim, odim, ksize, padding) -> None:
        """
        Makes pytorch CNN model.
        Parameters:
        ndim - int, number of input dimensions
        hdim - int, CNN number of hidden dimensions
        odim - int, number of outputs per location
        ksize - tuple of ints, CNN kernel size
        padding - tuple of ints, CNN padding
        """
        super().__init__()

        self.cnn1 = torch.nn.Conv3d(ndim, hdim, ksize, padding=padding)
        self.cnn2 = torch.nn.Conv3d(hdim, hdim, ksize, padding=padding)
        self.cnn3 = torch.nn.Conv3d(hdim, hdim, ksize, padding=padding)
        self.linear = torch.nn.Conv3d(hdim, odim, (1, 1, 1))

        w, d = 1e-6, 1e-3
        self.cd1 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.cd2 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.cd3 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.cd4 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, test=False) -> Tensor:
        """
        Parameters:
        x - tensor, predictors
        test - boolean, True at test time. At test time we need to use True drop out rather
               than an approximation
        """
        x = self.cd1(x, nn.Sequential(self.cnn1, self.relu), test)
        x = self.cd2(x, nn.Sequential(self.cnn2, self.relu), test)
        x = self.cd3(x, nn.Sequential(self.cnn3, self.relu), test)
        x = self.cd4(x, self.linear, test)

        return x


def get_device():
    """
    Determines whether to use cuda or cpu for tensors
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def set_default_tensor_type():
    """
    In my experience I run in to numerical issues w/ float
    but I haven't experimented with the data type in a while
    """
    torch.set_default_tensor_type(torch.DoubleTensor)
    return torch.double


def gp_mean(xi, sigma, thresh, eps=1e-4):
    """
    Given xi, sigma, and a threshold it computes the mean of
    a generalized Pareto distribution.
    """
    return thresh + sigma / (1 - xi)


def trunc_lognorm_mean(mu, var, upper, eps=1e-6):
    """
    Computes the mean of a truncated lognormal. Note that there's
    a couple different parameterizations of the lognormal distribution.
    This code is based on the parameterization here:
    https://en.wikipedia.org/wiki/Log-normal_distribution
    Parameters:
    mu - tensor, first lognormal parameter
    var - tensor, second lognormal parameter (variance)
    upper - tensor, threshold that defines right edge of distribution
    """
    beta = (torch.log(upper) - mu) / var ** 0.5
    sigma = var ** 0.5
    scaling_factor = norm_cdf(beta - sigma, torch.zeros_like(beta), torch.ones_like(beta)) / (
            eps + norm_cdf(beta, torch.zeros_like(beta), torch.ones_like(beta)))
    exp = torch.exp(mu + var / 2)
    return exp * scaling_factor


def norm_cdf(vals, mu, sigma):
    """
    Computes cdf of normal distribution
    Parameters:
    vals - tensor, where to evaluate cdf
    mu - tensor, mean of distribution
    sigma - tensor, standard deviation
    """
    return 0.5 * (1 + torch.erf((vals - mu) * sigma.reciprocal() / math.sqrt(2)))


def torch_rmse(w_nans_l, w_nans_r):
    """
    Computes RMSE between two tensors while ignoring nans while preserving gradients
    """
    denan_l = torch.zeros_like(w_nans_l, device=get_device())
    denan_r = torch.zeros_like(w_nans_r, device=get_device())
    nonan_mask = ~torch.isnan(w_nans_l + w_nans_r)
    denan_l[nonan_mask] += w_nans_l[nonan_mask]
    denan_r[nonan_mask] += w_nans_r[nonan_mask]
    return torch.sqrt(torch.mean(((denan_l - denan_r) ** 2)[nonan_mask]))


def lognorm_mean(mu, var):
    """
    Computes mean of lognormal distribution
    This code is based on the parameterization here:
    https://en.wikipedia.org/wiki/Log-normal_distribution

    Parameters:
    mu - tensor, first lognormal parameter
    var - tensor, second lognormal parameter (variance)
    """
    return torch.exp(mu + var / 2)


def all_mean(gpd_stats, moderate_stats, zero_probs, excess_probs, threshes, moderate_func):
    """
    Computes the mean of the mixture model. By setting excess probs to 0 and threshes to 99999999 it ignores gpd component
    gpd_stats - tensor, statistics of gpd distribution: gpd_stats[:, 0] is xi and gpd_stats[:, 1] is sigma
    moderate_stats - tensor, statistics of lognormal distribution moderate_stats[:, 0] is mu and moderate_stats[:, 1] is variance
    zero_probs - tensor, probability of zero rainfall
    excess_probs - tensor, probability of excess rainfall given non-zero
    threshes - tensor, threshold defining excess vs non-excess value
    """
    weighted_zero = 0.  # weighted mean of zero rainfall component
    # Compute weighted mean of lognormal component of the model
    if moderate_func == 'lognormal':
        weighted_moderate = trunc_lognorm_mean(moderate_stats[:, 0], moderate_stats[:, 1], threshes)
        weighted_moderate *= (1 - zero_probs) * (1 - excess_probs)
    else:
        raise ValueError('only lognormal function is supported for mean calculations')
    # Compute weighted mean of gpd component if we are using EVT
    if torch.all(torch.isinf(threshes)):
        weighted_excess = 0.     # we are not using EVT
    else:
        weighted_excess = gp_mean(gpd_stats[:, 0], gpd_stats[:, 1], threshes)
        weighted_excess *= (1 - zero_probs) * excess_probs

    # return weighted mean
    return weighted_zero + weighted_moderate + weighted_excess


def prob_constraint(k, eps=0.01):
    """
    Given unconstrained input returns output satisfying 0 < output < 1
    """
    return torch.sigmoid(k) * (1 - 2 * eps) + eps


def pos_constraint(k, func='exp', eps=1e-2):
    """
    Given unconstrained input ensures the output is positive. This can be accomplished with 1 of 3
    different functions as determined by the 'func' parameter:
        exponential (exp), absolute value (abs), squaring (square).
    """
    if func == 'exp':
        pos = torch.exp(k)
    elif func == 'abs':
        pos = torch.abs(k)
    elif func == 'square':
        pos = torch.square(k)
    else:
        raise ValueError('unsupported positivity enforcement function')
    return pos + eps


def upper_thresh_constraint(x, thresh, constrain_positive, beta):
    """
    Constrains the input to asymptotically approach a threshold without surpassing it

    Parameters:
    x - tensor, raw input to be constrained
    thresh - scalar, threshold
    constrain_positive - boolean, setting to true further constrains input to be positive
    beta - scalar, a parameter of the softplus function that I generally set to 10 in practice so that
           the softplus function behaves nicely (i.e. monotonically increasing if I remember correctly)
           but it's no big deal if set to 1.

    Returns:
    tensor, constrained input
    """
    if constrain_positive:
        out = pos_constraint(x)
    else:
        out = x
    return -torch.nn.functional.softplus(-out + thresh, beta) + thresh


def lognormal_constraint(mu, var):
    """
    Enforces constraints on the lognormal parameters
    Parameters:
    mu - tensor, not constrained
    var - tensor, the variance. Constrained to be positive and < 30. Constraining < 30 may
          no longer be necessary.

    Returns:
    lognormal_stats - tensor, lognormal_stats[:, 0] is mu and lognormal_stats[:, 1] is variance
    """
    lognormal_stats = torch.stack([mu, upper_thresh_constraint(var, 30, True, 1)], axis=1)
    return lognormal_stats


def gp_constraint(k1, k2, maxis, eps=1e-6):
    """
    This is enforces the generalized Pareto constraints. This is the old approach from the AAAI paper
    Parameters:
    k1 - tensor, unconstrained neural net output
    k2 - tensor, unconstrained neural net output
    maxis - tensor, maximum value the gpd must assign non-zero probability

    Returns:
    gpd_stats - tensor, xi is gpd_stats[:, 0] and sigma is gpd_stats[:, 1]
    """
    k1, sigma = pos_constraint(k1), pos_constraint(k2)
    xi = k1 - sigma / (maxis + eps)

    gpd_stats = torch.stack([xi, sigma], axis=1)
    return gpd_stats


def gp_constraint2(k1, k2, maxis, continuous_evt, initial_density=None, eps=1e-6):
    """
    This is enforces the generalized Pareto constraints usin gthe new approach

    Parameters:
    k1 - tensor, unconstrained neural net output
    k2 - tensor, unconstrained neural net output
    maxis - tensor, maximum value the gpd must assign non-zero probability
    continuous_evt - boolean, if True forces the GPD to be continuous w/ the lognormal function
    initial_density - tensor, this is the density of the lognormal distribution at 0

    Returns:
    gpd_stats - tensor, xi is gpd_stats[:, 0] and sigma is gpd_stats[:, 1]
    """
    k1 = pos_constraint(k1)
    if continuous_evt:
        assert not initial_density is None
        sigma = 1 / initial_density
    else:
        sigma = upper_thresh_constraint(k2, 40, True, 1)
    xi = (k1 - 1) * sigma / (maxis + eps)

    # Stacks xi and sigma. Xi is constrained to be < 0.9
    gpd_stats = torch.stack([gp_upper_thresh_constraint(xi, 0.9, 5), sigma], axis=1)
    return gpd_stats


def gp_upper_thresh_constraint(x, thresh, beta):
    """
    takes input x which is predicted xi values from gpd and thresholds them w/ upper bound
    when thresholding it ensures that the negative values of xi aren't made more negative
    this ensures that the gp constraint won't be violated

    Big picture: when xi > 0 we take weighted average of input and soft-threshold output
       Assigns input weight of 1 when xi < 0 and converges to weight on ouput of 1 as xi increases
       Leaving xi unchanged when < 0 ensures no GP constraints are violated
       Using smoothly varying weighted average gaurantees functions is continuous and differentiable almost everywhere

    Parameters:
    x - tensor, input xi values to be constrained
    thresh - scalar, threshold that xi should be less than
    beta - scalar, a parameter that doesn't matter too much. I generally set to 10

    Returns:
    tensor, constrained xi values
    """

    # Here we compute the thresholded output
    upper_threshed_output = upper_thresh_constraint(x, thresh, False, beta)

    # Next we compute the weight for weighted average to ensure it has the desired properties
    output_weight = x / thresh
    output_weight = -1 * nn.functional.threshold(-1 * nn.functional.threshold(output_weight, 0, 0), -1, -1)

    return output_weight * upper_threshed_output + (1 - output_weight) * x


def all_constraints(binaries, gpd_stats, main_stats, maxis, continuous_evt, main_func='lognormal', thresholds=None):
    """
    Enforces all constraints on mixture model parameters
    Parameters:
    binaries - tensor, tensor raw neural net output to be converted into probabilities of
               zero rainfall and probability of excess
    gpd_stats - tensor, raw neural net output to be converted to gpd statistical parameters
    main_stats - tensor, raw neural net output to be converted to lognormal parmaters
    maxis - tensor, max value which gpd must assign non-zero probability
    continuous_evt - boolean, if True the mixture model must be continuous at the threshold
    main_func - string, what density function to use for non-excess values. Must be lognormal
    thresholds - tensor, the threshold between non-excess and excess values

    Returns:
    binaries - tensor, constrained probability of zero (binaries[:, 0]) and excess (binaries[:, 1]) rainfall
    gpd_stats - tensor, constrained gpd parameters xi (gpd_stats[:, 0]) and sigma (gpd_stats[:, 1])
    """
    binaries = prob_constraint(binaries)
    if main_func == 'lognormal':
        main_stats = lognormal_constraint(main_stats[:, 0], main_stats[:, 1])
    else:
        raise ValueError('Only lognormal distribution is supported for non-excess values.')

    # This deals with the case where we want the mixture model to be continuous
    if continuous_evt:
        assert main_func == 'lognormal'
        assert not thresholds is None
        initial_densisty = 1 / torch.exp(lognormal(thresholds, main_stats[:, 0], main_stats[:, 1], reduction=None))
    else:
        initial_densisty = None

    gpd_stats = gp_constraint2(gpd_stats[:, 0], gpd_stats[:, 1], maxis, continuous_evt,
                               initial_density=initial_densisty)
    return binaries, gpd_stats, main_stats


def to_tensor(a):
    """
    Converts a numpy array or scalar to a tensor
    """
    device = get_device()
    if 'int' in str(type(a)) or 'float' in str(type(a)):
        return torch.tensor(np.array([a]), device=device)
    elif 'tuple' in str(type(a)):
        return tuple(torch.tensor(np.array([x]), device=device) for x in a)
    else:
        return torch.tensor(a, device=device)


def to_np(a):
    """
    Converts a tensor or list of tensor to a numpy array or list of numpy arrays respectively 
    """
    if "torch" in str(type(a)):
        return a.cpu().detach().numpy()
    if ("list" in str(type(a)) or "tuple" in str(type(a))) and "torch" in str(type(a[0])):
        return [item.cpu().detach().numpy() for item in a]
    else:
        return a


def to_item(a):
    """
    Converts a single-element (SE) tensor or list of SE tensor or SE numpy array or list of SE numpy array to a float
    """
    if "list" in str(type(a)):
        return [x.item() for x in a]
    elif "tuple" in str(type(a)):
        return tuple(x.item() for x in a)
    else:
        return a.item()


def gpd(raw_samples, xi, sigma):
    """
    Computes log-likelihood of gpd distribution

    Parameters:
    raw_samples - tensor, excesses over threshold
    xi - tensor, shape parameter
    sigma - tensor, scale parameter

    Returns:
    out - tensor, log-likelihood of each excess value
    """
    samples = torch.zeros_like(raw_samples) + raw_samples  # Create dummy array to ensure gradients flow correctly
    mask = ~torch.isnan(raw_samples)  # create a mask for non-nan samples
    alt_mask = ~torch.isnan(xi)  # create a mask for non-nan shape values
    mask = mask & alt_mask
    samples[~mask] = 0
    xi_is_zero = (xi == 0)
    """
    Case where xi != 0
    """
    xi_nz = torch.log(sigma[~xi_is_zero]) + (1 + 1 / xi[~xi_is_zero]) * mask[~xi_is_zero] * torch.log(
        1 + xi[~xi_is_zero] * samples[~xi_is_zero] / sigma[~xi_is_zero])
    # change from negative log-likelihood to log-likelihood
    xi_nz *= -1.

    """
    Case where xi = 0
    """
    xi_z = torch.log(sigma[xi_is_zero]) + (1 / sigma[xi_is_zero]) * samples[xi_is_zero]
    # change from negative log-likelihood to log-likelihood
    xi_z *= -1

    # Have to create a dummy tensor to ensure gradients are computed correctly
    out = torch.zeros_like(samples)
    out[xi_is_zero] += xi_z
    out[~xi_is_zero] += xi_nz
    out[~mask] = np.nan

    return out


def true_gpd(samps, xis, sigmas):
    """
    Computes gpd log likelihood w/ scipy. Strictly for debugging
    """
    logliks = list()
    for i in range(samps.shape[0]):
        logliks.append(genpareto.logpdf(samps[i, :], xis[i], scale=sigmas[i]))
    return np.concatenate(logliks)


def torch_nanmean(vals):
    """
    Torch version of np.nanmean
    """
    inds = ~torch.isnan(vals)
    return torch.mean(vals[inds])


def lognorm_cdf(vals, mu, var):
    """
    Computes cdf of lognormal function. Note that there's
    a couple different parameterizations of the lognormal distribution.
    This code is based on the parameterization here:
    https://en.wikipedia.org/wiki/Log-normal_distribution
    Parameters:
    vals - tensor, samples
    mu - tensor, mu parameter
    var - tensor, variance parameter
    """
    return 0.5 + 0.5 * torch.erf((torch.log(vals) - mu) / (var ** 0.5 * math.sqrt(2)))


def lognormal(samples, mu, var):
    """
    Lognormal distribution log likelihood function. Note that there's
    a couple different parameterizations of the lognormal distribution.
    This code is based on the parameterization here:
    https://en.wikipedia.org/wiki/Log-normal_distribution
    Parameters:
    samples - tensor, samples
    mu - tensor, mu parameter
    var - tensor, variance parameter
    """
    samples[samples == 0] += 10  # this is a janky way of avoiding nans. You can add any positive value here
    # without affecting the computation.
    first_term = -torch.log(samples) - 0.5 * torch.log(var) - 0.5 * torch.log(to_tensor(2 * 3.14159274))
    second_term = -((torch.log(samples) - mu) ** 2 / (2 * var))
    return first_term + second_term


def threshed_lognorm(samples, threshes, mu, var, eps=1e-6):
    """
    Thresholded lognormal distribution log likelihood function. Note that there's
    a couple different parameterizations of the lognormal distribution.
    This code is based on the parameterization here:
    https://en.wikipedia.org/wiki/Log-normal_distribution
    Parameters:
    samples - tensor, samples
    threshes - tensor, upper threshold
    mu - tensor, mu parameter
    var - tensor, variance parameter
    """
    return lognormal(samples, mu, var) - torch.log(lognorm_cdf(threshes, mu, var) + eps)


def threshed_lognorm_cdf(vals, mu, var, lower=None, upper=None):
    """
    Thresholded lognormal distribution cdf. Note that there's
    a couple different parameterizations of the lognormal distribution.
    This code is based on the parameterization here:
    https://en.wikipedia.org/wiki/Log-normal_distribution
    Parameters:
    samples - tensor, samples
    mu - tensor, mu parameter
    var - tensor, variance parameter
    upper - tensor, upper threshold
    lower - tensor, lower threshold
    """
    if upper is None:
        upper = 99999999.
        upper_cdf = 1
    else:
        upper_cdf = lognorm_cdf(upper, mu, var)
    if lower is None:
        lower = -99999999.
        lower_cdf = 0.
    else:
        lower_cdf = lognorm_cdf(lower, mu, var)
    denom = upper_cdf - lower_cdf
    out = lognorm_cdf(vals, mu, var) / denom
    out[vals < lower] = 0
    out[vals > upper] = 1
    out[denom == 0] = 0
    return out


def gpd_cdf(vals, xi, sigma, thresh=0.):
    """
    Computes cdf of GPD
    """
    return 1 - (1 + xi * (vals - thresh) / sigma) ** (-1 / xi)


def all_cdf(samples, gpd_stats, moderate_stats, zero_probs, excess_probs, effective_threshes, actual_threshes,
            moderate_func):
    """
    Computes the cdf of the mixture model
    Parameters:
    samples - tensor, samples
    gpd_stats - tensor, gpd_stats
    moderate_stats - tensor, lognormal distribution stats
    zero_probs - tensor, probability of 0
    excess_probs - tensor, probability of excess given non-zero
    effective_threshes - tensor, threshold used internally by the model. This will be the same as actual_threshes
                         for DeepGPD but very large for the hurdle baseline which lacks EVT
    actual threshes - tensor, actual threshold which defines extreme values -- effectively ignored by Hurdle baseline
    moderate_func - string, determines which density function is used for non-extreme values. Must be 'lognormal'
    """
    out = torch.zeros_like(samples)
    nz_inds = samples > 0
    excess_inds = samples > effective_threshes
    out[samples >= 0] += zero_probs[samples >= 0]
    if moderate_func == 'lognormal':
        out[nz_inds] += (1 - zero_probs[nz_inds]) * (1 - excess_probs[nz_inds]) * threshed_lognorm_cdf(samples[nz_inds],
                                                                                                       moderate_stats[
                                                                                                           nz_inds, 0],
                                                                                                       moderate_stats[
                                                                                                           nz_inds, 1],
                                                                                                       upper=
                                                                                                       effective_threshes[
                                                                                                           nz_inds])
    else:
        raise ValueError('only lognormal function is supported for non-excess values')
    out[excess_inds] += (1 - zero_probs[excess_inds]) * excess_probs[excess_inds] * gpd_cdf(samples[excess_inds],
                                                                                            gpd_stats[excess_inds, 0],
                                                                                            gpd_stats[excess_inds, 1],
                                                                                            thresh=actual_threshes[
                                                                                                excess_inds])
    return out


def loglik_zero(samples, zero_probs):
    """
    Computes log-likelihood of the mixture model's first component which governs probability of 0 rainfall
    Parameters:
    samples - tensor, samples
    zero_probs - tensor, predicted probability of zero rainfall
    """
    z_bool = is_zero(samples, False)  # binary tensor that's 1 if no rainfall
    nz_bool = is_zero(samples, True)  # binary tensor that's 0 if no rainfall
    out_zero = torch.zeros_like(samples) + z_bool * torch.log(zero_probs)  # log likelihood of 0 rainfall samples
    out_nonzero = torch.zeros_like(samples) + nz_bool * (
        torch.log(1 - zero_probs))  # log likelihood of nonzero rainfall samples
    nan_inds = torch.isnan(samples)
    out_zero[nan_inds] += np.nan
    out_nonzero[nan_inds] += np.nan
    return out_zero, out_nonzero


def loglik_above_thresh(samples, threshes, above_thresh_probs):
    """
    Computes log-likelihood contributed by the boolean probability excess rainfall given non-zero rainfall
    Parameters:
    samples - tensor, samples
    threshes - tensor, thresholds defining transition from non-excess to excess
    above_thresh_probs - tensor, predicted probability of excess rainfall given non-zero rainfall
    """
    t_bool = is_above_thresh(samples, threshes, False)  # binary tensor that's 1 if excess and non-zero
    nt_bool = is_above_thresh(samples, threshes, True)  # binary tensor that's 1 if non-excess and non-zero
    return t_bool * torch.log(above_thresh_probs), nt_bool * torch.log(1 - above_thresh_probs)


def is_above_thresh(samples, threshes, flip):
    """
    Returns binary tensor that indicates if samples are excess or not
    Parameters:
    samples - tensor, samples
    threshes - tensor, threshold between non-excess and excess
    flip - if False excess values are 1 and everything else 0
           if True non-zero non-excess values are 1 and everything else 0
    """
    if flip:
        return (samples < threshes) & (samples > 0)
    else:
        return (samples >= threshes) & (samples > 0)


def is_zero(samples, flip):
    """
    Returns binary tensor that indicates if samples are 0 or not
    Parameters:
    samples - tensor, samples
    flip - if False all 0 sample values return 1 and everything else 0
           if True all non-zero samples return 1 and everything else 0
    """
    if flip:
        return (samples != 0) & (~torch.isnan(samples))
    else:
        return (samples == 0)


def nan_to_num(x, fill_val=0.):
    """
    Replaces all nans w/ specified fill value
    """
    x[torch.isnan(x)] = fill_val
    return x


def nan_transfer(w_nan, wo_nan):
    """
    Ensures that any indices where there's a nan in w_nan have a nan in wo_nan too.
    """
    out = torch.zeros_like(wo_nan)
    out += wo_nan
    out[torch.isnan(w_nan)] += np.nan
    return out


def loglik(samples, gpd_stats, moderate_stats, zero_probs, excess_probs, threshes, moderate_func):
    """
    Computes log-likelihood of mixture model
    Parameters:
    samples - tensor, samples
    gpd_stats - tensor, gpd statistics (gpd_stats[:, 0] is xi, gpd_stats[:, 1] is sigma)
    moderate_stats - tensor, lognormal statistics (moderate_stats[:, 0] is mu, moderate_stats[:, 1] is variance)
    zero_probs - tensor, probability of zero rainfall
    excess_probs - tensor, probability of excess rainfall given non-zero
    threshes - tensor, threshold between excess and non-excess
    moderate_func - string, determines density function governing non-excess values. Must be 'lognormal'
    """
    nan_inds = torch.isnan(samples)  # remember which samples are nan
    zero_loglik, nz_loglik = loglik_zero(samples, zero_probs)
    thresh_loglik, nthresh_loglik = loglik_above_thresh(samples, threshes, excess_probs)

    # Create tensor w/ just the excesses and all other entries nan
    excesses = torch.zeros_like(samples)
    excess_inds = samples > threshes
    excesses[excess_inds] += samples[excess_inds] - threshes[excess_inds]
    excesses[~excess_inds] /= 0
    # Compute log-likelihood contributed by excess values
    excess_loglik = gpd(excesses, gpd_stats[:, 0], gpd_stats[:, 1])

    # Create tensor w/ just the non-zero non-excess values
    nonzeros = torch.zeros_like(samples)
    nz_inds = (samples > 0) & ~(excess_inds)
    nonzeros[nz_inds] += samples[nz_inds]
    if moderate_func == 'lognormal':
        # Compute log-likelihood contributed by non-zero non-excess values
        main_loglik = threshed_lognorm(nonzeros, threshes,
                                       moderate_stats[:, 0],
                                       moderate_stats[:, 1])
    else:
        raise ValueError('only lognormal is supported for non-excess values')
    main_loglik[~nz_inds] = torch.zeros_like(main_loglik[~nz_inds]) / 0.  # Set all 0 values and excess values to nan

    zero_loglik, nz_loglik, nthresh_loglik, main_loglik, thresh_loglik, excess_loglik = \
        nan_transfer(samples, zero_loglik), \
        nan_transfer(samples, nz_loglik), \
        nan_transfer(samples, nthresh_loglik), \
        nan_transfer(samples, main_loglik),\
        nan_transfer(samples, thresh_loglik), \
        nan_transfer(samples, excess_loglik)  # Make sure nans are propogating correctly
    total_loglik = \
        nan_to_num(zero_loglik) + \
        nan_to_num(nz_loglik) + \
        nan_to_num(nthresh_loglik) + \
        nan_to_num(main_loglik) + \
        nan_to_num(thresh_loglik) +\
        nan_to_num(excess_loglik)  # Add up all log-likelihoods
    total_loglik[nan_inds] += np.nan  # Make sure nans are propogating correctly
    return total_loglik
