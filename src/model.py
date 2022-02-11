import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from util import *


class SpatiotemporalLightningModule(pl.LightningModule):

    def __init__(self, st_params, model_params, seed, lr, n_epoch):
        super().__init__()
        self.save_hyperparameters()
        self.st_params = st_params
        self.model_params = model_params
        self.seed = seed
        self.lr = lr
        self.n_epoch = n_epoch

        # build model backbone and spatiotemporal wrapper
        if st_params["backbone"] == "cnn":
            model = make_cnn(**model_params).to(get_device())
        elif st_params["backbone"] == "ding":
            model = ExtremeTime2(**model_params)
        else:
            raise ValueError()
        self.st_model = SpatiotemporalModel(model=model, **st_params).to(get_device())
        pl.seed_everything(self.seed)

    def training_step(self, batch, batch_idx):
        self.train()
        x = batch["x"].type(torch.FloatTensor).to(self.device)
        y = batch["y"].type(torch.FloatTensor).to(self.device)

        # choose threshold
        if self.st_model.variable_thresh:
            # generate random thresholds in [0.5, 0.95] and augment predictors
            threshes = 0.45 * torch.rand_like(y) + 0.5
            x = torch.cat([x, threshes[:, np.newaxis].repeat(1, 1, x.shape[2], 1, 1)], axis=1)
        else:
            # generate fixed threshold but do not augment predictors
            t = np.nanquantile(to_np(y), self.st_model.quantile)
            threshes = torch.ones_like(y) * t

        # apply appropriate forward pass (logic for each model type is handled in forward() definition
        pred = self.st_model(x, threshes, test=False)

        loss, nll_loss, rmse_loss = self.st_model.compute_losses(pred, y, threshes)
        self.log("t_loss", loss)
        self.log("t_nll_loss", nll_loss)
        self.log("t_rmse_loss", rmse_loss)  # t for train
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        x = batch["x"].type(torch.FloatTensor).to(self.device)
        y = batch["y"].type(torch.FloatTensor).to(self.device)

        # choose threshold
        if self.st_model.variable_thresh:
            # fix threshold at test time and augment predictors
            t = np.nanquantile(to_np(y), self.st_model.quantile)
            threshes = torch.ones_like(y) * t
            x = torch.cat([x, threshes[:, np.newaxis].repeat(1, 1, x.shape[2], 1, 1)], axis=1)
        else:
            # generate fixed threshold but do not augment predictors
            t = np.nanquantile(to_np(y), self.st_model.quantile)
            threshes = torch.ones_like(y) * t

        # apply appropriate forward pass (logic for each model type is handled in forward() definition
        pred = self.st_model(x, threshes, test=True)

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
                self.log(f"v_{metric_name}", metric_mean)  # v for validation

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        metric_names = outputs[0].keys()
        for metric_name in metric_names:
            metric_mean = np.mean([o[metric_name] for o in outputs])
            self.log(f"f_{metric_name}", metric_mean)  # f for final

    def configure_optimizers(self):
        return torch.optim.Adam(self.st_model.parameters(), lr=self.lr)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpatiotemporalModel(nn.Module):

    def __init__(self, model, use_evt, moderate_func, ymax, mean_multiplier, dropout_multiplier, continuous_evt,
                 variable_thresh, quantile, use_mc, mc_forwards, backbone, deterministic, ev_index):
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
        backbone - str, type of model used
        deterministic - bool, true for use with Vandal et al and Ding et al
        ev_index - float, extreme value index hyperparameter in Ding et al
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
        self.backbone = backbone
        self.deterministic = deterministic
        self.ev_index = ev_index
        if self.continuous_evt:
            assert use_evt
        self.ymax = ymax

    def forward(self, x, threshes, test=False):
        if self.use_mc:
            # output (n, 6, h, w) tensor of distribution parameters
            return self.compute_mc_stats(x, threshes, self.mc_forwards, test)
        elif self.deterministic and not self.use_evt:
            # output (n, 1, h, w) tensor of predicted values
            pred = self.model(x, test).squeeze(2)
            return torch.relu(pred)                 # in deterministic precipitation prediction, all values >= 0
        elif self.deterministic and self.use_evt:
            # output (n, 2, h, w) of (predicted values, predicted probability of excesses)
            pred = self.model(x, test)
            pred[:, 0] = torch.relu(pred[:, 0])         # all predicted values >= 0
            pred[:, 1] = torch.sigmoid(pred[:, 1])      # all predicted probabilities in [0, 1]
            return pred
        else:
            # output (n, 6, h, w) tensor of distribution parameters
            return self.compute_stats(x, threshes, test)

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
        bin_pred_, gpd_pred_, norm_pred_ = to_stats(cur_raw)  # This is just splitting the data
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
        if self.deterministic and not self.use_evt:
            # deterministic model loss
            # output (n, 1, h, w) tensor of predicted values
            rmse_loss = torch_rmse(y, pred)
            nll_loss = torch.zeros_like(rmse_loss)
        elif self.deterministic and self.use_evt:
            # Ding et al. model loss
            # output (n, 2, h, w) of (predicted values, predicted probability of excesses)
            point_pred, excess_pred = pred[:, [0]], pred[:, [1]]
            excess_true = 1. * (y > self.effective_thresh(threshes))
            rmse_loss = torch_rmse(y.squeeze(), point_pred)
            beta_0, beta_1 = self.quantile, 1 - self.quantile
            nll_loss = torch.nanmean(
                -beta_0 * (1 - excess_pred / self.ev_index)**self.ev_index * excess_true * torch.log(excess_pred) + \
                -beta_1 * (1 - (1 - excess_pred) / self.ev_index)**self.ev_index * (1 - excess_true) * torch.log(1 - excess_pred)
            )
        else:
            # probabilistic model loss
            # output (n, 6, h, w) tensor of distribution parameters
            rmse_loss = self.compute_rmse(y, pred, threshes)
            bin_pred, gpd_pred, norm_pred = self.split_pred(pred)
            # I'm pretty sure this line was necessary to gpd_pred later on for debugging purposes. I don't think
            # its needed now but decided to keep it just in case.
            gpd_pred = split_var(gpd_pred)
            nll_loss = -torch_nanmean(
                loglik(
                    y, gpd_pred, norm_pred, bin_pred[:, 0], bin_pred[:, 1], self.effective_thresh(threshes),
                    self.moderate_func
                )
            )

        loss = (1 - self.mean_multiplier) * nll_loss + self.mean_multiplier * rmse_loss + \
               (0 if (self.dropout_multiplier == 0) else (self.dropout_multiplier * self.model.regularisation()))
        return loss, nll_loss, rmse_loss

    def compute_point_pred(self, pred, threshes):
        if self.deterministic and not self.use_evt:
            # deterministic model
            # output (n, 1, h, w) tensor of predicted values
            point_pred = pred
        elif self.deterministic and self.use_evt:
            # Ding et al. model
            # output (n, 2, h, w) of (predicted values, predicted probability of excesses)
            point_pred, excess_pred = pred[:, [0]], pred[:, [1]]
        else:
            # probabilistic model
            # output (n, 6, h, w) of predicted parameters
            bin_pred, gpd_pred, moderate_pred = self.split_pred(pred)
            point_pred = all_mean(gpd_pred, moderate_pred, bin_pred[:, 0], bin_pred[:, 1],
                                  self.effective_thresh(threshes), self.moderate_func)
        return point_pred

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
        if self.deterministic and not self.use_evt:
            # deterministic model loss
            # output (n, 1, h, w) tensor of predicted values
            rmse_loss = torch_rmse(y, pred)
            nll_loss = torch.zeros_like(rmse_loss)
        elif self.deterministic and self.use_evt:
            # Ding et al. model loss
            # output (n, 2, h, w) of (predicted values, predicted probability of excesses)
            point_pred, excess_pred = pred[:, [0]], pred[:, [1]]
            excess_true = 1. * (y > self.effective_thresh(threshes))
            rmse_loss = torch_rmse(y.squeeze(), point_pred)
            beta_0, beta_1 = self.quantile, 1 - self.quantile
            nll_loss = torch.nanmean(
                -beta_0 * (1 - excess_pred / self.ev_index)**self.ev_index * excess_true * torch.log(excess_pred) + \
                -beta_1 * (1 - (1 - excess_pred) / self.ev_index)**self.ev_index * (1 - excess_true) * torch.log(1 - excess_pred)
            )
        else:
            # probabilistic model metrics
            rmse_loss = self.compute_rmse(y, pred, threshes)
            bin_pred, gpd_pred, norm_pred = self.split_pred(pred)
            nll_loss = -torch_nanmean(
                loglik(
                    y, gpd_pred, norm_pred, bin_pred[:, 0], bin_pred[:, 1], self.effective_thresh(threshes),
                    self.moderate_func
                )
            )

        zero_brier, moderate_brier, excess_brier, acc, f1_micro, f1_macro, auc_macro_ovo, auc_macro_ovr = \
            self.compute_class_metrics(y, pred, threshes)

        loss = nll_loss + self.mean_multiplier * rmse_loss + \
               (0 if (self.dropout_multiplier == 0) else (self.dropout_multiplier * self.model.regularisation()))
        return to_np(loss), to_np(nll_loss), to_np(
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

        zero_brier = brier_score(true_zero, pred_zero)
        moderate_brier = brier_score(true_moderate, pred_moderate)
        excess_brier = brier_score(true_excess, pred_excess)
        return zero_brier, moderate_brier, excess_brier

    def compute_all_probs(self, pred, threshes, aslist):
        """
        Compute the predicted probability of each of the 3 classes (zero, non-zero non-excess, and excess)
        Parameters:
        pred - list of tensors, list of mixture model parameters
        threshes - tensor, thresholds
        aslist - boolean, if True returns results as a list of arrays if False stacks the arrays into one large array
        """
        if self.deterministic and not self.use_evt:
            # compute hard estimates
            pred_zero = to_np(torch.where(pred == 0, 1, 0))
            pred_excess = to_np(torch.where(pred > threshes, 1, 0))
            pred_moderate = to_np(1 - pred_zero - pred_excess)
        elif self.deterministic and self.use_evt:
            # compute soft estimates
            point_pred, excess_pred = pred[:, [0]], pred[:, [1]]
            pred_zero = to_np(torch.where(point_pred == 0, 1, 0))
            pred_excess = to_np(torch.where(point_pred != 0, 1, 0) * excess_pred)
            pred_moderate = to_np(1 - pred_zero - pred_excess)
        else:
            # compute soft estimates
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
        # predicted classes are (0=zero, 1=nonzero-nonexcess, 2=excess)
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
        true_labels = compute_class_labels(y, threshes)

        acc = accuracy(true_labels, pred_labels)
        f1_micro, f1_macro = f1(true_labels, pred_labels)
        auc_macro_ovo, auc_macro_ovr = auc(true_labels, np.stack(pred_probs, axis=0))
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
            y[:, 0], gpd_pred, norm_pred, bin_pred[:, 0], bin_pred[:, 1], self.effective_thresh(threshes),
            self.moderate_func
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
            x = F.dropout3d(x, self.p.item(), training=True)  # keep training True to keep dropout on
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


@concrete_regulariser
class ExtremeTime2(nn.Module):
    """
    Implementation of Ding et al.
    Pulled ExtremeTime2 directly from https://github.com/tymefighter/Forecast/blob/master/ts/model/extreme_time2.py
    """

    def __init__(self, forecast_horizon, ndim, hdim, odim, spatial_dims, window_size, memory_dim, context_size):
        """
        Initialize the model parameters and hyperparameters
        :param forecast_horizon: How much further in the future the model has to
        predict the target series variable
        :param memory_dim: Size of the explicit memory unit used by the model, it
        should be a scalar value (CHANGED from 80 -> 7 for this task)
        :param window_size: Size of each window which is to be compressed and stored
        as a memory cell (CHANGED from 60 -> 7 for this task)
        :param hdim: Size of the hidden state of the GRU encoder
        :param context_size: Size of context produced from historical sequences (CHANGED from 10 -> 7 for this task)
        :param ndim: Number of exogenous variables the model takes as input
        this is for internal use only (i.e. it is an implementation detail)
        If True, then object is normally created, else object is created
        without any member values being created. This is used when model
        is created by the static load method
        """
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.memory_dim = memory_dim
        self.window_size = window_size
        self.ndim = ndim
        self.hdim = hdim
        self.odim = odim
        self.spatial_dims = spatial_dims
        self.context_dim = context_size
        self.memory = None
        self.context = None
        self.gru_input = nn.GRUCell(input_size=self.ndim, hidden_size=self.hdim)
        self.gru_memory = nn.GRUCell(input_size=self.ndim, hidden_size=self.hdim)
        self.gru_context = nn.GRUCell(input_size=self.ndim, hidden_size=self.context_dim)
        final_weight_size = self.hdim + self.context_dim
        self.linear = nn.Linear(in_features=final_weight_size, out_features=int(np.prod(odim)))
        self.b = nn.Linear(in_features=1, out_features=1, bias=False)

    def forward(self, x, test=False):
        """
        Forecast using the model parameters on the provided input data
        :param x: Input target variable with shape (b, n, ...)
        :param test: Ignore this. Added to match CNNConcreteDropout signature.
        :return: Forecast targets predicted by the model
        """
        reshaped = x.reshape((-1, self.ndim, self.window_size))     # (bhw, t, n)
        self.build_memory(reshaped)
        out = self.predict_timestep(reshaped)
        return out.reshape(-1, self.odim, 1, *self.spatial_dims)     # (b, n, t, h, w)

    def build_memory(self, x):
        """
        Build Model Memory using the timesteps seen up till now
        :param x: Features, has shape (n, self.inputShape)
        timestep earlier than the current timestep
        :return: None
        """
        self.memory = [None] * self.memory_dim
        self.context = [None] * self.memory_dim
        b, c, length = x.shape

        for i in range(self.memory_dim):
            tmax = np.random.randint(low=0, high=length)
            self.memory[i], self.context[i] = self.run_gru_on_window(x, tmax)

        self.memory = torch.stack(self.memory)
        self.context = torch.stack(self.context)

    def run_gru_on_window(self, x, tmax):
        """
        Runs GRU on the window and returns the final state
        :param x: Features, has shape (n, self.inputShape)
        :param tmax: Maximum timestep to run to in x
        :return: The final state after running on the window, it has shape (self.encoderStateSize,)
        """
        # apply to flattened (b, c, t=0, h, w) -> (b, chw)
        gru_memory_state = self.gru_memory(x[:, :, 0])
        gru_context_state = self.gru_context(x[:, :, 0])

        for t in range(1, tmax):
            gru_memory_state = self.gru_memory(x[:, :, t], gru_memory_state)
            gru_context_state = self.gru_context(x[:, :, t], gru_context_state)

        return gru_memory_state, gru_context_state

    def predict_timestep(self, x, current_time=-1):
        """
        Predict on a Single Timestep
        :param x: Features, has shape (b, n, ...)
        :param current_time: Current Timestep
        :return: The predicted value on current timestep and the next state
        """
        b, n, t = x.shape
        embedding = self.gru_input(x[:, :, current_time])           # (b, n, t) @ (b, , hd) -> (b, hd)
        attention_weights = self.compute_attention(embedding)       # (b, 1, t)
        con = self.context.permute(1, 0, 2)                         # (n, b, hd) -> (b, n, hd)
        weighted_context = (attention_weights @ con).squeeze()      # (b, 1, n) @ (b, n, hd) -> (b, 1, hd) -> (b, hd)
        concat_vector = torch.concat([embedding, weighted_context], dim=1)  # (b, hd) | (b, c) -> (b, hd+c)
        out = self.linear(concat_vector)                            # (b, hd+c) -> (b, cnhw)
        o_tilde, u = out[:, [0]], out[:, [1]]                     # o_tilde and u from paper
        o = o_tilde + self.b(u)                                     # b.T @ u from paper
        o_and_u = torch.concat([o, u], dim=1)
        o_and_u = o_and_u.reshape([b] + [self.odim])            # (b, cnhw) -> (b, c, n, h, w)
        return o_and_u

    def compute_attention(self, embedding):
        """
        Computes Attention Weights by taking softmax of the inner product
        between embedding of the input and the memory states
        :param embedding: Embedding of the input, it has shape (self.hdim,)
        :return: Attention Weight Values
        """
        mem = self.memory.permute(1, 0, 2)  # shape (n, b, h) -> (b, n, h)
        emb = embedding.unsqueeze(2)        # shape (b, h, 1)
        out = F.softmax(mem @ emb, dim=1)   # shape (b, n, 1) with softmax over time dimension (n)
        return out.permute(0, 2, 1)         # shape (b, 1, n) such that we may multiply with (b, n, h) feature matrix
