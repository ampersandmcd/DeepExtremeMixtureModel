import numpy as np
import torch
import utils
import modelling as m

from torch import nn
from concrete_dropout import concrete_regulariser, ConcreteDropout
from torch import Tensor


class STModel:
    def __init__(self, model, use_evt, moderate_func, ymax, mean_multiplier, dropout_multiplier, continuous_evt):
        """
        This is used for making and evaluating predictions
        Parameters:
        model - pytorch model, the pytorch model
        use_evt - boolean, whether or not to use extreme value theory
        moderate_func - string, what density function to use for non-zero non-excess values. Must be 'lognormal'
        ymax - scalar, the max y value that must be assigned non-zero probability by the mixture model
        mean_multiplier - scalar, the weight assigned to the MSE component of the loss function (the other component is NLK)
        dropout_multiplier - scalar, the weight for dropout regularization in Vandal et al
        continuous_evt - boolean, whether or not the mixture model's density must be continuous at 0. Setting to True
                         doesn't work well.
        """
        m.set_default_tensor_type()
        self.model = model
        self.mean_multiplier = mean_multiplier
        self.dropout_multiplier = dropout_multiplier
        self.moderate_func = moderate_func
        self.use_evt = use_evt
        self.continuous_evt = continuous_evt

        if self.continuous_evt: assert use_evt
        self.ymax = ymax
    
    def effective_thresh(self, threshes):
        """
        The effective threshold is the threshold to be used by the mixture model. If we're not using EVT
        then the threshold is basically infinite (all non-zero values are modeled by lognormal) so we
        set it to a large positive value in that case.
        Parameters:
        threshes - tensor, the actual thresholds
        """
        if self.use_evt: return threshes
        else:
            return torch.ones_like(threshes, device=m.get_device()) * 999999999.
        
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
        bin_pred_, gpd_pred_, norm_pred_ = utils.to_stats(cur_raw) # This is just splitting the data
        bin_pred, gpd_pred, norm_pred = m.all_constraints(
            bin_pred_, gpd_pred_, norm_pred_, torch.ones_like(gpd_pred_[:, 0], device=m.get_device())*self.ymax, self.continuous_evt, main_func=self.moderate_func, thresholds=threshes)
        # If we're not using evt then set probability of excess to 0. The weird stuff w/ relu was necessary
        # to make sure gradients weren't broken.
        if not self.use_evt:
            bin_pred = torch.cat( [bin_pred[:, 0:1], torch.relu(-1*torch.abs(bin_pred[:, 1:2]))], dim=1)
        return bin_pred, gpd_pred, norm_pred
    
    def pred_stats(self, x, threshes, do_mc_tru_dropout=False):
        """
        Makes predictions then converts raw predictions to constrained mixture model parameters.
        Parameters:
        x - tensor, predictors
        threshes - tensor, threshold
        do_mc_tru_dropout - boolean, set to True for Vandal et al at test time so we do true
                            dropout not approximate dropout
                            
        Returns:
        bin_pred - tensor, bin_pred[:, 0] is probability of 0 rainfall and bin_pred[:, 1] is probability of excess rainfall
                   given non-zero rainfall
        gpd_pred - tensor, GPD parameters -- gpd_pred[:, 0] is xi and gpd_pred[:, 1] is sigma
        norm_pred - tensor, lognormal parameters -- norm_pred[:, 0] is mu, norm_pred[:, 1] is variance
        """
        if do_mc_tru_dropout:
            cur_raw = self.model(x, test=True)
        else:
            cur_raw = self.model(x)
        if np.isnan(m.to_np(cur_raw)).any():
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
        Computes NLK, MSE, and their weighted sum.
        Parameters:
        pred - list of tensors, list of mixture model parameters
        y - tensor, target
        threshes - tensor, thresholds
        
        Returns:
        loss - tensor, this is the weighted average of NLK and MSE. This is the loss used for training
               so it can be back-propogated
        predicted_llks - array, negative log-likelihood
        mse_loss - array, MSE of point predictions
        """
        bin_pred, gpd_pred, norm_pred = self.split_pred(pred)
        # I'm pretty sure this line was necessary to gpd_pred later on for debugging purposes. I don't think
        # its needed now but decided to keep it just in case.
        gpd_pred = utils.split_var(gpd_pred)
        predicted_llks = -1 * m.torch_nanmean(m.llk(y, gpd_pred, norm_pred, bin_pred[:, 0], bin_pred[:, 1], self.effective_thresh(threshes), moderate_func=self.moderate_func))
        mse_loss = self.compute_mse(y, pred, threshes)

        loss = predicted_llks + self.mean_multiplier * mse_loss + (0 if (self.dropout_multiplier == 0) else (self.dropout_multiplier * self.model.regularisation()) )
        return loss, m.to_np(predicted_llks), m.to_np(mse_loss)
    
    def compute_metrics(self, y, pred, threshes):
        """
        Computes a wide range of evaluation metrics
        Parameters:
        y - tensor, target variable
        pred - list of tensors, list of mixture model parameters
        threshes - tensor, thresholds
        
        Returns:
        loss - scalar, this is the weighted average of NLK and MSE.
        predicted_llks - scalar, negative log-likelihood
        mse_loss - scalar, MSE of point predictions
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
        predicted_llks = -1 * m.torch_nanmean(m.llk(y, gpd_pred, norm_pred, bin_pred[:, 0], bin_pred[:, 1], self.effective_thresh(threshes), moderate_func=self.moderate_func))
        mse_loss = self.compute_mse(y, pred, threshes)
        
        zero_brier, moderate_brier, excess_brier, acc, f1_micro, f1_macro, auc_macro_ovo, auc_macro_ovr = self.compute_class_metrics(y, pred, threshes)
        
        loss = predicted_llks + self.mean_multiplier * mse_loss + (0 if (self.dropout_multiplier == 0) else (self.dropout_multiplier * self.model.regularisation()) )
        return m.to_np(loss), m.to_np(predicted_llks), m.to_np(mse_loss), zero_brier, moderate_brier, excess_brier, acc, f1_micro, f1_macro, auc_macro_ovo, auc_macro_ovr

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
        tru_zero = m.to_np((y == 0) * 1.)
        tru_excess = m.to_np((y > threshes) * 1.)
        tru_moderate = m.to_np(1 - tru_zero - tru_excess)
        
        # Compute predicted probabilities
        pred_zero, pred_moderate, pred_excess = self.compute_all_probs(pred, threshes, aslist=True)
        
        zero_brier = utils.brier_score(tru_zero, pred_zero)
        moderate_brier = utils.brier_score(tru_moderate, pred_moderate)
        excess_brier = utils.brier_score(tru_excess, pred_excess)
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
        tru_labels = utils.compute_class_labels(y, threshes)

        acc = utils.accuracy(tru_labels, pred_labels)
        f1_micro, f1_macro = utils.f1(tru_labels, pred_labels)
        auc_macro_ovo, auc_macro_ovr = utils.auc(tru_labels, np.stack(pred_probs, axis=0))
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
        return m.to_np(m.to_np(bin_pred[:, 0]))
    
    def compute_nonzero_prob(self, pred):
        """
        Computes predicted probability of non-zero rainfall
        Parameters:
        pred - list of tensors, list of mixture model parameters
        
        Returns:
        bin_pred - array, probability of non-zero rainfall
        """
        return m.to_np(1 - self.compute_zero_prob(pred))
    
    def compute_moderate_prob(self, pred, threshes):
        """
        Computes predicted probability of non-zero non-excess rainfall
        Parameters:
        pred - list of tensors, list of mixture model parameters
        
        Returns:
        bin_pred - array, probability of non-zero non-excess rainfall
        """
        return m.to_np(1 - self.compute_zero_prob(pred) - self.compute_excess_prob(pred, threshes))
            
    def compute_excess_prob(self, pred, threshes):
        """
        Computes predicted probability of excess rainfall
        Parameters:
        pred - list of tensors, list of mixture model parameters
        
        Returns:
        bin_pred - array, probability of excess rainfall
        """
        bin_pred, gpd_pred, norm_pred = self.split_pred(pred)
        return m.to_np((1 - m.all_cdf(threshes, gpd_pred, norm_pred, bin_pred[:, 0], bin_pred[:, 1], self.effective_thresh(threshes), threshes, self.moderate_func)))

    def compute_llk(self, y, pred, threshes):
        """
        Computes log-likelihood
        Parameters:
        y - tensor, target variable
        pred - list of tensors, list of mixture model parameters
        threshes - tensor, thresholds
        """
        bin_pred, gpd_pred, norm_pred = self.split_pred(pred)
        return m.llk(y[:, 0], gpd_pred, norm_pred, bin_pred[:, 0], bin_pred[:, 1], self.effective_thresh(threshes), moderate_func=self.moderate_func)
    
    def compute_mse(self, y, pred, threshes):
        """
        Computes MSE of point predictions
        Parameters:
        y - tensor, target variable
        pred - list of tensors, list of mixture model parameters
        threshes - tensor, thresholds
        """
        bin_pred, gpd_pred, moderate_pred = self.split_pred(pred)
        point_pred = m.all_mean(gpd_pred, moderate_pred, bin_pred[:, 0], bin_pred[:, 1], self.effective_thresh(threshes), self.moderate_func)
        return m.torch_mse(y, point_pred)

    def _mc_forwards(self, x, threshes, n_forwards):
        """
        Compute n_forwards forward passes through network with dropout and returns stacked predictions.
        Used for Vandal et al
        """
        results = list()
        for _ in range(n_forwards):
            bin_pred, gpd_pred, lognorm_pred = self.pred_stats(x, threshes)
            results.append(torch.cat([bin_pred, lognorm_pred], axis=1))
        preds = torch.stack(results, axis=0)
        return preds

    def _first_moment(self, non_zeros, mus, sigs):
        """
        Computes first moment for mc dropout w/ zero inflated lognormal
        Used for Vandal et al
        """
        return torch.mean(non_zeros * torch.exp(mus + 0.5 * sigs**2), axis=0)

    def _second_moment(self, non_zeros, mus, sigs):
        """
        Computes second moment for mc dropout w/ zero inflated lognormal
        Used for Vandal et al
        """
        return torch.mean(non_zeros ** 2 * torch.exp(2 * mus + 2 * sigs**2), axis=0)

    def _get_sig(self, first_moment, second_moment):
        """
        Computes sigma of zero inflated lognormal from first two moments
        Used for Vandal et al
        """
        return torch.log(1 + 0.5 * (4 * second_moment / first_moment**2 + 1)**0.5)

    def _get_mu(self, first_moment, sig):
        """
        Computes mu of zero inflated lognormal from first two moments
        Used for Vandal et al
        """
        return first_moment - sig**2 / 2

    def compute_mc_stats(self, x, threshes, n_forwards):
        """
        Computes mus and sigmas of zero-inflated lognormal
        Used for Vandal et al
        """
        preds = self._mc_forwards(x, threshes, n_forwards)
        non_zeros = 1 - preds[:, :, 0]
        mus = preds[:, :, 2]
        sigs = preds[:, :, 3]**0.5
        first_m = self._first_moment(non_zeros, mus, sigs)
        second_m = self._second_moment(non_zeros, mus, sigs)
        final_sigs = self._get_sig(first_m, second_m)
        final_mus = self._get_mu(first_m, final_sigs)
        
        bin_pred_avgs = torch.mean(preds[:, :, 0:2], axis=0)
        lognormal_stats = torch.stack([final_mus, final_sigs**2], axis=1)
        return bin_pred_avgs, torch.zeros_like(lognormal_stats) + 0.1, lognormal_stats


def make_cnn(ndim, hdim, odim, ksize, padding, use_mc, variable_thresh, use_bnorm, nonlin=None):
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
    use_bnorm - boolean, if True use batch norm
    nonlin - string, represents activation function. Must be either relu or tanh
    """
    if use_mc:
        assert not use_bnorm
        return CNNConcreteDropout(ndim, hdim, odim, ksize, padding)
    else:
        model = CNN(ndim, hdim, odim, ksize, padding, variable_thresh, use_bnorm, nonlin)
    return model    


class CNN(nn.Module):

    def __init__(self, ndim, hdim, odim, ksize, padding, variable_thresh, use_bnorm, nonlin=None) -> None:
        """
        Makes pytorch CNN model.
        Parameters:
        ndim - int, number of input dimensions
        hdim - int, CNN number of hidden dimensions
        odim - int, number of outputs per location
        ksize - tuple of ints, CNN kernel size
        padding - tuple of ints, CNN padding
        variable_thresh - boolean, if True thresholds are randomized during training
        use_bnorm - boolean, if True use batch norm
        nonlin - string, represents activation function. Must be either relu or tanh
        """
        if nonlin is None or nonlin == 'relu':
            non_lin = torch.nn.ReLU
        elif nonlin == 'tanh':
            non_lin = torch.nn.Tanh
        else:
            raise ValueError('only relu and tanh supported')
        super().__init__()
        self.variable_thresh = variable_thresh
        self.cnns = torch.nn.Sequential(
            nn.BatchNorm3d(ndim) if use_bnorm else nn.Identity(),
            torch.nn.Conv3d(ndim, hdim, ksize, padding=padding),
            non_lin(),
            nn.BatchNorm3d(hdim) if use_bnorm else nn.Identity(),
            torch.nn.Conv3d(hdim, hdim, ksize, padding=padding),
            non_lin(),
            nn.BatchNorm3d(hdim) if use_bnorm else nn.Identity(),
            torch.nn.Conv3d(hdim, hdim, ksize, padding=padding),
            non_lin(),
            nn.BatchNorm3d(hdim) if use_bnorm else nn.Identity()
        )
        self.fcs = torch.nn.Sequential(
            torch.nn.Conv3d((hdim + 1) if variable_thresh else hdim, hdim, (1, 1, 1)),
            non_lin(),
            nn.BatchNorm3d(hdim) if use_bnorm else nn.Identity(),
            torch.nn.Conv3d(hdim, hdim, (1, 1, 1)),
            non_lin(),
            nn.BatchNorm3d(hdim) if use_bnorm else nn.Identity(),
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
            threshes = (x[:, -1:] - 2.79)/2.26
            x = x[:, :-1]
        else:
            threshes = None
        out = self.cnns(x)
        if not threshes is None:
            out = torch.cat([out, threshes[:, :, -1:]], axis=1)
        out = self.fcs(out)
        return out


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