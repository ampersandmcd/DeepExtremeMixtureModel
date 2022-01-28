import math

import numpy as np
import torch
from scipy.stats import genpareto
from torch import nn


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


def torch_mse(w_nans, nonans):
    """
    Computes MSE between two tensors while ignoring nans while preserving gradients
    """
    denan = torch.zeros_like(w_nans, device=get_device())
    nonan_mask = ~torch.isnan(w_nans)
    denan[nonan_mask] += w_nans[nonan_mask]
    return torch.mean(((denan - nonans) ** 2)[nonan_mask])


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
    # Compute weighted mean of gpd component
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


def totensor(a):
    """
    Converts a numpy array or scalar to a tensor
    """
    device = get_device()
    if 'int' in str(type(a)) or 'float' in str(type(a)):
        return torch.tensor(np.array([a]), device=device, dtype=torch.float64)
    else:
        return torch.tensor(a, device=device, dtype=torch.float64)


def tonp(a):
    """
    Converts a tensor or list of tensor to a numpy array or list of numpy arrays respectively 
    """
    if 'torch' in str(type(a)):
        return a.cpu().detach().numpy()
    if ('list' in str(type(a)) or 'tuple' in str(type(a))) and 'torch' in str(type(a[0])):
        return [item.cpu().detach().numpy() for item in a]
    else:
        return a


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


def tru_gpd(samps, xis, sigmas):
    """
    Computes gpd log likelihood w/ scipy. Strictly for debugging
    """
    llks = list()
    for i in range(samps.shape[0]):
        llks.append(genpareto.logpdf(samps[i, :], xis[i], scale=sigmas[i]))
    return np.concatenate(llks)


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
    first_term = -torch.log(samples) - 0.5 * torch.log(var) - 0.5 * torch.log(totensor(2 * 3.14159274))
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


def llk_zero(samples, zero_probs):
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


def llk_above_thresh(samples, threshes, above_thresh_probs):
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


def llk(samples, gpd_stats, moderate_stats, zero_probs, excess_probs, threshes, moderate_func):
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
    zero_llk, nz_llk = llk_zero(samples, zero_probs)
    thresh_llk, nthresh_llk = llk_above_thresh(samples, threshes, excess_probs)

    # Create tensor w/ just the excesses and all other entries nan
    excesses = torch.zeros_like(samples)
    excess_inds = samples > threshes
    excesses[excess_inds] += samples[excess_inds] - threshes[excess_inds]
    excesses[~excess_inds] /= 0
    # Compute log-likelihood contributed by excess values
    excess_llk = gpd(excesses, gpd_stats[:, 0], gpd_stats[:, 1])

    # Create tensor w/ just the non-zero non-excess values
    nonzeros = torch.zeros_like(samples)
    nz_inds = (samples > 0) & ~(excess_inds)
    nonzeros[nz_inds] += samples[nz_inds]
    if moderate_func == 'lognormal':
        # Compute log-likelihood contributed by non-zero non-excess values
        main_llk = threshed_lognorm(nonzeros, threshes,
                                    moderate_stats[:, 0],
                                    moderate_stats[:, 1])
    else:
        raise ValueError('only lognormal is supported for non-excess values')
    main_llk[~nz_inds] = torch.zeros_like(main_llk[~nz_inds]) / 0.  # Set all 0 values and excess values to nan

    zero_llk, nz_llk, nthresh_llk, main_llk, thresh_llk, excess_llk = nan_transfer(samples, zero_llk), nan_transfer(
        samples, nz_llk), nan_transfer(samples, nthresh_llk), nan_transfer(samples, main_llk), nan_transfer(samples,
                                                                                                            thresh_llk), nan_transfer(
        samples, excess_llk)  # Make sure nans are propogating correctly
    total_llk = nan_to_num(zero_llk) + nan_to_num(nz_llk) + nan_to_num(nthresh_llk) + nan_to_num(main_llk) + nan_to_num(
        thresh_llk) + nan_to_num(excess_llk)  # Add up all log-likelihoods
    total_llk[nan_inds] += np.nan  # Make sure nans are propogating correctly
    return total_llk
