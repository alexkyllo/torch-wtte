"""loss.py

Weibull Time-To-Event loss functions for PyTorch.
"""

import torch
from torch import nn

EPS = torch.finfo(torch.float32).eps


def log_likelihood_discrete(y, u, a, b, epsilon=EPS):
    """Discrete version of log likelihood for the Weibull TTE loss function.

    Parameters
    ----------
    y : torch.tensor
        tensor of each subject's time to event at each time step.
    u : torch.tensor
        tensor indicating whether each data point is censored (0) or
        not (1)
    a : torch.tensor
        Estimated Weibull distribution scale parameter per subject,
        per time step.
    b : torch.tensor
        Estimated Weibull distribution shape parameter per subject,
        per time step.

    Examples
    --------
    FIXME: Add docs.

    """

    hazard_0 = torch.pow((y + epsilon) / a, b)
    hazard_1 = torch.pow((y + 1.0) / a, b)
    return u * torch.log(torch.exp(hazard_1 - hazard_0) - (1.0 - epsilon)) - hazard_1


def log_likelihood_continuous(y, u, a, b, epsilon=EPS):
    """Continuous version of log likelihood for the Weibull TTE loss function.

    Parameters
    ----------
    y : torch.tensor
        tensor of each subject's time to event at each time step.
    u : torch.tensor
        tensor indicating whether each data point is censored (0) or
        not (1)
    a : torch.tensor
        Estimated Weibull distribution scale parameter per subject,
        per time step.
    b : torch.tensor
        Estimated Weibull distribution shape parameter per subject,
        per time step.

    Examples
    --------
    FIXME: Add docs.

    """
    y_a = (y + epsilon) / a
    return u * (torch.log(b) + b * torch.log(y_a)) - torch.pow(y_a, b)


def weibull_censored_nll_loss(
    y: torch.tensor,
    u: torch.tensor,
    a: torch.tensor,
    b: torch.tensor,
    discrete: bool = False,
    reduction: str = "mean",
):
    """Compute the loss.

    Compute the Weibull censored negative log-likelihood loss for
    forward propagation.

    Parameters
    ----------
    y : torch.tensor
        tensor of each subject's time to event at each time step.
    u : torch.tensor
        tensor indicating whether each data point is censored (0) or
        not (1)
    a : torch.tensor
        Estimated Weibull distribution scale parameter per subject,
        per time step.
    b : torch.tensor
        Estimated Weibull distribution shape parameter per subject,
        per time step.
    """
    reducer = {"mean": torch.mean, "sum": torch.sum}.get(reduction)
    likelihood = log_likelihood_discrete if discrete else log_likelihood_continuous
    log_likelihoods = likelihood(y, u, a, b)
    if reducer:
        log_likelihoods = reducer(log_likelihoods, dim=-1)
    return -1.0 * log_likelihoods


class WeibullCensoredNLLLoss(nn.NLLLoss):
    """A negative log-likelihood loss function for Weibull distribution
    parameter estimation with right censoring.
    """

    def __init__(self, discrete: bool = False, reduction: str = "mean"):
        """Constructor.

        Construct the Weibull censored negative log-likelihood loss object.

        Parameters
        ----------
        discrete : bool
             Specifies whether to use the discrete (True) or continuous (False)
             variant of the Weibull distribution for parameter estimation.
        reduction : str
             Specifies the reduction to apply to the output: 'none' |
             'mean' | 'sum'. 'none': no reduction will be applied,
             'mean': the weighted mean of the output is taken, 'sum':
             the output will be summed. Note: size_average and reduce
             are in the process of being deprecated, and in the
             meantime, specifying either of those two args will
             override reduction. Default: 'mean'

        """
        super().__init__(reduction=reduction)
        self.discrete = discrete
        self.reduction = reduction

    def forward(
        self,
        y: torch.tensor,
        u: torch.tensor,
        a: torch.tensor,
        b: torch.tensor,
    ):
        """Compute the loss.

        Compute the Weibull censored negative log-likelihood loss for
        forward propagation.

        Parameters
        ----------
        y : torch.tensor
            tensor of each subject's time to event at each time step.
        u : torch.tensor
            tensor indicating whether each data point is censored (0) or
            not (1)
        a : torch.tensor
            Estimated Weibull distribution scale parameter per subject,
            per time step.
        b : torch.tensor
            Estimated Weibull distribution shape parameter per subject,
            per time step.
        """
        return weibull_censored_nll_loss(y, u, a, b, self.discrete, self.reduction)
