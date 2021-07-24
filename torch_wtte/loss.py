"""loss.py

Weibull Time-To-Event loss functions for PyTorch.
"""

import torch
from torch import nn

EPS = torch.finfo(torch.float32).eps


def log_likelihood_discrete(tte, uncensored, alpha, beta, epsilon=EPS):
    """Discrete version of log likelihood for the Weibull TTE loss function.

    Parameters
    ----------
    tte : torch.tensor
        tensor of each subject's time to event at each time step.
    uncensored : torch.tensor
        tensor indicating whether each data point is censored (0) or
        not (1)
    alpha : torch.tensor
        Estimated Weibull distribution scale parameter per subject,
        per time step.
    beta : torch.tensor
        Estimated Weibull distribution shape parameter per subject,
        per time step.

    Examples
    --------
    FIXME: Add docs.

    """

    hazard_0 = torch.pow((tte + epsilon) / alpha, beta)
    hazard_1 = torch.pow((tte + 1.0) / alpha, beta)
    return uncensored * torch.log(torch.exp(hazard_1 - hazard_0) - (1.0 - epsilon)) - hazard_1


def log_likelihood_continuous(tte, uncensored, alpha, beta, epsilon=EPS):
    """Continuous version of log likelihood for the Weibull TTE loss function.

    Parameters
    ----------
    tte : torch.tensor
        tensor of each subject's time to event at each time step.
    uncensored : torch.tensor
        tensor indicating whether each data point is censored (0) or
        not (1)
    alpha : torch.tensor
        Estimated Weibull distribution scale parameter per subject,
        per time step.
    beta : torch.tensor
        Estimated Weibull distribution shape parameter per subject,
        per time step.

    Examples
    --------
    FIXME: Add docs.

    """
    y_a = (tte + epsilon) / alpha
    return uncensored * (torch.log(beta) + beta * torch.log(y_a)) - torch.pow(y_a, beta)


def weibull_censored_nll_loss(
    tte: torch.tensor,
    uncensored: torch.tensor,
    alpha: torch.tensor,
    beta: torch.tensor,
    discrete: bool = False,
    reduction: str = "mean",
):
    """Compute the loss.

    Compute the Weibull censored negative log-likelihood loss for
    forward propagation.

    Parameters
    ----------
    tte : torch.tensor
        tensor of each subject's time to event at each time step.
    uncensored : torch.tensor
        tensor indicating whether each data point is censored (0) or
        not (1)
    alpha : torch.tensor
        Estimated Weibull distribution scale parameter per subject,
        per time step.
    beta : torch.tensor
        Estimated Weibull distribution shape parameter per subject,
        per time step.
    """
    reducer = {"mean": torch.mean, "sum": torch.sum}.get(reduction)
    likelihood = log_likelihood_discrete if discrete else log_likelihood_continuous
    log_likelihoods = likelihood(tte, uncensored, alpha, beta)
    if reducer:
        log_likelihoods = reducer(log_likelihoods, dim=-1)
    return -1.0 * log_likelihoods


class WeibullCensoredNLLLoss(nn.Module):
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
        super().__init__()
        self.discrete = discrete
        self.reduction = reduction

    def forward(
        self,
        tte: torch.tensor,
        uncensored: torch.tensor,
        alpha: torch.tensor,
        beta: torch.tensor,
    ):
        """Compute the loss.

        Compute the Weibull censored negative log-likelihood loss for
        forward propagation.

        Parameters
        ----------
        tte : torch.tensor
            tensor of each subject's time to event at each time step.
        uncensored : torch.tensor
            tensor indicating whether each data point is censored (0) or
            not (1)
        alpha : torch.tensor
            Estimated Weibull distribution scale parameter per subject,
            per time step.
        beta : torch.tensor
            Estimated Weibull distribution shape parameter per subject,
            per time step.
        """
        return weibull_censored_nll_loss(
            tte, uncensored, alpha, beta, self.discrete, self.reduction
        )


class WeibullActivation(nn.Module):
    """Layer that initializes, activates and regularizes alpha and beta parameters of
    a Weibull distribution."""

    def __init__(self, init_alpha: float = 1.0, max_beta: float = 5.0, epsilon: float = EPS):
        super().__init__()
        self.init_alpha = init_alpha
        self.max_beta = max_beta
        self.epsilon = epsilon

    def forward(self, x: torch.tensor):
        """Compute the activation function.

        Parameters
        ----------
        x : torch.tensor
            An input tensor with innermost dimension = 2 ([alpha,
            beta])
        """

        alpha = x[..., 0]
        beta = x[..., 1]

        alpha = self.init_alpha * torch.exp(alpha)

        if self.max_beta > 1.05:
            shift = torch.log(self.max_beta - 1.0)
            beta = beta - shift

        beta = self.max_beta * torch.clamp(
            torch.sigmoid(beta), min=self.epsilon, max=1.0 - self.epsilon
        )

        return torch.stack([alpha, beta], axis=-1)
