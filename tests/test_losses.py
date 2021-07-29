import torch
from torch_wtte import losses


def test_log_likelihood_discrete():
    """Test that the discrete version of the log-likelihood function
    returns the expected result.
    """
    tte = torch.tensor([[6, 5, 4, 3, 2], [5, 4, 3, 2, 1]])
    uncensored = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    alpha = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9], [0.99, 0.99, 0.99, 0.99, 0.99]])
    beta = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9], [1.1, 1.1, 1.1, 1.1, 1.1]])

    loss_values = losses.log_likelihood_discrete(tte, uncensored, alpha, beta)
    # results from wtte-rnn package
    expected = torch.tensor(
        [
            [-6.09469436, -5.24937802, -4.38501338, -3.49575045, -2.95522717],
            [-6.24961923, -4.96687732, -3.71907366, -2.5180084, -1.3889585],
        ]
    )
    eq_t = torch.isclose(
        loss_values,
        expected,
    )
    assert eq_t.all()


def test_log_likelihood_continuous():
    """Test that the discrete version of the log-likelihood function
    returns the expected result.
    """
    tte = torch.tensor([[6, 5, 4, 3, 2], [5, 4, 3, 2, 1]])
    uncensored = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    alpha = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9], [0.99, 0.99, 0.99, 0.99, 0.99]])
    beta = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9], [1.1, 1.1, 1.1, 1.1, 1.1]])

    loss_values = losses.log_likelihood_continuous(tte, uncensored, alpha, beta)
    # results from wtte-rnn package
    expected = torch.tensor(
        [
            [-3.91260149, -3.24213782, -2.59143362, -1.97701222, -2.0516759],
            [-4.06163704, -3.01458314, -2.07075338, -1.29854872, -0.90475116],
        ]
    )
    eq_t = torch.isclose(
        loss_values,
        expected,
    )
    assert eq_t.all()


def test_loss_fn():
    """Test that the discrete version of the loss function
    returns the expected result.
    """
    tte = torch.tensor([[6, 5, 4, 3, 2], [5, 4, 3, 2, 1]])
    uncensored = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    alpha = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9], [0.99, 0.99, 0.99, 0.99, 0.99]])
    beta = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9], [1.1, 1.1, 1.1, 1.1, 1.1]])
    inputs = torch.stack([alpha, beta], axis=-1)
    target = torch.stack([tte, uncensored], axis=-1)
    loss_values = losses.weibull_censored_nll_loss(inputs, target, discrete=True, reduction=None)
    # results from wtte-rnn package
    expected = torch.tensor(
        [
            [6.09469436, 5.24937802, 4.38501338, 3.49575045, 2.95522717],
            [6.24961923, 4.96687732, 3.71907366, 2.5180084, 1.3889585],
        ]
    )
    eq_t = torch.isclose(
        loss_values,
        expected,
    )
    assert eq_t.all()
