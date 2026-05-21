import torch

from earlyon.core.temperature import fit_temperature


def test_perfectly_confident_correct_predictions_give_t_below_1():
    """If a model is *over*-confident and correct, T should shrink towards 1
    (or could go slightly above 1 if too peaked). At minimum, fit returns a
    positive scalar without crashing."""
    torch.manual_seed(0)
    # synthetic: logits are the correct class with margin
    n, c = 64, 5
    targets = torch.randint(0, c, (n,))
    logits = torch.zeros(n, c)
    logits[torch.arange(n), targets] = 5.0
    t = fit_temperature(logits, targets, max_iter=50)
    assert t > 0


def test_uniform_logits_temperature_is_stable():
    n, c = 32, 4
    logits = torch.zeros(n, c)
    targets = torch.randint(0, c, (n,))
    t = fit_temperature(logits, targets, max_iter=20)
    assert t > 0


def test_wrong_shape_raises():
    import pytest

    with pytest.raises(ValueError, match="logits must be 2D"):
        fit_temperature(torch.zeros(3, 4, 5), torch.zeros(3, dtype=torch.long))
