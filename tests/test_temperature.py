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


def test_overconfident_model_recovers_temperature_near_nll_optimum():
    """Regression for LBFGS underfitting. Build a genuinely miscalibrated model:
    labels are sampled from softmax(z) (the true distribution) while the observed
    logits are ``z * 3`` (three times too peaked). The NLL-optimal temperature is
    then ~3.0, and the fit must recover it. The old single-step lr=0.01 fit
    returned ~1.35 here and would fail the tolerance check."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    n, c = 2000, 10
    z = torch.randn(n, c)
    targets = torch.multinomial(torch.softmax(z, dim=-1), 1).squeeze(-1)
    overconfident = z * 3.0

    grid = torch.linspace(0.5, 6.0, 56)
    nlls = [F.cross_entropy(overconfident / t, targets).item() for t in grid]
    t_opt = grid[int(torch.tensor(nlls).argmin())].item()

    t_fit = fit_temperature(overconfident, targets)
    assert abs(t_fit - t_opt) < 0.5, f"fit {t_fit:.3f} far from optimum {t_opt:.3f}"
    assert t_fit > 2.0, "over-confident model must yield T well above 1"


def test_nonfinite_logits_raise_instead_of_poisoning_temperature():
    """A single NaN/Inf logit must raise, not silently return NaN that would
    poison every downstream softmax via model.config.temperature."""
    import pytest

    targets = torch.zeros(8, dtype=torch.long)
    nan_logits = torch.full((8, 4), float("nan"))
    with pytest.raises(ValueError, match="NaN or Inf"):
        fit_temperature(nan_logits, targets)

    inf_logits = torch.zeros(8, 4)
    inf_logits[0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        fit_temperature(inf_logits, targets)


def test_fit_temperature_full_reports_status():
    """The status-carrying fit must return a finite positive temperature and a
    coherent (converged, fallback) pair on well-behaved logits."""
    import torch

    from earlyon.core.temperature import fit_temperature_full

    torch.manual_seed(0)
    logits = torch.randn(200, 10) * 3.0
    targets = torch.randint(0, 10, (200,))
    result = fit_temperature_full(logits, targets)
    assert result.temperature > 0.0
    import math

    assert math.isfinite(result.temperature)
    assert isinstance(result.converged, bool)
    assert isinstance(result.fallback, bool)
    assert not result.fallback  # nothing degenerate here


def test_fit_temperature_full_never_returns_invalid_temperature():
    """Even on tiny/degenerate batches the fit must return a value that
    EarlyExitConfig.validate() accepts (finite, > 0) — falling back to 1.0
    with fallback=True rather than emitting a poison value."""
    import math

    import torch

    from earlyon.core.temperature import fit_temperature_full

    torch.manual_seed(1)
    # single-sample, extreme logits: classic LBFGS divergence bait
    logits = torch.tensor([[1e4, -1e4, 0.0]])
    targets = torch.tensor([1])
    result = fit_temperature_full(logits, targets, max_outer_steps=3)
    assert math.isfinite(result.temperature)
    assert result.temperature > 0.0
