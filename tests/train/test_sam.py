"""Numerical SAM checks, including stochastic state and error recovery."""

import copy

import pytest

torch = pytest.importorskip("torch")

from fashion.train.sam import POLICY, SAMStep  # noqa: E402


def test_sam_matches_two_gradient_reference_and_one_adamw_update():
    torch.manual_seed(2753)
    model = torch.nn.Linear(3, 2).double()
    reference = copy.deepcopy(model)
    x, y = torch.randn(7, 3).double(), torch.randn(7, 2).double()
    originals = [p.detach().clone() for p in reference.parameters()]
    criterion = torch.nn.MSELoss()
    criterion(reference(x), y).backward()
    gradient = torch.cat([p.grad.flatten() for p in reference.parameters()])
    expected_norm = gradient.norm()
    with torch.no_grad():
        for p in reference.parameters():
            p.add_(0.05 * p.grad / (expected_norm + 1e-12))
    perturbed = [p.detach().clone() for p in reference.parameters()]
    reference.zero_grad(set_to_none=True)
    criterion(reference(x), y).backward()
    with torch.no_grad():
        for p, original in zip(reference.parameters(), originals, strict=True):
            p.copy_(original)
    expected_opt = torch.optim.AdamW(reference.parameters(), lr=0.003, weight_decay=0.02)
    expected_opt.step()
    actual_opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.02)
    sam = SAMStep(model, actual_opt)
    sam.begin_epoch(1)
    visited = []

    def closure():
        visited.append([p.detach().clone() for p in model.parameters()])
        output = model(x)
        return output, criterion(output, y)

    sam.step(closure, rows=7)
    for actual, expected in zip(visited[1], perturbed, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    displacement = torch.cat(
        [(p - q).flatten() for p, q in zip(visited[1], originals, strict=True)]
    )
    assert displacement.norm().item() == pytest.approx(0.05, abs=1e-12)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        assert actual_opt.state[actual]["step"].item() == 1
    stats = sam.end_epoch({"epoch": 1, "rows": 7, "batches": 1})
    assert stats["optimizer_steps"] == 1 and stats["forward_backward_passes"] == 2
    assert stats["gradient_norm_min"] == pytest.approx(expected_norm.item())


def test_dropout_replays_mask_bn_updates_once_and_rng_advances_once():
    torch.manual_seed(12)
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 2)
    )
    reference = copy.deepcopy(model)
    x, y = torch.randn(24, 4), torch.arange(24) % 2
    before = torch.get_rng_state()
    reference(x)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(before)
    masks = []
    hook = model[1].register_forward_hook(lambda m, a, output: masks.append(output.eq(0)))
    opt = torch.optim.AdamW(model.parameters())
    sam = SAMStep(model, opt)
    sam.begin_epoch(1)

    def closure():
        output = model(x)
        return output, torch.nn.functional.cross_entropy(output, y)

    sam.step(closure, rows=len(y))
    hook.remove()
    assert len(masks) == 2 and torch.equal(masks[0], masks[1])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for (name, actual), (_, expected) in zip(
        model.named_buffers(), reference.named_buffers(), strict=True
    ):
        assert torch.equal(actual, expected), name
    assert model[0].num_batches_tracked.item() == 1


@pytest.mark.parametrize(
    "fault", ["first_loss", "second_loss", "second_exception", "gradient", "optimizer"]
)
def test_failures_restore_weights_and_buffers_and_do_not_count_step(fault):
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 2))
    opt = torch.optim.AdamW(model.parameters())
    sam = SAMStep(model, opt)
    sam.begin_epoch(1)
    original = copy.deepcopy(model.state_dict())
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        output = model(torch.ones(5, 4))
        loss = output.square().mean()
        if (fault == "first_loss" and calls == 1) or (fault == "second_loss" and calls == 2):
            loss = loss * float("nan")
        if fault == "second_exception" and calls == 2:
            raise RuntimeError("broken second pass")
        return output, loss

    if fault == "gradient":
        next(model.parameters()).register_hook(lambda grad: grad * float("inf"))
    if fault == "optimizer":

        def broken_step():
            with torch.no_grad():
                next(model.parameters()).add_(1)
            raise RuntimeError("broken optimizer")

        opt.step = broken_step
    with pytest.raises((FloatingPointError, RuntimeError)):
        sam.step(closure, rows=5)
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name]), name
    assert sam.current["optimizer_steps"] == 0
    assert all(p.grad is None for p in model.parameters())


def test_pass_mixes_rows_once_and_eval_does_not_perturb():
    from test_task3_gender_mixup import MAPPING, _training

    from fashion.train.mixup import TrainingMixUp
    from fashion.train.task3_baseline import _pass

    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 5)
    )
    opt = torch.optim.AdamW(model.parameters())
    sam = SAMStep(model, opt)
    mix = TrainingMixUp(_training(), validation_fold=0, label_to_index=MAPPING)
    batch = dict(
        image=torch.randn(10, 4),
        label=torch.arange(10) % 5,
        id=torch.arange(10),
        cv_fold=torch.ones(10),
        product_family_group=[f"f{i}" for i in range(10)],
        path=[str(i) for i in range(10)],
    )
    mix.begin_epoch(1)
    sam.begin_epoch(1)
    _, labels, _, _ = _pass(
        model,
        [batch],
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        optimizer=opt,
        mixup=mix,
        sam=sam,
    )
    mix.end_epoch()
    sam.end_epoch(mix.epochs[-1])
    assert labels.size == 0
    assert mix.epochs[0]["rows"] == 10 and mix.epochs[0]["batches"] == 1
    assert sam.receipt()["policy"] == POLICY
    before = copy.deepcopy(model.state_dict())
    _pass(model, [batch], torch.nn.CrossEntropyLoss(), torch.device("cpu"))
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert len(sam.epochs) == len(mix.epochs) == 1
    with pytest.raises(ValueError, match="training-only"):
        _pass(model, [batch], torch.nn.CrossEntropyLoss(), torch.device("cpu"), sam=sam)
