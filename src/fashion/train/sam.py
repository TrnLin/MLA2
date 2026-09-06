"""First-order L2 SAM over AdamW, with one stochastic batch and one update."""

import math

import torch

POLICY = {
    "version": "gender_sam005_adamw_v1",
    "rho": 0.05,
    "adaptive": False,
    "norm": "global L2 over all trainable parameters, including BatchNorm affine",
    "epsilon": 1e-12,
    "base_optimizer": "AdamW",
    "gradient": "mixed cross-entropy before weight decay",
    "batch": "same augmented mixed inputs and labels for both passes",
    "dropout": "same mask on both passes; advance torch RNG once",
    "batchnorm": "training statistics twice; retain first-pass running buffers only",
    "update": "restore exact original weights; apply second gradient and AdamW decay once",
    "diagnostic_epochs": [10, 15, 20, 25, 30],
    "checkpoint": "final_epoch",
}


class SAMStep:
    """Own the two backward passes; leave scheduling on the original AdamW."""

    def __init__(self, model, optimizer):
        if type(optimizer) is not torch.optim.AdamW:
            raise ValueError("The frozen SAM trial requires AdamW")
        self.model, self.optimizer = model, optimizer
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        supplied = [p for group in optimizer.param_groups for p in group["params"]]
        if len(supplied) != len(self.parameters) or {id(p) for p in supplied} != {
            id(p) for p in self.parameters
        }:
            raise ValueError("SAM must update all trainable parameters exactly once")
        if not self.parameters or len({p.device for p in self.parameters}) != 1:
            raise ValueError("SAM requires one device and at least one trainable parameter")
        device = self.parameters[0].device
        self.cuda_devices = [device.index] if device.type == "cuda" else []
        self.buffers = [
            b
            for module in model.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for b in (module.running_mean, module.running_var, module.num_batches_tracked)
            if b is not None
        ]
        self.epochs = []
        self.current = None

    def begin_epoch(self, epoch):
        if self.current is not None or epoch != len(self.epochs) + 1:
            raise ValueError("SAM epochs must be consecutive and complete")
        self.current = dict(
            epoch=epoch,
            rows=0,
            batches=0,
            forward_backward_passes=0,
            optimizer_steps=0,
            first_loss_sum=0.0,
            second_loss_sum=0.0,
            gradient_norm_min=math.inf,
            gradient_norm_max=0.0,
        )

    @staticmethod
    def _restore(values, originals):
        with torch.no_grad():
            for value, original in zip(values, originals, strict=True):
                value.copy_(original)

    def _backward(self, closure):
        self.optimizer.zero_grad(set_to_none=True)
        logits, loss = closure()
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("SAM encountered a non-finite or non-scalar loss")
        loss.backward()
        gradients = [p.grad for p in self.parameters if p.grad is not None]
        if not gradients or any(g.is_sparse for g in gradients):
            raise ValueError("SAM requires dense gradients")
        norm = torch.linalg.vector_norm(torch.stack([g.detach().norm(2) for g in gradients]))
        if not torch.isfinite(norm):
            raise FloatingPointError("SAM encountered a non-finite gradient norm")
        return logits.detach(), loss.detach(), norm

    def step(self, closure, *, rows):
        if self.current is None or not self.model.training or rows <= 0:
            raise ValueError("SAM requires an active training epoch and a nonempty batch")
        originals = [p.detach().clone() for p in self.parameters]
        initial_buffers = [b.clone() for b in self.buffers]
        cpu_rng = torch.get_rng_state()
        cuda_rng = [torch.cuda.get_rng_state(d) for d in self.cuda_devices]
        try:
            logits, first_loss, norm = self._backward(closure)
            first_buffers = [b.clone() for b in self.buffers]
            with torch.no_grad():
                scale = POLICY["rho"] / (norm + POLICY["epsilon"])
                for p in self.parameters:
                    if p.grad is not None:
                        p.add_(p.grad * scale)
            # fork_rng restores the post-first-pass RNG even if the second pass fails.
            with torch.random.fork_rng(devices=self.cuda_devices):
                torch.set_rng_state(cpu_rng)
                for device, state in zip(self.cuda_devices, cuda_rng, strict=True):
                    torch.cuda.set_rng_state(state, device)
                try:
                    _, second_loss, _ = self._backward(closure)
                finally:
                    self._restore(self.buffers, first_buffers)
            self._restore(self.parameters, originals)
            self.optimizer.step()
        except BaseException:
            self._restore(self.parameters, originals)
            self._restore(self.buffers, initial_buffers)
            self.optimizer.zero_grad(set_to_none=True)
            raise
        stats = self.current
        stats["rows"] += rows
        stats["batches"] += 1
        stats["forward_backward_passes"] += 2
        stats["optimizer_steps"] += 1
        stats["first_loss_sum"] += float(first_loss) * rows
        stats["second_loss_sum"] += float(second_loss) * rows
        stats["gradient_norm_min"] = min(stats["gradient_norm_min"], float(norm))
        stats["gradient_norm_max"] = max(stats["gradient_norm_max"], float(norm))
        return logits, first_loss

    def end_epoch(self, mixup_epoch):
        if self.current is None or any(
            self.current[key] != mixup_epoch[key] for key in ("epoch", "rows", "batches")
        ):
            raise ValueError("SAM and MixUp batch coverage disagree")
        stats = dict(self.current)
        for name in ("first", "second"):
            stats[name + "_loss"] = stats.pop(name + "_loss_sum") / stats["rows"]
        self.epochs.append(stats)
        self.current = None
        return stats

    def receipt(self):
        if self.current is not None:
            raise ValueError("Cannot save an unfinished SAM epoch")
        return {"policy": dict(POLICY), "epochs": list(self.epochs)}
