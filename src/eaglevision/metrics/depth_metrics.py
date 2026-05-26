from __future__ import annotations

import math

import torch


def _as_bhw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor.squeeze(1)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    return tensor


def valid_depth_mask(
    prediction: torch.Tensor,
    target: torch.Tensor | None = None,
    min_depth: float = 1e-3,
    max_depth: float = 300.0,
) -> torch.Tensor:
    pred = _as_bhw(prediction)
    mask = torch.isfinite(pred) & (pred >= min_depth) & (pred <= max_depth)
    if target is not None:
        tgt = _as_bhw(target)
        mask = mask & torch.isfinite(tgt) & (tgt >= min_depth) & (tgt <= max_depth)
    return mask


def _valid_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float = 300.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = _as_bhw(prediction)
    tgt = _as_bhw(target)
    if pred.shape != tgt.shape:
        raise ValueError(f"Prediction and target shapes must match, got {tuple(pred.shape)} and {tuple(tgt.shape)}")
    mask = valid_depth_mask(pred, tgt, min_depth=min_depth, max_depth=max_depth)
    return pred[mask], tgt[mask]


def _nan() -> float:
    return float("nan")


def depth_l1(prediction: torch.Tensor, target: torch.Tensor, min_depth: float = 1e-3, max_depth: float = 300.0) -> float:
    pred, tgt = _valid_values(prediction, target, min_depth=min_depth, max_depth=max_depth)
    return float(torch.abs(pred - tgt).mean().item()) if pred.numel() else _nan()


def abs_rel(prediction: torch.Tensor, target: torch.Tensor, min_depth: float = 1e-3, max_depth: float = 300.0) -> float:
    pred, tgt = _valid_values(prediction, target, min_depth=min_depth, max_depth=max_depth)
    return float((torch.abs(pred - tgt) / tgt.clamp_min(1e-6)).mean().item()) if pred.numel() else _nan()


def sq_rel(prediction: torch.Tensor, target: torch.Tensor, min_depth: float = 1e-3, max_depth: float = 300.0) -> float:
    pred, tgt = _valid_values(prediction, target, min_depth=min_depth, max_depth=max_depth)
    return float((((pred - tgt) ** 2) / tgt.clamp_min(1e-6)).mean().item()) if pred.numel() else _nan()


def rmse(prediction: torch.Tensor, target: torch.Tensor, min_depth: float = 1e-3, max_depth: float = 300.0) -> float:
    pred, tgt = _valid_values(prediction, target, min_depth=min_depth, max_depth=max_depth)
    return float(torch.sqrt(((pred - tgt) ** 2).mean()).item()) if pred.numel() else _nan()


def rmse_log(prediction: torch.Tensor, target: torch.Tensor, min_depth: float = 1e-3, max_depth: float = 300.0) -> float:
    pred, tgt = _valid_values(prediction, target, min_depth=min_depth, max_depth=max_depth)
    if not pred.numel():
        return _nan()
    diff = torch.log(pred.clamp_min(1e-6)) - torch.log(tgt.clamp_min(1e-6))
    return float(torch.sqrt((diff**2).mean()).item())


def silog(prediction: torch.Tensor, target: torch.Tensor, min_depth: float = 1e-3, max_depth: float = 300.0) -> float:
    pred, tgt = _valid_values(prediction, target, min_depth=min_depth, max_depth=max_depth)
    if not pred.numel():
        return _nan()
    diff = torch.log(pred.clamp_min(1e-6)) - torch.log(tgt.clamp_min(1e-6))
    value = ((diff**2).mean() - diff.mean() ** 2).clamp_min(0.0)
    return float((torch.sqrt(value) * 100.0).item())


def delta_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    min_depth: float = 1e-3,
    max_depth: float = 300.0,
) -> float:
    pred, tgt = _valid_values(prediction, target, min_depth=min_depth, max_depth=max_depth)
    if not pred.numel():
        return _nan()
    ratio = torch.maximum(pred / tgt.clamp_min(1e-6), tgt / pred.clamp_min(1e-6))
    return float((ratio < threshold).float().mean().item())


def delta1(prediction: torch.Tensor, target: torch.Tensor, min_depth: float = 1e-3, max_depth: float = 300.0) -> float:
    return delta_accuracy(prediction, target, 1.25, min_depth=min_depth, max_depth=max_depth)


def delta2(prediction: torch.Tensor, target: torch.Tensor, min_depth: float = 1e-3, max_depth: float = 300.0) -> float:
    return delta_accuracy(prediction, target, 1.25**2, min_depth=min_depth, max_depth=max_depth)


def delta3(prediction: torch.Tensor, target: torch.Tensor, min_depth: float = 1e-3, max_depth: float = 300.0) -> float:
    return delta_accuracy(prediction, target, 1.25**3, min_depth=min_depth, max_depth=max_depth)


def compute_depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float = 300.0,
) -> dict[str, float]:
    metrics = {
        "absrel": abs_rel(prediction, target, min_depth=min_depth, max_depth=max_depth),
        "sqrel": sq_rel(prediction, target, min_depth=min_depth, max_depth=max_depth),
        "rmse": rmse(prediction, target, min_depth=min_depth, max_depth=max_depth),
        "rmse_log": rmse_log(prediction, target, min_depth=min_depth, max_depth=max_depth),
        "silog": silog(prediction, target, min_depth=min_depth, max_depth=max_depth),
        "delta1": delta1(prediction, target, min_depth=min_depth, max_depth=max_depth),
        "delta2": delta2(prediction, target, min_depth=min_depth, max_depth=max_depth),
        "delta3": delta3(prediction, target, min_depth=min_depth, max_depth=max_depth),
    }
    return {key: (value if math.isfinite(value) else float("nan")) for key, value in metrics.items()}
