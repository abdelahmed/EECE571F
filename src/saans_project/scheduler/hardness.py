from __future__ import annotations


def coord_only_hardness(coord_loss: float) -> float:
    return float(coord_loss)


def combined_hardness(coord_loss: float, feat_loss: float, lambda_coord: float = 1.0, lambda_feat: float = 1.0) -> float:
    return float(lambda_coord * coord_loss + lambda_feat * feat_loss)
