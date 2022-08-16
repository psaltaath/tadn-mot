from typing import Callable

import torch
from torchvision.ops import generalized_box_iou


def pairwise_iou_metric(bbs1: torch.Tensor, bbs2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU metric

    Args:
        bbs1 (torch.Tensor): (num_boxes1, 4) Set #1 of bboxes in xyxy format
        bbs2 (torch.Tensor): (num_boxes2, 4) Set #2 of bboxes in xyxy format

    Returns:
        torch.Tensor: (num_boxes1, num_boxes2) Pairwise distance matrix
    """
    return generalized_box_iou(bbs1, bbs2)


def pairwise_ulbr1_metric(bbs1: torch.Tensor, bbs2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise ulbr1 metric

    Args:
        bbs1 (torch.Tensor): (num_boxes1, 4) Set #1 of bboxes in xyxy format
        bbs2 (torch.Tensor): (num_boxes2, 4) Set #2 of bboxes in xyxy format

    Returns:
        torch.Tensor: (num_boxes1, num_boxes2) Pairwise distance matrix
    """

    Nominator = torch.cdist(bbs1, bbs2, p=1)

    D1 = torch.norm(bbs1[..., :2] - bbs1[..., -2:], p=1, keepdim=True, dim=-1)  # type: ignore
    D2 = torch.norm(bbs2[..., :2] - bbs2[..., -2:], p=1, keepdim=True, dim=-1)  # type: ignore

    Denominator = torch.cdist(D1, -D2, p=1)

    return -(Nominator / Denominator)


# Global metric default preference
_PAIRWISE_METRIC: Callable[
    [torch.Tensor, torch.Tensor], torch.Tensor
] = pairwise_iou_metric


def set_metric(metric_name: str) -> None:
    """Change global metric preference

    Options:
        "iou"
        "ulbr1"
    """
    global _PAIRWISE_METRIC
    metric_name_dict = {
        "iou": pairwise_iou_metric,
        "ulbr1": pairwise_ulbr1_metric,
    }
    assert metric_name in metric_name_dict
    _PAIRWISE_METRIC = metric_name_dict[metric_name]


def pairwise(bbs1: torch.Tensor, bbs2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distances using the global metric

    Args:
        bbs1 (torch.Tensor): (num_boxes1, 4) Set #1 of bboxes in xyxy format
        bbs2 (torch.Tensor): (num_boxes2, 4) Set #2 of bboxes in xyxy format

    Returns:
        torch.Tensor: (num_boxes1, num_boxes2) Pairwise distance matrix
    """
    return _PAIRWISE_METRIC(bbs1, bbs2)
