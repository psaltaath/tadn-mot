from typing import List
import torch


def bbox_xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from xywh to xyxy format

    Args:
        boxes (torch.Tensor): (N, 4) Boxes in xywh format

    Returns:
        torch.Tensor: (N, 4) Boxes in xyxy format
    """
    new_boxes = boxes.clone()
    new_boxes[..., 2:] += new_boxes[..., :2]
    return new_boxes


def convert_MOTC_format(frame: int, tgt_state: dict) -> List[str]:
    """Convert target state into MOTChallenge format for outputing results

    MOTChallenge format spec:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>, <is_hit>

    Args:
        frame (int): Frame-id (zero-indexed)
        tgt_state (dict): Targets state for specified frame from MOTManager instance

    Returns:
        list: List of str MOTChallenge formatted lines.
    """

    out_str = "{0}, {1}, {2}, {3}, {4}, {5}, 1, -1, -1, -1, {6}\n"

    lines = []
    for id, state_dict in tgt_state.items():
        lines.append(
            out_str.format(
                frame + 1, id, *state_dict["bbox"].tolist(), int(state_dict["is_hit"])
            )
        )

    return lines
