import torch
from torch import nn


class SigmoidScheduler(nn.Module):
    """Sigmoid transition class based on current epoch"""

    def __init__(
        self,
        starting_value: float,
        target_value: float,
        starting_epoch: int,
        ending_epoch: int,
    ) -> None:
        """Constructor

        Args:
            starting_value (float): Pre-transition value
            target_value (float): Post-transition value
            starting_epoch (int): Transition starting epoch
            ending_epoch (int): Transition end epoch
        """
        super().__init__()
        assert starting_epoch < ending_epoch

        self.starting_value = starting_value
        self.target_value = target_value
        self.starting_epoch = starting_epoch
        self.ending_epoch = ending_epoch

    def _scale(self, current_epoch):
        """Map to classic sigmoid "useful" range (-6,6)"""
        ratio = (current_epoch - self.starting_epoch) / (
            self.ending_epoch - self.starting_epoch
        )

        return torch.tensor(-6 + ratio * 12)

    def step(self, current_epoch: int) -> float:
        """Return value for current epoch

        Args:
            current_epoch (int): Current epoch

        Returns:
            float: Value for current epoch
        """

        if current_epoch < self.starting_epoch:
            return self.starting_value
        elif current_epoch > self.ending_epoch:
            return self.target_value
        else:
            return float(
                self.starting_value
                + (torch.sigmoid(self._scale(current_epoch)))
                * (self.target_value - self.starting_value)
            )
