from abc import ABC, abstractmethod

import torch


class AbstractAppearanceModel(ABC):
    """Abstract class for Appearance model"""

    @abstractmethod
    def __init__(self, app_vector: torch.Tensor) -> None:
        """Base constructor

        Args:
            app_vector (torch.Tensor): (app_dim, ) Initial appearance features vector.
        """
        super().__init__()

    @abstractmethod
    def update(self, app_vector: torch.Tensor) -> None:
        """Update method.
        Updates state using new observation (app_vector).

        Args:
            app_vector (torch.Tensor): Updated appearance features vector.
        """
        pass

    @property
    @abstractmethod
    def current_state(self):
        """Retrieve current state

        Returns:
            (torch.Tensor): (app_dim, ) Current appearance features vector.
        """
        pass


class LastAppearanceVector(AbstractAppearanceModel):
    """<<Last Appearance>> Appearance model class"""

    def __init__(self, app_vector: torch.Tensor) -> None:
        """Constructor

        Args:
            app_vector (torch.Tensor): (app_dim, ) Initial appearance features vector.
        """
        super().__init__(app_vector)
        self.app_vector = app_vector

    def update(self, app_vector: torch.Tensor) -> None:
        """Update method.
        Updates state using new observation (app_vector).

        Args:
            app_vector (torch.Tensor): Updated appearance features vector.
        """
        self.app_vector = app_vector

    @property
    def current_state(self):
        """Retrieve current state

        Returns:
            (torch.Tensor): (app_dim, ) Current appearance features vector.
        """
        return self.app_vector
