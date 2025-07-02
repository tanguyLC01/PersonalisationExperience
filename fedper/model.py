import torch
import torch.nn as nn
from typing import Tuple
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Type, Union
import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Type, Union

from omegaconf import DictConfig
from torch import Tensor
from torch import nn as nn

import logging
log = logging.getLogger(__name__)


class ModelSplit(ABC, nn.Module):
    """Abstract class for splitting a model into global_net and local_net."""

    def __init__(
        self,
        model: nn.Module,
    ):
        """Initialize the attributes of the model split.

        Args:
            model: dict containing the vocab sizes of the input attributes.
        """
        super().__init__()

        self._global_net, self._local_net = self._get_model_parts(model)

    @abstractmethod
    def _get_model_parts(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """Return the global_net and local_net of the model.

        Args:
            model: model to be split into local_net and global_net

        Returns
        -------
            Tuple where the first element is the global_net of the model
            and the second is the local_net.
        """

    @property
    def global_net(self) -> nn.Module:
        """Return model global_net."""
        return self._global_net

    @global_net.setter
    def global_net(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """Set model global_net.

        Args:
            state_dict: dictionary of the state to set the model global_net to.
        """
        self.global_net.load_state_dict(state_dict, strict=True)

    @property
    def local_net(self) -> nn.Module:
        """Return model local_net."""
        return self._local_net

    @local_net.setter
    def local_net(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """Set model local_net.

        Args:
            state_dict: dictionary of the state to set the model local_net to.
        """
        self.local_net.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        """Get global_net parameters 

        Returns
        -------
            global_net parameters
        """
        return [
            val.cpu().numpy()
            for val in [
                *self.global_net.state_dict().values()
            ]
        ]
        
    def personalise_last_module(self):
        last_child = list(self.global_net.children())[-1]
        self._local_net = nn.Sequential(last_child, *(list(self.local_net.children())))
        self._global_net = nn.Sequential(*(list(self.global_net.children())[:-1]))

    def set_parameters(self, state_dict: Dict[str, Tensor]) -> None:
        """Set model parameters.

        Args:
            state_dict: dictionary of the state to set the model to.
        """
        ordered_state_dict = OrderedDict(self.state_dict().copy())
        # Update with the values of the state_dict
        ordered_state_dict.update(dict(state_dict.items()))
        self.load_state_dict(ordered_state_dict, strict=False)

    def enable_local_net(self) -> None:
        """Enable gradient tracking for the local_net parameters."""
        for param in self.local_net.parameters():
            param.requires_grad = True

    def enable_global_net(self) -> None:
        """Enable gradient tracking for the global_net parameters."""
        for param in self.global_net.parameters():
            param.requires_grad = True

    def disable_local_net(self) -> None:
        """Disable gradient tracking for the local_net parameters."""
        for param in self.local_net.parameters():
            param.requires_grad = False

    def disable_global_net(self) -> None:
        """Disable gradient tracking for the global_net parameters."""
        for param in self.global_net.parameters():
            param.requires_grad = False

    def forward(self, inputs: Any) -> Any:
        """Forward inputs through the global_net and the local_net."""
        x = self.global_net(inputs)
        return self.local_net(x)
    

class ModelManager(ABC):
    """Manager for models with global_net/local_net split."""

    def __init__(
        self,
        client_id: int,
        config: DictConfig,
        model_split_class: Type[Any],  # ModelSplit
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
            model_split_class: Class to be used to split the model into global_net and local_net\
                (concrete implementation of ModelSplit).
        """
        super().__init__()

        self.client_id = client_id
        self.config = config
        self._model = model_split_class(self._create_model())
        
    def _get_eig_vals(self, feats):
        # center features
        feats = feats - torch.mean(feats, dim=0)
        avg_cov_feat = None
        for idx in range(feats.shape[0]):
            # build feature cov matrix
            cov_feat = torch.mm(feats[idx].unsqueeze(1), feats[idx].unsqueeze(1).t())
            # average cov rep
            if avg_cov_feat is None:
                avg_cov_feat = cov_feat
            else:
                avg_cov_feat += cov_feat
        avg_cov_feat /= feats.shape[0]

        _, eig_vals, _ = torch.linalg.svd(avg_cov_feat) # for symmetric matrix, eig_vals == singular values
        return eig_vals.numpy()
    

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Return model to be splitted into local_net and global_net."""

    @abstractmethod
    def train(
        self,
        epochs: int = 1,
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Args:
            epochs: number of training epochs.

        Returns
        -------
            Dict containing the train metrics.
        """

    @abstractmethod
    def test(
        self,
    ) -> Dict[str, float]:
        """Test the model maintained in self.model.

        Returns
        -------
            Dict containing the test metrics.
        """

    @abstractmethod
    def train_dataset_size(self) -> int:
        """Return train data set size."""

    @abstractmethod
    def test_dataset_size(self) -> int:
        """Return test data set size."""

    @abstractmethod
    def total_dataset_size(self) -> int:
        """Return total data set size."""

    @property
    def model(self) -> nn.Module:
        """Return model."""
        return self._model

        
