from flwr_datasets.partitioner import Partitioner
from datasets import Dataset
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict

class DirichletSkewedPartitioner(Partitioner):
    def __init__(self, num_partitions: int, rich_clients: List[int], alpha_rich=5.0, alpha_poor=0.1, seed: Optional[int] = 42) -> None:
        """
        Args:
            num_partitions (int): Number of clients.
            rich_clients (List[int]): Clients who get many labels (less skewed).
            alpha_rich (float): Dirichlet alpha for rich clients (e.g., 5.0).
            alpha_poor (float): Dirichlet alpha for label-skewed clients (e.g., 0.1).
        """
        super().__init__()
        self._num_partitions = num_partitions
        self.rich_clients = set(rich_clients)
        self.alpha_rich = alpha_rich
        self.alpha_poor = alpha_poor
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_indices_determined = False
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)

    def call(self, dataset: Dataset) -> Dict[int, Dataset]:
        if self._partition_id_to_indices_determined:
            return
        
        label_list = dataset["label"]
        num_classes = len(set(label_list))
        
        # Group sample indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(label_list):
            class_indices[label].append(idx)

        # Prepare per-client index lists
        client_indices = defaultdict(list)

        for cls in range(num_classes):
            indices = class_indices[cls]
            self._rng.shuffle(indices)

            # Dirichlet distribution over clients for this class
            alphas = [
                self.alpha_rich if client in self.rich_clients else self.alpha_poor
                for client in range(self._num_partitions)
            ]
            proportions = self._rng.dirichlet(alphas)

            # Compute the split sizes so that all indices are distributed
            split_sizes = np.floor(proportions * len(indices)).astype(int)
            # Distribute the remainder to clients with largest fractional parts
            remainder = len(indices) - split_sizes.sum()
            if remainder > 0:
                fractional = proportions * len(indices) - split_sizes
                top_clients = np.argsort(fractional)[-remainder:]
                for client_id in top_clients:
                    split_sizes[client_id] += 1

            # Assign indices to clients
            start = 0
            for client_id, size in enumerate(split_sizes):
                if size > 0:
                    client_indices[client_id].extend(indices[start : start + size])
                start += size

        # Convert to Datasets
        self._partition_id_to_indices = {
            client_id: idxs for client_id, idxs in client_indices.items()
        }

    @property
    def num_partitions(self) -> int:
        return self._num_partitions
    
    def load_partition(self, partition_id: int) -> Dataset:
        """Load a specific partition by its ID."""
        self.call(self.dataset)
        if partition_id < 0 or partition_id >= self._num_partitions:
            raise ValueError(f"Partition ID {partition_id} is out of bounds.")
        return self.dataset.select(self._partition_id_to_indices[partition_id])