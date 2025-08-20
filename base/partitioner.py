from flwr_datasets.partitioner import Partitioner
from datasets import Dataset
from typing import Dict, List, Optional, Literal, Any
import datasets
import numpy as np
from collections import defaultdict
from flwr_datasets.common.typing import NDArray
import warnings
from flwr_datasets.partitioner import DirichletPartitioner, PathologicalPartitioner
from omegaconf import DictConfig



class DirichletSkewedPartitioner(Partitioner):
    def __init__(self, num_partitions: int, rich_client_ratio: int,  alpha_rich=5.0, alpha_poor=0.1, seed: Optional[int] = 42) -> None:
        """
        Args:
            num_partitions (int): Number of clients.
            rich_clients (List[int]): Clients who get many labels (less skewed).
            alpha_rich (float): Dirichlet alpha for rich clients (e.g., 5.0).
            alpha_poor (float): Dirichlet alpha for label-skewed clients (e.g., 0.1).
        """
        super().__init__()
        self._num_partitions = num_partitions
        self.alpha_rich = alpha_rich
        self.alpha_poor = alpha_poor
        self.rich_client_ratio = rich_client_ratio
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_indices_determined = False
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)

    def call(self, dataset: Dataset) -> Dict[int, List[int]]:
        if self._partition_id_to_indices_determined:
            return
        
        label_list = dataset["label"]
        num_classes = len(set(label_list))
        
        # Group sample indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(label_list):
            class_indices[label].append(idx)
            
        num_rich = int(self.rich_client_ratio * self._num_partitions)
        all_clients = list(range(self._num_partitions))
        self._rng.shuffle(all_clients)
        self.rich_clients = set(all_clients[:num_rich])
        self.poor_clients = set(all_clients[num_rich:])

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


class VariablePathologicalPartitioner(Partitioner):
    
    def __init__(  # pylint: disable=R0917
        self,
        num_partitions: int,
        partition_by: str,
        num_classes_per_partition: List[int],
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        self._num_partitions = num_partitions
        self._partition_by = partition_by
        self._num_classes_per_partition = num_classes_per_partition
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)

        # Utility attributes
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_unique_labels: dict[int, list[Any]] = {
            pid: [] for pid in range(self._num_partitions)
        }
        self._unique_labels: list[Any] = []
        # Count in how many partitions the label is used
        self._unique_label_to_times_used_counter: dict[Any, int] = {}
        self._partition_id_to_indices_determined = False


    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            The index that corresponds to the requested partition.

        Returns
        -------
        dataset_partition : Dataset
            Single partition of a dataset.
        """
        # The partitioning is done lazily - only when the first partition is
        # requested. Only the first call creates the indices assignments for all the
        # partition indices.
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])



    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    def _determine_partition_id_to_indices_if_needed(self) -> None:
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return
        self._determine_partition_id_to_unique_labels()
        assert self._unique_labels is not None
        self._count_partitions_having_each_unique_label()

        labels = np.asarray(self.dataset[self._partition_by])
        self._check_correctness_of_unique_label_to_times_used_counter(labels)
        for partition_id in range(self._num_partitions):
            self._partition_id_to_indices[partition_id] = []

        unused_labels = []
        for unique_label in self._unique_labels:
            if self._unique_label_to_times_used_counter[unique_label] == 0:
                unused_labels.append(unique_label)
                continue
            # Get the indices in the original dataset where the y == unique_label
            unique_label_to_indices = np.where(labels == unique_label)[0]

            split_unique_labels_to_indices = np.array_split(
                unique_label_to_indices,
                self._unique_label_to_times_used_counter[unique_label],
            )

            split_index = 0
            for partition_id in range(self._num_partitions):
                if unique_label in self._partition_id_to_unique_labels[partition_id]:
                    self._partition_id_to_indices[partition_id].extend(
                        split_unique_labels_to_indices[split_index]
                    )
                    split_index += 1

        if len(unused_labels) >= 1:
            warnings.warn(
                f"Classes: {unused_labels} will NOT be used due to the chosen "
                f"configuration. If it is undesired behavior consider setting"
                f" 'first_class_deterministic_assignment=True' which in case when"
                f" the number of classes is smaller than the number of partitions will "
                f"utilize all the classes for the created partitions.",
                stacklevel=1,
            )
        if self._shuffle:
            for indices in self._partition_id_to_indices.values():
                # In place shuffling
                self._rng.shuffle(indices)

        self._partition_id_to_indices_determined = True

    def _check_num_partitions_correctness_if_needed(self) -> None:
        """Test num_partitions when the dataset is given (in load_partition)."""
        if not self._partition_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    "The number of partitions needs to be smaller than the number of "
                    "samples in the dataset."
                )

    def _determine_partition_id_to_unique_labels(self) -> None:
        """Determine the assignment of unique labels to the partitions."""
        self._unique_labels = sorted(self.dataset.unique(self._partition_by))
        num_unique_classes = len(self._unique_labels)

        if max(self._num_classes_per_partition) > num_unique_classes:
            raise ValueError(
                f"One of the specified `num_classes_per_partition`"
                f"={max(self._num_classes_per_partition)} is greater than the number "
                f"of unique classes in the given dataset={num_unique_classes}. "
                f"Reduce the `num_classes_per_partition` or make use different dataset "
                f"to apply this partitioning."
            )
        for partition_id in range(self._num_partitions):
            labels = self._rng.choice(
                self._unique_labels,
                size=self._num_classes_per_partition[partition_id],
                replace=False,
            ).tolist()
            self._partition_id_to_unique_labels[partition_id] = labels

    def _count_partitions_having_each_unique_label(self) -> None:
        """Count the number of partitions that have each unique label.

        This computation is based on the assignment of the label to the partition_id in
        the `_determine_partition_id_to_unique_labels` method.
        Given:
        * partition 0 has only labels: 0,1 (not necessarily just two samples it can have
          many samples but either from 0 or 1)
        *  partition 1 has only labels: 1, 2 (same count note as above)
        * and there are only two partitions then the following will be computed:
        {
          0: 1,
          1: 2,
          2: 1
        }
        """
        for unique_label in self._unique_labels:
            self._unique_label_to_times_used_counter[unique_label] = 0
        for unique_labels in self._partition_id_to_unique_labels.values():
            for unique_label in unique_labels:
                self._unique_label_to_times_used_counter[unique_label] += 1

    def _check_correctness_of_unique_label_to_times_used_counter(
        self, labels: NDArray
    ) -> None:
        """Check if partitioning is possible given the presence requirements.

        The number of times the label can be used must be smaller or equal to the number
        of times that the label is present in the dataset.
        """
        for unique_label in self._unique_labels:
            num_unique = np.sum(labels == unique_label)
            if self._unique_label_to_times_used_counter[unique_label] > num_unique:
                raise ValueError(
                    f"Label: {unique_label} is needed to be assigned to more "
                    f"partitions "
                    f"({self._unique_label_to_times_used_counter[unique_label]})"
                    f" than there are samples (corresponding to this label) in the "
                    f"dataset ({num_unique}). Please decrease the `num_partitions`, "
                    f"`num_classes_per_partition` to avoid this situation, "
                    f"or try `class_assignment_mode='deterministic'` to create a more "
                    f"even distribution of classes along the partitions. "
                    f"Alternatively use a different dataset if you can not adjust"
                    f" the any of these parameters."
                )
                

class FedPerPartitioner(Partitioner):
    def __init__(self, num_partitions: int, num_classes_per_partition: int, seed: Optional[int] = 42) -> None:
        super().__init__()
        self._num_partitions = num_partitions
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_indices_determined = False
        self._seed = seed
        self.num_classes_per_partition = num_classes_per_partition
        self._rng = np.random.default_rng(seed=self._seed)

    def call(self, dataset: Dataset) -> Dict[int, List[int]]:
        if self._partition_id_to_indices_determined:
            return
        
        num_classes = len(np.unique(dataset['label']))
        class_to_indices = {i: [] for i in range(num_classes)}
        for idx, label in enumerate(dataset['label']):
            class_to_indices[label].append(idx)
            
        # for class_id in class_to_indices:
        #     self._rng.shuffle(class_to_indices[class_id])
        
        sample_number = sum([len(idxs) for idxs in class_to_indices.values()])
        shard_per_class = int(self.num_classes_per_partition * self.num_partitions/ num_classes)
        samples_per_user = int(sample_number / self.num_partitions)
        
        dict_users = defaultdict(list)
        for label in class_to_indices.keys():
            x = np.array(class_to_indices[label])
            shards = np.array_split(x, shard_per_class)
            class_to_indices[label] = [shard.tolist() for shard in shards]
            
        rand_set_all = list(range(num_classes)) * shard_per_class
        self._rng.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((self.num_partitions, -1))
        
        for cid in range(self.num_partitions):
            rand_set_label = rand_set_all[cid]
            rand_set = []
            for label in rand_set_label:
                idx = np.random.choice(len(class_to_indices[label]), replace=False)
                rand_set.append(class_to_indices[label].pop(idx))
            dict_users[cid] = np.concatenate(rand_set)

        self._partition_id_to_indices = dict_users
        self._partition_id_to_indices_determined = True
        
    @property
    def num_partitions(self) -> int:
        return self._num_partitions
    
    def load_partition(self, partition_id: int) -> Dataset:
        """Load a specific partition by its ID."""
        self.call(self.dataset)
        if partition_id < 0 or partition_id >= self._num_partitions:
            raise ValueError(f"Partition ID {partition_id} is out of bounds.")
        return self.dataset.select(self._partition_id_to_indices[partition_id])
    
    
def load_partitioner(cfg: DictConfig, ) -> Partitioner:
    if cfg.dataset.partitioner.name == "dirichlet":
        train_partitioner = DirichletPartitioner(alpha=cfg.dataset.partitioner.alpha, num_partitions=cfg.num_clients, partition_by="label", seed=cfg.seed, min_partition_size=1)
        test_partitioner = DirichletPartitioner(alpha=cfg.dataset.partitioner.alpha, num_partitions=cfg.num_clients, partition_by="label", seed=cfg.seed, min_partition_size=1)
        
    elif cfg.dataset.partitioner.name == "dirichletskewed":
        test_partitioner = DirichletSkewedPartitioner(num_partitions=cfg.num_clients, rich_client_ratio=cfg.dataset.partitioner.rich_client_ratio, alpha_rich=cfg.dataset.partitioner.alpha_rich,  alpha_poor=cfg.dataset.partitioner.alpha_poor, seed=cfg.seed)
        train_partitioner = DirichletSkewedPartitioner(num_partitions=cfg.num_clients, rich_client_ratio=cfg.dataset.partitioner.rich_client_ratio, alpha_rich=cfg.dataset.partitioner.alpha_rich,  alpha_poor=cfg.dataset.partitioner.alpha_poor, seed=cfg.seed)
        
    elif cfg.dataset.partitioner.name == "variable_pathological":
        train_partitioner = VariablePathologicalPartitioner(
            num_partitions=cfg.num_clients,
            partition_by="label",
            num_classes_per_partition=cfg.dataset.partitioner.num_classes_per_partition,
            shuffle=True,
            seed=cfg.seed,
        )
        test_partitioner = VariablePathologicalPartitioner(
            num_partitions=cfg.num_clients,
            partition_by="label",
            num_classes_per_partition=cfg.dataset.partitioner.num_classes_per_partition,
            shuffle=True,
            seed=cfg.seed,
        )
    elif cfg.dataset.partitioner.name == "pathological":
        train_partitioner = PathologicalPartitioner(num_partitions=cfg.num_clients, 
                                              partition_by='label', 
                                              num_classes_per_partition=cfg.dataset.partitioner.num_classes_per_partition,
                                              class_assignment_mode='random',
                                              seed=cfg.seed)
        test_partitioner = PathologicalPartitioner(num_partitions=cfg.num_clients, 
                                              partition_by='label', 
                                              num_classes_per_partition=cfg.dataset.partitioner.num_classes_per_partition,
                                              class_assignment_mode='random',
                                              seed=cfg.seed)
    elif cfg.dataset.partitioner.name == "fedrep":
        train_partitioner = FedPerPartitioner(num_partitions=cfg.num_clients, 
                                              num_classes_per_partition=cfg.dataset.partitioner.num_classes_per_partition,
                                              seed=cfg.seed)
        test_partitioner = FedPerPartitioner(num_partitions=cfg.num_clients, 
                                              num_classes_per_partition=cfg.dataset.partitioner.num_classes_per_partition,
                                              seed=cfg.seed)
        
    return train_partitioner, test_partitioner