from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.common import event, EventType
import os
# Import the parent class (adjust the import path based on your project structure)
from flwr_datasets.federated_dataset import FederatedDataset
import datasets
from datasets import concatenate_datasets, ClassLabel, DatasetDict, Image, Value, Features

class BDD100KDataset(FederatedDataset):
    """
    BDD100K Dataset class that derives from FederatedDataset.
    
    Expected directory structure:
    100k/
    ├── images/
    │   ├── train/
    │   ├── test/
    │   └── val/
    └── labels/
        ├── train/
        ├── test/
        └── val/
    """
    
    def __init__(self, 
                 dataset: str,
                 partitioners: str,
                 root_path: str = "100k/",
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 preprocessor=None,
                 **kwargs):
        """
        Initialize DBB100K dataset.
        
        Args:
            root_path: Path to the 100k dataset directory
            shuffle: Whether to shuffle the dataset splits
            seed: Random seed for shuffling
            preprocessor: Function to preprocess the loaded dataset
            **kwargs: Additional arguments passed to parent class
        """
        self.root_path = Path(root_path)
        self._shuffle = shuffle
        self._seed = seed
        self._preprocessor = preprocessor
        
        # Validate root path exists
        if not self.root_path.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_path}")
        
        # Initialize parent class attributes that are expected
        self._dataset_prepared = False
        self._event = {}
        self._dataset = None
        
        super(BDD100KDataset, self).__init__(dataset=dataset, partitioners=partitioners)
    
    
    def _load_dbb100k_dataset(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Load the DBB100K dataset from local directory structure.
        
        Returns:
            Dict with splits as keys and list of (image_path, label_path) tuples as values
        """
        dataset_dict = {}
        expected_splits = ['train', 'test', 'val']
        
        for split in expected_splits:
            images_path = self.root_path / "images" / split
            labels_path = self.root_path / "labels" / split
            
            # Skip splits that don't exist
            if not images_path.exists() or not labels_path.exists():
                print(f"Warning: Split '{split}' not found, skipping...")
                continue
            
            # Get all image files
            image_extensions = {'.jpg'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(list(images_path.glob(f"*{ext}")))
                image_files.extend(list(images_path.glob(f"*{ext.upper()}")))
            
            image_files.sort()
            
            # Get corresponding label files
            split_samples = []
            missing_labels = []
            
            for img_file in image_files:
                # Try different label file extensions
                label_extensions = ['.json']
                label_file = None
                
                for ext in label_extensions:
                    potential_label = labels_path / (img_file.stem + ext)
                    if potential_label.exists():
                        label_file = potential_label
                        break
                
                with open(label_file, "r") as f:
                    label_data = json.load(f)['attributes']['scene']
                    
                if label_file:
                    split_samples.append((str(img_file), str(label_data)))
                else:
                    missing_labels.append(img_file.name)
            
            if missing_labels:
                print(f"Warning: {len(missing_labels)} images in '{split}' have no corresponding labels")
            
            images, labels = zip(*split_samples)
            dictionnaries_form = {
                "img": list(images),
                "label": list(labels),
            }
            
            dataset_dict[split] = datasets.Dataset.from_dict(dictionnaries_form)
            print(f"Loaded {len(split_samples)} samples for '{split}' split")
        
        if not dataset_dict:
            raise ValueError(f"No valid splits found in {self.root_path}")
        
        if "train" in dataset_dict and "val" in dataset_dict:
            dataset_dict["train"] = concatenate_datasets([dataset_dict["train"], dataset_dict["val"]])
            del dataset_dict["val"]
            print(f"Merged train + val → {len(dataset_dict['train'])} samples")

        return dataset_dict
    
    def _shuffle_dataset(self, dataset_dict: Dict[str, Dataset]) -> Dict[str, List]:
        """
        Shuffle all splits in the dataset.
        
        Args:
            dataset_dict: Dictionary with splits and their samples
            
        Returns:
            Shuffled dataset dictionary
        """
        shuffled_dict = {}
        
        for split, samples in dataset_dict.items():
            shuffled_dict[split] = samples.shuffle(seed=self._seed)
        
        return shuffled_dict

    def _prepare_dataset(self) -> None:
        """Prepare the dataset (prior to partitioning) by loading from local DBB100K structure.

            Run only ONCE when triggered by load_* function. (In future more control whether
            this should happen lazily or not can be added). The operations done here should
            not happen more than once.

            It is controlled by a single flag, `_dataset_prepared` that is set True at the
            end of the function.

            Notes
            -----
            This method loads the DBB100K dataset from the local directory structure:
            100k/images/{train,test,val}/ and 100k/labels/{train,test,val}/
            The shuffling should happen before any resplitting operations.
        """
        # Load dataset from local DBB100K structure
        self._dataset = self._load_dbb100k_dataset()
        
        if not isinstance(self._dataset, dict):
            raise ValueError(
                f"The dataset loading failed. Expected dict with splits, "
                f"but got: {type(self._dataset)}."
            )
        
        if self._shuffle:
            # Shuffle all splits in the dataset
            self._dataset = self._shuffle_dataset(self._dataset)
        
        if self._preprocessor:
            self._dataset = self._preprocessor(self._dataset)
        
        available_splits = list(self._dataset.keys())
        self._event["load_split"] = {split: False for split in available_splits}
        self._dataset_prepared = True
        
    def load_partition(self, partition_id: int, split: Optional[str] = None) -> Dataset:
        """Load the partition specified by the idx in the selected split.

        The dataset is downloaded only when the first call to `load_partition` or
        `load_split` is made.

        Parameters
        ----------
        partition_id : int
            Partition index for the selected split, idx in {0, ..., num_partitions - 1}.
        split : Optional[str]
            Name of the (partitioned) split (e.g. "train", "test"). You can skip this
            parameter if there is only one partitioner for the dataset. The name will be
            inferred automatically. For example, if `partitioners={"train": 10}`, you do
            not need to provide this argument, but if `partitioners={"train": 10,
            "test": 100}`, you need to set it to differentiate which partitioner should
            be used.
            The split names you can choose from vary from dataset to dataset. You need
            to check the dataset on the `Hugging Face Hub`<https://huggingface.co/
            datasets>_ to see which splits are available. You can resplit the dataset
            by using the `preprocessor` parameter (to rename, merge, divide, etc. the
            available splits).

        ReturnsZ
        -------
        partition : Dataset
            Single partition from the dataset split.
        """
        if not self._dataset_prepared:
            self._prepare_dataset()
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        if split is None:
            self._check_if_no_split_keyword_possible()
            split = list(self._partitioners.keys())[0]
        self._check_if_split_present(split)
        self._check_if_split_possible_to_federate(split)
        partitioner: Partitioner = self._partitioners[split]
        self._assign_dataset_to_partitioner(split)
        partition = partitioner.load_partition(partition_id)
        if not self._event["load_partition"][split]:
            event(
                EventType.LOAD_PARTITION_CALLED,
                {
                    "federated_dataset_id": id(self),
                    "dataset_name": self._dataset_name,
                    "split": split,
                    "partitioner": partitioner.__class__.__name__,
                    "num_partitions": partitioner.num_partitions,
                },
            )
            self._event["load_partition"][split] = True

        return DBB100KTorchDataset(partition['img'], partition['label'])
    
class DBB100KTorchDataset(Dataset):
    """
    PyTorch Dataset wrapper for DBB100K data.
    """
    
    def __init__(self, 
                 image_files: list,
                 labels: List[str]):

        self.image_files = image_files
        self.labels = labels
      

        class_names = sorted(set(labels))
        self.class_label = ClassLabel(names=class_names)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load label
        label_path = self.label_files[idx]
        label = self._load_label(label_path)
        label_id = self.class_label.str2int(label)
        return image, label_id
    
    def _load_label(self, label_path: str) -> Any:
        """
        Load label from file. Handles different label formats.
        """
        if label_path.endswith('.json'):
            with open(label_path, 'r') as f:
                label = json.load(f)
        elif label_path.endswith('.txt'):
            with open(label_path, 'r') as f:
                label = f.read().strip()
        else:
            # For other formats, read as text
            with open(label_path, 'r') as f:
                label = f.read().strip()
        
        return label['attributes']['scene']


def load_dbb100k_dataset(root_path: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Load the DBB100K dataset from local directory structure.
        
        Returns:
            Dict with splits as keys and list of (image_path, label_path) tuples as values
        """
        dataset_dict = {}
        expected_splits = ['train', 'test', 'val']
        root_path = Path(root_path)
        label_extension = ".json"
        
        all_labels= set()
        for split in expected_splits:
            labels_path = root_path / "labels" / split
            for lbl_file in os.listdir(labels_path):
                label_file = labels_path / lbl_file
                with open(label_file, "r") as f:
                    label_data = json.load(f)['attributes']['scene']
                    all_labels.add(label_data)
        
        all_labels = sorted(all_labels)
        label_feature = ClassLabel(names=all_labels)

        for split in expected_splits:
            images_path = root_path / "images" / split
            labels_path = root_path / "labels" / split
            
            # Skip splits that don't exist
            if not images_path.exists() or not labels_path.exists():
                print(f"Warning: Split '{split}' not found, skipping...")
                continue
            
            # Get all image files
            image_extension = '.jpg'
            image_files = []
            
            image_files.extend(list(images_path.glob(f"*{image_extension}")))
            image_files.extend(list(images_path.glob(f"*{image_extension.upper()}")))
        
            image_files.sort()
            
            # Get corresponding label files
            split_samples = []
            all_labels = set()
            for img_file in image_files:
                # Try different label file extensions
                
                label_file = labels_path / (img_file.stem + label_extension)
                    
                with open(label_file, "r") as f:
                    label_data = json.load(f)['attributes']['scene']

                split_samples.append({'img': str(img_file), 'label': str(label_data)})
            
            features = Features({
                    "img": Image(),
                    "label":label_feature,
            })
        
            dataset_dict[split] = datasets.Dataset.from_list(split_samples, features=features)
            print(f"Loaded {len(split_samples)} samples for '{split}' split")
        
        if not dataset_dict:
            raise ValueError(f"No valid splits found in {root_path}")
        
        if "train" in dataset_dict and "val" in dataset_dict:
            dataset_dict["train"] = concatenate_datasets([dataset_dict["train"], dataset_dict["val"]])
            del dataset_dict["val"]
            print(f"Merged train + val → {len(dataset_dict['train'])} samples")
           
        for split in dataset_dict:
            dataset_dict[split].set_format(type="torch", columns=["img", "label"])
        return DatasetDict(dataset_dict)