from typing import List, Dict, Union
import torch
from torch.utils.data import Dataset
import datasets

# Workaround toolkit misreporting available disk space.
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True
from datasets import load_dataset, load_from_disk
from datasets.builder import DatasetBuildError
from transformers import AutoTokenizer
from embedding.preprocessing_utils import (
    perturb_tokens,
    get_special_tokens_mask,
    get_pooling_mask,
    pre_process_codesearchnet_train,
    pre_process_codesearchnet_test,
    pre_process_gfg,
    pre_process_the_stack,
)
from embedding.constants import MASK_TOKEN, PAD_TOKEN, SEPARATOR_TOKEN, CLS_TOKEN

DATASET_NAME_TO_PREPROCESSING_FUNCTION = {
    "the-stack": {
        "train": pre_process_the_stack,
        "test": pre_process_the_stack,
    },
    "code_search_net": {
        "train": pre_process_codesearchnet_train,
        "test": pre_process_codesearchnet_test,
    },
    "gfg": {
        "train": pre_process_gfg,
        "test": pre_process_gfg,
    },
}


class RandomlyPairedDataset(Dataset):
    """Indexed dataset class with randomly picked negative pairs."""

    def __init__(self, base_dataset: datasets.Dataset) -> None:
        """Intanstiates an indexed dataset wrapping a base data source.
        We use this class to be able to get examples from the dataset including negative pairs.

        Args:
            base_dataset (datasets.Dataset): Base indexed data source.
        """
        self.data_source = base_dataset

    def __len__(self) -> int:
        """Returns the length of the dataset which matches that of the base data source.

        Returns:
            int: Dataset length.
        """
        return len(self.data_source)

    def __getitem__(self, i: int) -> List[Dict]:
        """Reads from the base dataset and returns an addition random entry that serves as negative example.

        Args:
            i (int): Index to be read.

        Returns:
            List[Dict]: Pair of examples. The example indexed by i is returned along with a different random point.
        """
        rand_idx = torch.randint(0, len(self.data_source), (1,)).item()
        while rand_idx == i:
            rand_idx = torch.randint(0, len(self.data_source), (1,)).item()

        example = self.data_source[i]
        negative_example = self.data_source[rand_idx]

        return example["source"], example["target"], negative_example["source"]


class PairedDataset(Dataset):
    """Indexed dataset class yielding source/target pairs."""

    def __init__(self, base_dataset: datasets.Dataset) -> None:
        """Intanstiates an indexed dataset wrapping a base data source.
        We use this class to be able to get paired examples from the base dataset.
        The base dataset must be pre-processed to include the fields 'source' and 'target'.

        Args:
            base_dataset (datasets.Dataset): Base indexed pre-processed data source.
        """
        self.data_source = base_dataset

    def __len__(self) -> int:
        """Returns the length of the dataset which matches that of the base data source.

        Returns:
            int: Dataset length.
        """
        return len(self.data_source)

    def __getitem__(self, i: int) -> List[Dict]:
        """Reads from the base dataset and returns a paier of examples.

        Args:
            i (int): Index to be read.

        Returns:
            List[Dict]: Pair of examples. The 'source' and 'target' fields of the example indexed by i are returned.
        """

        example = self.data_source[i]

        return example["source"], example["target"]


def get_dataset(
    dataset_name: str,
    path_to_cache: str,
    split: str,
    maximum_raw_length: int,
    force_preprocess: bool = False,
    maximum_row_cout: int = None,
) -> Union[PairedDataset, RandomlyPairedDataset]:
    """Get dataset instance.

    Args:
        dataset_name (str): Name of the base dataset.
        path_to_cache (str): Path to the base dataset.
        split (str): data split in {'train', 'valid', 'test'}.
        maximum_raw_length (int, optional): Maximum length of the raw entries from the source dataset.
        force_preprocess (bool, optional): Whether to force pre-processing. Defaults to False.
        maximum_row_cout (int, optional) = Maximum size of the dataset in term of row count. Defaults to None.

    Returns:
        dataset: An indexed dataset object.
    """
    try:
        base_dataset = load_dataset(
            dataset_name,
            use_auth_token=True,
            cache_dir=path_to_cache,
            split=split,
        )
    except DatasetBuildError:
        # Try to specify data files. Specific for The Stack.
        base_dataset = load_dataset(
            dataset_name,
            use_auth_token=True,
            cache_dir=path_to_cache,
            data_files="sample.parquet",
            split=split,
        )
    except FileNotFoundError:
        # Try to load from disk if above failed.
        base_dataset = load_from_disk(path_to_cache)

    if force_preprocess:
        base_dataset.cleanup_cache_files()

    base_dataset = base_dataset.shuffle(seed=42)

    if maximum_row_cout is not None:
        base_dataset = base_dataset.select(
            range(min(len(base_dataset), maximum_row_cout))
        )

    if "train" in split.lower():
        split_preproc_key = "train"
    else:
        split_preproc_key = "test"

    try:
        pre_proc_fn = DATASET_NAME_TO_PREPROCESSING_FUNCTION[dataset_name][
            split_preproc_key
        ]
    except KeyError:
        for k in DATASET_NAME_TO_PREPROCESSING_FUNCTION.keys():
            if "the-stack" in dataset_name.lower() and "the-stack" in k:
                pre_proc_fn = DATASET_NAME_TO_PREPROCESSING_FUNCTION[k][
                    split_preproc_key
                ]

    base_dataset = base_dataset.map(pre_proc_fn(maximum_raw_length), num_proc=96)

    base_dataset = base_dataset.shuffle(seed=42)

    if "train" in split_preproc_key:
        return RandomlyPairedDataset(base_dataset)
    else:
        return PairedDataset(base_dataset)


def prepare_tokenizer(tokenizer_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=True)

    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.add_special_tokens({"sep_token": SEPARATOR_TOKEN})
    tokenizer.add_special_tokens({"cls_token": CLS_TOKEN})
    tokenizer.add_special_tokens({"mask_token": MASK_TOKEN})
    return tokenizer


class TrainCollator:
    """Train collator object mapping sequences of items from dataset instance
    into batches of IDs and masks used for training models.
    """

    def __init__(
        self,
        tokenizer_path: str,
        maximum_length: int,
        mlm_masking_probability: float,
        contrastive_masking_probability: float,
        ignore_contrastive_loss_data: bool = False,
        **kwargs,
    ) -> None:
        """Creates instance of collator.

        Args:
            tokenizer_path (str): Path to tokenizer.
            maximum_length (int): Truncating length of token sequences.
            mlm_masking_probability (float): Masking probability for MLM objective.
            contrastive_masking_probability (float): Masking probability for contrastive objective.
            ignore_contrastive_loss_data (bool, optional): Do not add append positive pairs to batch. Defaults to False.
        """
        self.mlm_masking_probability = mlm_masking_probability
        self.contrastive_masking_probability = contrastive_masking_probability
        self.maximum_length = maximum_length
        self.ignore_contrastive_loss_data = ignore_contrastive_loss_data

        self.tokenizer = prepare_tokenizer(tokenizer_path)

        self.sep_token_id = self.tokenizer.get_vocab()[self.tokenizer.sep_token]
        self.pad_token_id = self.tokenizer.get_vocab()[self.tokenizer.pad_token]
        self.mask_token_id = self.tokenizer.get_vocab()[self.tokenizer.mask_token]
        self.cls_token_id = self.tokenizer.get_vocab()[self.tokenizer.cls_token]

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Maps list of triplets of examples to batches of token ids, masks, and labels used for training.
        The firt two elements in a triplet correspond to neighbor chunkes from the same file. The third
        element corresponds to a chunk from a random file.

        Args:
            batch (List[Dict]): List of pairs of examples.

        Returns:
            Dict[str, torch.Tensor]: Batches of tokens, masks, and labels.
        """
        source_list = [
            el[0] for el in batch
        ]  # el[0] is the first half of a code snippet.
        # Following are the labels for the seq relationship loss: 0 -> negative pair, 1 -> positive pair.
        seq_relationship_labels = torch.randint(0, 2, (len(batch),)).long()
        target_list = [
            # seq_relationship_label==1 -> positive pair -> we take the second half of the code snippet
            # seq_relationship_label==0 -> negative pair -> we take a random code snippet given in el[1]
            el[1] if seq_relationship_labels[i] == 1 else el[2]
            for i, el in enumerate(batch)
        ]

        input_examples_list = [  # Combine source and target w/ template: [CLS] SOURCE [SEP] [TARGET] [SEP]
            f"{CLS_TOKEN}{source_list[i]}{SEPARATOR_TOKEN}{target_list[i]}{SEPARATOR_TOKEN}"
            for i in range(len(batch))
        ]

        input_examples_encoding = self.tokenizer(
            input_examples_list,
            padding="longest",
            max_length=self.maximum_length,
            truncation=True,
            return_tensors="pt",
        )

        input_examples_ids = input_examples_encoding.input_ids
        input_examples_att_mask = (
            input_examples_encoding.attention_mask
        )  # Padding masks.

        special_tokens_mask = get_special_tokens_mask(
            self.tokenizer, input_examples_ids
        )

        input_examples_ids, mlm_labels = perturb_tokens(
            input_examples_ids,
            special_tokens_mask,
            self.mlm_masking_probability,
            self.mask_token_id,
            len(self.tokenizer),
        )  # Dynamically perturbs input tokens and generates corresponding mlm labels.

        if not self.ignore_contrastive_loss_data:
            positive_examples_ids, positive_mlm_labels = perturb_tokens(
                input_examples_ids,
                special_tokens_mask,
                self.contrastive_masking_probability,
                self.mask_token_id,
                len(self.tokenizer),
            )  # Positve examples are independently perturbed versions of the source, used for the contrastive loss.

            input_ids = torch.cat([input_examples_ids, positive_examples_ids], 0)
            attention_mask = torch.cat(
                [input_examples_att_mask, input_examples_att_mask.clone()], 0
            )
            pooling_mask = get_pooling_mask(
                input_ids, self.sep_token_id
            )  # Pooling masks indicate the first [SEP] occurrence, used for seq embedding.
            labels = torch.cat([mlm_labels, positive_mlm_labels], 0)
            next_sentence_label = torch.cat(
                [seq_relationship_labels, seq_relationship_labels.clone()], 0
            )
        else:
            input_ids = input_examples_ids
            attention_mask = input_examples_att_mask
            pooling_mask = get_pooling_mask(
                input_ids, self.sep_token_id
            )  # Pooling masks indicate the first [SEP] occurrence, used for seq embedding.
            labels = mlm_labels
            next_sentence_label = seq_relationship_labels

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pooling_mask": pooling_mask,
            "labels": labels,
            "next_sentence_label": next_sentence_label,
        }


class TestCollator:
    """Test collator object mapping sequences of items from dataset instance
    into batches of IDs and masks used for training models.
    """

    def __init__(self, tokenizer_path: str, maximum_length: int, **kwargs) -> None:
        """Creates instance of collator.

        Args:
            tokenizer_path (str): Path to tokenizer.
            maximum_length (int): Truncating length of token sequences.
        """
        self.maximum_length = maximum_length

        self.tokenizer = prepare_tokenizer(tokenizer_path)

        self.sep_token_id = self.tokenizer.get_vocab()[self.tokenizer.sep_token]
        self.pad_token_id = self.tokenizer.get_vocab()[self.tokenizer.pad_token]
        self.mask_token_id = self.tokenizer.get_vocab()[self.tokenizer.mask_token]
        self.cls_token_id = self.tokenizer.get_vocab()[self.tokenizer.cls_token]

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Maps list of pairs of examples to batches of token ids, masks, and labels used for training.

        Args:
            batch (List[Dict]): List of pairs of examples.

        Returns:
            Dict[str, torch.Tensor]: Batches of tokens and masks.
        """
        source_list = [el[0] for el in batch]
        target_list = [el[1] for el in batch]

        source_examples_list = [
            f"{CLS_TOKEN}{source_list[i]}{SEPARATOR_TOKEN}" for i in range(len(batch))
        ]
        target_examples_list = [
            f"{CLS_TOKEN}{target_list[i]}{SEPARATOR_TOKEN}" for i in range(len(batch))
        ]

        source_target_examples_encoding = self.tokenizer(
            source_examples_list + target_examples_list,
            padding="longest",
            max_length=self.maximum_length,
            truncation=True,
            return_tensors="pt",
        )

        source_target_examples_ids = source_target_examples_encoding.input_ids
        source_target_examples_att_mask = source_target_examples_encoding.attention_mask

        return {
            "source_target_ids": source_target_examples_ids,
            "source_target_att_mask": source_target_examples_att_mask,
            "return_loss": True,  # This is so Trainer.prediction_step() will call Trainer.compute_loss during eval()
        }


class Collator:
    """Object wrapping both train and test collators.
    Decides which collator to use depending on the configuration of the batch.
    Pair of examples --> test instance
    Triplet of examples --> train instances
    """

    def __init__(
        self,
        tokenizer_path: str,
        maximum_length: int,
        mlm_masking_probability: float = 0.5,
        contrastive_masking_probability: float = 0.5,
        ignore_contrastive_loss_data: bool = False,
    ) -> None:
        """Creates instance of collator.

        Args:
            tokenizer_path (str): Path to tokenizer.
            maximum_length (int): Truncating length of token sequences.
            mlm_masking_probability (float, optional): Masking probability for MLM objective. Defaults to 0.5.
            contrastive_masking_probability (float, optional): Masking probability for contrastive objective. Defaults to 0.5.
            ignore_contrastive_loss_data (bool, optional): Do not add append positive pairs to batch. Defaults to False.
        """
        self.train_collator = TrainCollator(
            tokenizer_path=tokenizer_path,
            maximum_length=maximum_length,
            mlm_masking_probability=mlm_masking_probability,
            contrastive_masking_probability=contrastive_masking_probability,
            ignore_contrastive_loss_data=ignore_contrastive_loss_data,
        )
        self.test_collator = TestCollator(
            tokenizer_path=tokenizer_path,
            maximum_length=maximum_length,
        )

        self.vocabulary_size = len(self.train_collator.tokenizer.vocab)
        self.pad_token_id = self.train_collator.pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Maps list of pairs of examples to batches of token ids, masks, and labels used for training.

        Args:
            batch (List[Dict]): List of pairs of examples.

        Returns:
            Dict[str, torch.Tensor]: Batches of tokens and masks.
        """

        if len(batch[0]) == 3:
            return self.train_collator(batch)
        elif len(batch[0]) == 2:
            return self.test_collator(batch)
        else:
            raise AttributeError("Unknown batch configuration.")
