from typing import Dict, List, Union
import torch
from embedding.constants import PADDING_ID_FOR_LABELS
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def perturb_tokens(
    input_ids: torch.Tensor,
    special_tokens_masks: torch.Tensor,
    masking_fraction: float,
    masking_token_id: Union[int, float],
    vocab_size: Union[int, float],
) -> List[torch.Tensor]:
    """Perturb tokens in preparation for MLM loss computation.
    Adapted from:
    https://github.com/huggingface/transformers/blob/d4306daea1f68d8e854b7b3b127878a5fbd53489/src/transformers/data/data_collator.py#L750

    Args:
        input_ids (torch.Tensor): Batch of input tokens IDs.
        special_tokens_masks (torch.Tensor): Masks for special tokens that shouldn't be perturberd.
        masking_fraction (float): Probability of masking ou a given token.
        masking_token_id (Union[int, float]): Token id for masks.
        vocab_size (Union[int, float]): vocab_size used to replace inputs for random words.

    Returns:
        List[torch.Tensor]: Perturbed ids along with label ids.
    """

    perturbed_input_ids = input_ids.clone()
    labels = input_ids.clone()
    # We sample a few tokens in each sequence for MLM training (with probability masking_fraction)
    probability_matrix = torch.full(labels.shape, masking_fraction)

    probability_matrix.masked_fill_(special_tokens_masks, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[
        ~masked_indices
    ] = PADDING_ID_FOR_LABELS  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with masking_token_id
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    perturbed_input_ids[indices_replaced] = masking_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    perturbed_input_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return perturbed_input_ids, labels


def get_special_tokens_mask(
    tokenizer: PreTrainedTokenizerBase, token_ids: torch.Tensor
) -> torch.Tensor:
    """Get masks indicating the positions of special tokens in a batch of token ids.
    Adapted from:
    https://github.com/huggingface/transformers/blob/d4306daea1f68d8e854b7b3b127878a5fbd53489/src/transformers/data/data_collator.py#L759

    Args:
        tokenizer (PreTrainedTokenizerBase): Tokenizer used to gnerate encoding in token_ids.
        token_ids (torch.Tensor): batch of token ids.

    Returns:
        torch.Tensor: Batch of masks with 1 wherever a special token appears and 0 otherwise.
    """

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in token_ids.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    return special_tokens_mask


def truncate_sentences(
    sentence_list: List[str], maximum_length: Union[int, float]
) -> List[str]:
    """Truncates list of sentences to a maximum length.

    Args:
        sentence_list (List[str]): List of sentences to be truncated.
        maximum_length (Union[int, float]): Maximum length of any output sentence.

    Returns:
        List[str]: List of truncated sentences.
    """

    truncated_sentences = []

    for sentence in sentence_list:
        truncated_sentences.append(sentence[:maximum_length])

    return truncated_sentences


def split_sentence(
    sentence: str, maximum_length: Union[int, float] = None
) -> List[str]:
    """Truncates and splits a given sentence.

    Args:
        sentence (str): Input sentence.
        maximum_length (Union[int, float], optional): Maximum length. Defaults to None.

    Returns:
        List[str]: List of pair of sentences, each being a half of the input after truncation.
    """

    if maximum_length is None:
        maximum_length = len(sentence)
    else:
        maximum_length = min(maximum_length, len(sentence))

    half_length = maximum_length // 2

    return sentence[:half_length], sentence[half_length:maximum_length]


def get_pooling_mask(
    input_ids: torch.Tensor, sep_token_id: Union[int, float]
) -> torch.Tensor:
    """Gets pooling masks. For a sequence of input tokens, the mask will be
    a sequence of ones up until the first [SEP] occurrence, and 0 after that.

    Args:
        input_ids (torch.Tensor): Batch of input ids with shape [B, T].
        sep_token_id (Union[int, float]): Id for [SEP] token.

    Returns:
        torch.Tensor: Batch of pooling masks with shape [B, T]
    """
    # idx indicates the first occurrence of sep_token_id per along dim 0 of input_ids
    idx = (input_ids == sep_token_id).float().argmax(1)

    repeated_idx = idx.unsqueeze(1).repeat(1, input_ids.size(1))

    ranges = torch.arange(input_ids.size(1)).repeat(input_ids.size(0), 1)

    pooling_mask = (repeated_idx >= ranges).long()

    return pooling_mask


class pre_process_codesearchnet_train:
    def __init__(self, maximum_length: int) -> None:
        """Pre process code search net data by truncating and splitting code snippets.

        Args:
            maximum_length (int): Max length of code snippets.
        """
        self.maximum_length = maximum_length

    def __call__(self, example: Dict) -> Dict:
        """Reads code string, truncates it and splits in two pieces.

        Args:
            example (Dict): Input data example.

        Returns:
            Dict: Pre-processed example.
        """
        code_str = example["func_code_string"]
        code_str_source, code_str_target = split_sentence(code_str, self.maximum_length)
        example.update({"source": code_str_source, "target": code_str_target})
        return example


class pre_process_codesearchnet_test:
    def __init__(self, maximum_length: int) -> None:
        """Pre process code search net data by truncating and pairing code and docstring.

        Args:
            maximum_length (int): Max length of code snippets.
        """
        self.maximum_length = maximum_length

    def __call__(self, example: Dict) -> Dict:
        """Reads and truncates code and doc strings.

        Args:
            example (Dict): Input data example.

        Returns:
            Dict: Pre-processed example.
        """
        source = example["func_documentation_tokens"]
        source = (" ").join(source)[: self.maximum_length]
        target = example["func_code_string"][: self.maximum_length]
        example.update({"source": source, "target": target})
        return example


class pre_process_gfg:
    def __init__(self, maximum_length: int) -> None:
        """Pre process Python-Java Geeks4Geeks data by truncating and pairing code snippets.

        Args:
            maximum_length (int): Max length of code snippets.
        """
        self.maximum_length = maximum_length

    def __call__(self, example: Dict) -> Dict:
        """Reads and truncates code strings.

        Args:
            example (Dict): Input data example.

        Returns:
            Dict: Pre-processed example.
        """

        source = example["python_func"][: self.maximum_length]
        target = example["java_func"][: self.maximum_length]

        example.update({"source": source, "target": target})

        return example


class pre_process_the_stack:
    def __init__(self, maximum_length: int) -> None:
        """Pre process The Stack data by truncating and splitting code files.

        Args:
            maximum_length (int): Max length of code snippets.
        """
        self.maximum_length = maximum_length

    def __call__(self, example: Dict) -> Dict:
        """Reads, truncates, and splits code strings.

        Args:
            example (Dict): Input data example.

        Returns:
            Dict: Pre-processed example.
        """

        code_str = example["content"]
        code_str_source, code_str_target = split_sentence(code_str, self.maximum_length)
        example.update({"source": code_str_source, "target": code_str_target})
        return example
