from typing import List, Union, Optional
import torch
from embedding.distributed_utils import all_gather


class TempCoef(torch.nn.Module):
    """Module wrapping a temperature coeficient used to compute contrastive losses."""

    def __init__(self, initial_value: float) -> None:
        """Constructs TempCoef instance.

        Args:
            initial_value (float): Startting value of the temperature.
        """
        super().__init__()
        self.temp_coef = torch.nn.Parameter(torch.Tensor([initial_value]))

    def forward(self, logits_matrix: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module: Multiply input tensor by the temperature value.

        Args:
            logits_matrix (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Logits matrix multiplied by temp.
        """
        # Apply learnable temperature factor on similarities
        # Clamping after op to avoid numerical instabilities
        logits_matrix = logits_matrix * self.temp_coef.clamp(1e-4, 30.0)

        return logits_matrix

    def get_temp_coef(self) -> float:
        """Get temperature value.

        Returns:
            float: temperature value.
        """
        return self.temp_coef.data.detach()


def gather_embeddings(
    embedding_1: torch.Tensor, embedding_2: torch.Tensor
) -> List[torch.Tensor]:
    """Gathers embeddings across devices in distributed settings.

    Args:
        embedding_1 (torch.Tensor): First batch of embeddings. Expected shape: [n, d]
        embedding_2 (torch.Tensor): Second batch of embeddings. Expected shape: [n, d]

    Returns:
        torch.Tensor: List of tensors concatenated from all devices, each with shape [n*NUM_DEVICES, d].
    """

    # Creates extra dimension for concatenating batches into single tensor.
    embedding_1 = embedding_1.unsqueeze(1)
    embedding_2 = embedding_2.unsqueeze(1)

    embedding = torch.cat(
        (
            embedding_1,
            embedding_2,
        ),
        1,
    )

    # Gather embeddings across devices
    embedding_dist = all_gather(embedding)

    embedding_1_dist = embedding_dist[:, 0, :]
    embedding_2_dist = embedding_dist[:, 1, :]

    return embedding_1_dist, embedding_2_dist


def clip_contrastive_loss(
    emb_1: torch.Tensor,
    emb_2: torch.Tensor,
    temperature_coef: TempCoef,
    local_loss: bool = False,
) -> torch.Tensor:
    """Computes contrastive CLIP-style loss.

    Args:
        emb_1 (torch.Tensor): Input embeddings.
        emb_2 (torch.Tensor): Embedding of positive pairs (perturbed inputs)
        temperature_coef (TempCoef): Module wrapping trainable temperature parameter.
        local_loss (bool, optional): If set, contrastive loss will only use data in current device. Defaults to False.

    Returns:
        torch.Tensor: Contrastive loss.
    """

    if local_loss:
        emb_1_dist, emb_2_dist = emb_1, emb_2
    else:
        # Gathers embeddings across devices.
        emb_1_dist, emb_2_dist = gather_embeddings(emb_1, emb_2)

    # Compute similarity matrix
    similarities = emb_1_dist @ emb_2_dist.T

    # Multiply similarity matrix by temperature
    similarities = temperature_coef(similarities)

    # Matching representations of positive pairs assumed to be located at the main
    # dioagonal of the similarity matrix if targets are not given
    ce_labels = torch.arange(similarities.size(0)).long().to(similarities.device)

    # We use a cross-entropy criterion to increase the similarities between
    # matching representations of source and target
    # We follow CLIP and apply the loss across columns and rows.
    sim_loss = 0.5 * (
        torch.nn.functional.cross_entropy(similarities, ce_labels)
        + torch.nn.functional.cross_entropy(similarities.T, ce_labels)
    )

    return sim_loss


def pool_and_normalize(
    features_sequence: torch.Tensor,
    attention_masks: torch.Tensor,
    return_norms: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Temporal ooling of sequences of vectors and projection onto the unit sphere.

    Args:
        features_sequence (torch.Tensor): Inpute features with shape [B, T, F].
        attention_masks (torch.Tensor): Pooling masks with shape [B, T, F].
        return_norms (bool, optional): Whether to additionally return the norms. Defaults to False.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: Pooled and normalized vectors with shape [B, F].
    """

    pooled_embeddings = pooling(features_sequence, attention_masks)
    embedding_norms = pooled_embeddings.norm(dim=1)

    normalizing_factor = torch.where(  # Only normalize embeddings with norm > 1.0.
        embedding_norms > 1.0, embedding_norms, torch.ones_like(embedding_norms)
    )

    pooled_normalized_embeddings = pooled_embeddings / normalizing_factor[:, None]

    if return_norms:
        return pooled_normalized_embeddings, embedding_norms
    else:
        return pooled_normalized_embeddings


def pooling(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pools a batch of vector sequences into a batch of vector global representations.
    It does so by taking the last vector in the sequence, as indicated by the mask.

    Args:
        x (torch.Tensor): Batch of vector sequences with shape [B, T, F].
        mask (torch.Tensor): Batch of masks with shape [B, T].

    Returns:
        torch.Tensor: Pooled version of the input batch with shape [B, F].
    """

    eos_idx = mask.sum(1) - 1
    batch_idx = torch.arange(len(eos_idx), device=x.device)

    mu = x[batch_idx, eos_idx, :]

    return mu


def retrieval_eval(
    x_source: torch.Tensor, x_target: torch.Tensor, return_similarities: Optional[bool] = False
) -> List[torch.Tensor]:
    """Performs retrieval evaluation given paired embeddings of source and target data.

    Args:
        x_source (torch.Tensor): Source batch of embeddings with shape [B, emb_dim].
        x_target (torch.Tensor): Target batch of embeddings with shape [B, emb_dim].
        return_similarities (Optional[bool]): Whether to return similarity matrix. Defaults to False.

    Returns:
        List[torch.Tensor]: Various retrieval metrics: R@1, R@5, and MRR.
    """

    # Compute similarity matrix
    similarities = x_source @ x_target.T

    topk_indices = torch.topk(similarities, k=similarities.size(1), dim=1)[1]

    ce_labels = torch.arange(similarities.size(0)).long().view(similarities.size(0), 1)

    # Bool tensor indicating which rows contain the idx corresponding to the main diag. of the sim matrix
    results = topk_indices.eq(ce_labels)

    r_at_1 = results[:, :1].sum() / float(similarities.size(0))
    r_at_5 = results[:, :5].sum() / float(similarities.size(0))

    ranks = results.nonzero()[:, 1].float() + 1.0
    mrr = (1 / ranks).mean()

    if return_similarities:
        return r_at_1, r_at_5, mrr, similarities
    else:
        return r_at_1, r_at_5, mrr