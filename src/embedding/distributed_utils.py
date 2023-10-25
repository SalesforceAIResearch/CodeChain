import torch
import torch.distributed as dist


class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    Adapted from https://github.com/Lightning-AI/lightning-bolts/blob/5577453a6d7072724d9ae24184daf8f45d4baff7/pl_bolts/models/self_supervised/simclr/simclr_module.py#L20-L40
    """

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(
            grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False
        )

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


all_gather = AllGather.apply
