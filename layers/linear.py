import torch 
from torch import nn 
import torch.nn.functional as F 
import torch.distributed as dist

def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

class Linearbase(nn.Module):
    def __init__(self, input_size: int, output_size:int bias:bool=False, tp_dim: int| None=None):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank=dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ReplicatedLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool=False, ):
        super().__init__(input_size, output_size, bias):
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
# Splits weight along OUTPUT dimension. Example: Llama 70B MLP 8192→28672
# With tp_size=2: each GPU holds 8192→14336 weight. No communication in forward.
class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool=False, ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)
    
    def weight_loader(self, param: nn.Parameter, loaded_weight:torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        '''
        # GPU 0 (tp_rank=0):
shard_size = 14336  # 28672 / 2
start_idx = 0 * 14336 = 0
loaded_weight = loaded_weight.narrow(0, 0, 14336)
# Result: [14336, 8192] - Takes ROWS 0:14336
        '''
        start_idx = self.tp_rank * shard_size 
        #Purpose: Extract a slice of a tensor along a specific dimension without copying memory.
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size: int, output_size: list[int], bias: bool=False,):
        self.output_sizes = output_size
        super().__init_(input_size, sum(output_size), bias)    
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size    
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank] 
        param_data.copy_(loaded_weight)   

        
# Merges Q+K+V projections into one weight. Handles Grouped Query Attention (GQA).
# Example: Llama 70B has 64 Q heads, but only 8 K/V heads (8x fewer for memory savings).
# With tp_size=2: each GPU holds Q(32 heads) + K(4 heads) + V(4 heads) = [5120, 8192]
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int, total_sum_kv_heads: int | None = None, bias: bool =False):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size=head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2*total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)    
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

class RowParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size:int, bias: bool=False):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data 
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y