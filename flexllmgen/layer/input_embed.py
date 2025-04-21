import os
import numpy as np
from flexllmgen.utensor import TorchTensor
from flexllmgen.layer import Layer
from flexllmgen.layer.weight_init_utils import init_weight_list

# class 
class InputEmbed(Layer):
    def __init__(self, 
                 vocab_size: int, 
                 hidden_size: int, 
                 max_seq_len: int, 
                 pad_token_id: int, 
                 dtype: type, 
                 env, 
                 policy,
                 ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.dtype = dtype

        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight else self.compute)

    def set_task(self, task):
        pass

    def init_weight(self, path: str) -> list[TorchTensor]:
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((self.vocab_size, self.hidden_size), self.dtype, path + "decoder.embed_tokens.weight"), # todo decoupled with model
            # w_pos
            ((self.max_seq_len + 2, self.hidden_size), self.dtype, path + "decoder.embed_positions.weight"),
        ]
        return init_weight_list(weight_specs, self.policy, self.env)

    def load_weight(self, weight_home, weight_read_buf):
        w_token, w_pos = weight_home.val
        dst = self.weight_load_dst
        weight_read_buf.store((w_token.smart_copy(dst), w_pos.smart_copy(dst)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass

    def load_cache(self, cache_home, cache_read_buf, i):
        pass

    def store_cache(self, cache_home, cache_write_buf, i):
        pass

    def input_act_shape_and_dtype(self, batch_size: int, seq_len: int):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        h = self.compute.opt_input_embed(h, mask, w_token, w_pos, self.pad_token_id, donate)
        hidden.val = h