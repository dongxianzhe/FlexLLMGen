from enum import Enum, auto

class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()

    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        else:
            raise ValueError(f"Invalid name: {name}")

class Device:
    def allocate(self, shape, dtype, pin_memory=None, name=None):
        raise NotImplementedError

    def delete(self, tensor):
        raise NotImplementedError

    def init_cache_one_gpu_batch(self, config, task, policy):
        raise NotImplementedError

# class TorchDevice:
#     def add_link(self, link):
#     def allocate(self, shape, dtype, pin_memory=None, name=None):
#     def delete(self, tensor):
#     def init_attention_compute_workspace(self, config, task, policy):
#     def next_attention_compute_workspace(self):
#     def del_attention_compute_workspace(self):
#     def gen_attention_mask(self, token_ids, pad_token_id, donate):
#     def extend_attention_mask(self, attention_mask, donate):
#     def opt_input_embed(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
#     def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate, do_sample, temperature):
#     def init_cache_one_gpu_batch(self, config, task, policy):
#     def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config):
#     def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate, attn_sparsity, compress_cache, comp_config):
#     def _attention_weights(self, q, k, mask, b, src_s, n_head):
#     def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
#     def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b, src_s, tgt_s, n_head, head_dim, attn_sparsity):
#     def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new, mask, b, src_s, tgt_s, n_head, head_dim):
#     def mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
#     def synchronize(self):
#     def mem_stats(self):
#     def print_stats(self, output_file=None):

# class TorchDisk:
#     def add_link(self, link):
#     def allocate(self, shape, dtype, pin_memory=None, name=None):
#     def delete(self, tensor):
#     def init_cache_one_gpu_batch(self, config, task, policy):
#     def submit_copy(self, *args):
#     def synchronize(self):
#     def close_copy_threads(self):
#     def mem_stats(self):
#     def print_stats(self):

# class TorchMixedDevice:
#     def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
#     def delete(self, tensor):
#     def init_cache_one_gpu_batch(self, config, task, policy):

# class TorchCompressedDevice:
#     def allocate(self, shape, dtype, comp_config, pin_memory=None, name=None):
#     def init_cache_one_gpu_batch(self, config, task, policy):
#     def init_attention_compute_workspace(self, config, task, policy):
#     def compress(self, tensor, comp_config):
#     def decompress(self, tensor):