class Layer:
    def set_task(self, task):
        raise NotImplementedError
    def init_weight(self, weight_home, path):
        raise NotImplementedError
    def load_weight(self, weight_home, weight_read_buf):
        raise NotImplementedError
    def init_cache_one_gpu_batch(self, cache_home):
        raise NotImplementedError
    def load_cache(self, cache_home, cache_read_buf, i):
        raise NotImplementedError
    def store_cache(self, cache_home, cache_write_buf, i):
        raise NotImplementedError
    def input_act_shape_and_dtype(self, batch_size, seq_len):
        raise NotImplementedError
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask, cache_write_buf, i, k):
        raise NotImplementedError

# class OptLM:
#     def __init__(self,
#                  config: Union[str, OptConfig],
#                  env: ExecutionEnv,
#                  path: str,
#                  policy: Policy):

#     def set_task(self, task):
#     def init_weight(self, j):
#     def load_weight(self, i, j, k, overlap=True):
#     def delete_weight(self, j, k):
#     def init_cache(self, j, k):
#     def load_cache(self, i, j, k, overlap=True):
#     def store_cache(self, i, j, k, overlap=True):
#     def delete_cache(self, j, k):
#     def load_hidden(self, i, j, k):
#     def store_hidden(self, i, j, k):
#     def compute_layer(self, i, j, k):
#     def sync(self):
#     def init_all_weights(self):
#     def delete_all_weights(self):
#     def update_attention_mask(self, i, k):
#     def generate(self,
#                  inputs: Union[np.array, List[List[int]]],
#                  max_new_tokens: int = 32,
#                  do_sample: bool = False,
#                  temperature: float = 1.0,
#                  stop: Optional[int] = None,
#                  debug_mode: Optional[str] = None,
#                  cut_gen_len: Optional[int] = None,
#                  verbose: int = 0):
#     def generation_loop_normal(self):
#     def generation_loop_debug_normal(self):
#     def generation_loop_overlap_single_batch(self):
#     def generation_loop_overlap_multi_batch(self):
#     def generation_loop_debug_single_batch(self):
#     def generation_loop_debug_multi_batch(self):