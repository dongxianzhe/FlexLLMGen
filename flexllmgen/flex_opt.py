"""
Usage:
python3 -m flexllmgen.flex_opt --model facebook/opt-1.3b --percent 0 100 100 0 100 0
"""

import argparse
from transformers import AutoTokenizer
from flexllmgen.utensor import CompressionConfig, ExecutionEnv, TorchDevice, TorchDisk, TorchMixedDevice
from flexllmgen.utils import timers
from flexllmgen.utils import GB, str2bool, write_benchmark_log
from flexllmgen.model import OptModelFactory
from flexllmgen.engine import Policy, Engine, Task

def get_filename(args):
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"ngbs{args.num_gpu_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename


def get_test_inputs(prompt_len, num_prompts, tokenizer):
    prompts = ["Paris is the capital city of"]
    input_ids = tokenizer(prompts, padding="max_length", max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts


def run_flexllmgen(args):
    print(f"<run_flexllmgen>: args.model: {args.model}")
    if args.model == "facebook/galactica-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len = args.prompt_len, args.gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    factory = OptModelFactory(args.model)
    opt_config = factory.get_config()
    cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {opt_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    print("init weight...")
    model = Engine(args.model, env, args.path, policy)

    try:
        print("warmup - generate")
        task = Task(
            inputs=warmup_inputs,
            prompt_len=len(warmup_inputs[0]),
            gen_len=2,
            do_sample=False,
            temperature=1.0,
            stop=None,
        )
        output_ids = model.generate(task)

        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=args.gen_len,
            do_sample=False,
            temperature=1.0,
            stop=None,
        )
        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(task)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    show_str = "Outputs:\n" + 70 * '-' + "\n"
    for i in range(len(outputs)):
        show_str += f"{i}: {outputs[i]}\n"
        show_str += "-" * 70 + "\n"
    if args.verbose >= 2:
        print(show_str)

    gpu.print_stats()
    cpu.print_stats()

    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file

    log_str = write_benchmark_log(filename,
        opt_config.model_bytes(), cache_size, hidden_size,
        gpu_peak_mem, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b", help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights", help="The path to the model weights. If there are no cached weights, FlexLLMGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexllmgen_offload_dir", help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int, default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true", help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true", help="Whether to compress cache.")
    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--overlap", type=str2bool, nargs='?', const=True, default=True)
    args = parser.parse_args()

    assert len(args.percent) == 6

    run_flexllmgen(args)
