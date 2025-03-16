import os
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm

def download_weights(model_name, path):
    from huggingface_hub import snapshot_download
    import torch

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name

    folder = snapshot_download(hf_model_name, allow_patterns="*.bin")
    bin_files = glob.glob(os.path.join(folder, "*.bin"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file)
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())

            # shared embedding
            if "decoder.embed_tokens.weight" in name:
                shutil.copy(param_path, param_path.replace(
                    "decoder.embed_tokens.weight", "lm_head.weight"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--path", type=str, default="~/opt_weights")
    args = parser.parse_args()
    download_opt_weights(args.model, args.path)