import numpy as np
from dataclasses import dataclass
from flexllmgen.utensor import TorchTensor

@dataclass
class WeightLoaderConfig:
    weight_disk_percent: float


@dataclass
class WeightSpecs:
    pass


class WeightLoader:
    def __init__(self):
        pass

    def _get_choice(self, cur_percent, percents, choices):
        percents = np.cumsum(percents)
        assert np.abs(percents[-1] - 100) < 1e-5

        for i in range(len(percents)):
            if cur_percent < percents[i]:
                return choices[i]
        return choices[-1]

    def init_weight_list(self, weight_specs, policy, env) -> list[TorchTensor]:
        self.dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
        self.dev_choices = [env.disk, env.cpu, env.gpu]

        sizes = [np.prod(spec[0]) for spec in weight_specs]
        sizes_cumsum = np.cumsum(sizes)
        ret = []
        for i in range(len(weight_specs)):
            mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
            home = self._get_choice(mid_percent * 100, dev_percents, dev_choices)
            shape, dtype, filename = weight_specs[i]

            if len(shape) < 2:
                pin_memory = True
                compress = False
            else:
                pin_memory = policy.pin_weight
                compress = policy.compress_weight

            if not compress:
                weight = home.allocate(shape, dtype, pin_memory=pin_memory)
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight = home.compressed_device.allocate(shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)
                weight.load_from_np_file(weight_specs[i][2])

            ret.append(weight)
            for weight in ret:
                print(type(weight))
        return ret