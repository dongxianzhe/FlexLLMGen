import torch

def test_simulated_compression():
    torch.manual_seed(0)
    a = torch.normal(0, 1, (64, 64, 64), dtype=torch.float16).cuda()

    config = CompressionConfig(
        num_bits=4, group_size=32, group_dim=0, symmetric=False)
    packed_data = compress(a, config)
    b = decompress(packed_data, config)
    print(a[0])
    print(b[0])


def test_real_compression():
    torch.manual_seed(0)
    a = torch.normal(0, 1, (32, 1, 1), dtype=torch.float16).cuda()

    config = CompressionConfig(
        num_bits=4, group_size=32, group_dim=0, symmetric=False)
    dev = TorchDevice("cuda:0", 0, 0).compressed_device
    packed = dev.compress(a, config)
    b = dev.decompress(packed)

    print(a.flatten())
    print(b.flatten())

if __name__ == "__main__":
    fix_recursive_import()
    #test_simulated_compression()
    test_real_compression()