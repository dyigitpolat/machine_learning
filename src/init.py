import torch

def init():
    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda")
    print("CUDA device name: {}".format(torch.cuda.get_device_name(device)))
