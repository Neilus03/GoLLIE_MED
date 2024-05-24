import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("CUDA Device Name:", torch.cuda.get_device_name(0))
print("CUDA Current Device:", torch.cuda.current_device())
print("CUDA Device Capability:", torch.cuda.get_device_capability(0))
