import torch
print('CUDA:', torch.cuda.is_available())
print('CPU cores:', torch.get_num_threads())