import torch

def range_push(msg):
    torch.cuda.nvtx.range_push(msg)

def range_pop():
    torch.cuda.nvtx.range_pop()