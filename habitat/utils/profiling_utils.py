# todo: get rid of dependency on torch; we just need a small C library to call NVTX 
# todo: gracefully handle missing torch (revert to noop)
import torch

def range_push(msg):
    torch.cuda.nvtx.range_push(msg)

def range_pop():
    torch.cuda.nvtx.range_pop()