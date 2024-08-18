import torch
import torch.nn as nn

## Weight and Derivative: Wanda is a method of using first-order derivatives.
class UWrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler = torch.zeros_like(self.layer.weight.data, device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp
        
        inp = inp.type(torch.float32)
        ones = torch.ones(self.rows, inp.shape[1], device=self.dev)
        this_scaler = ones.matmul((inp.t() ** 2))
        self.scaler += this_scaler / self.nsamples
    