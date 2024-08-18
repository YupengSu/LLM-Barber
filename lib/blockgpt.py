import torch
import torch.nn as nn
import copy

class LlamaAttentionGPT:
    def __init__(self, attention, attention_mask, position_ids, block_id=0, block_name="attention"):
        self.attention = copy.deepcopy(attention)
        for param in self.attention.parameters():
            param.data = param.data.float()
        self.attention.train()
        self.attention_mask = attention_mask
        self.position_ids = position_ids

        self.scalers = {}
        self.scalers['self_attn.q_proj'] = torch.zeros_like(
            self.attention.q_proj.weight, 
            device=self.attention.q_proj.weight.device,
            dtype=torch.float32
            )
        self.scalers['self_attn.k_proj'] = torch.zeros_like(
            self.attention.k_proj.weight,
            device=self.attention.k_proj.weight.device, 
            dtype=torch.float32
            )
        self.scalers['self_attn.v_proj'] = torch.zeros_like(
            self.attention.v_proj.weight, 
            device=self.attention.v_proj.weight.device, 
            dtype=torch.float32
            )
        self.scalers['self_attn.o_proj'] = torch.zeros_like(
            self.attention.o_proj.weight, 
            device=self.attention.o_proj.weight.device, 
            dtype=torch.float32
            )
        self.nsamples = 0

        self.input = []
        self.output = []
        
        self.pre_loss = 0
        self.post_loss = 0
    
    def get_input(self, inp):
        self.input.append(inp.clone().detach())

    def get_output(self, out):
        self.output.append(out.clone().detach())

    def add_batch(self):
        inp = self.input[self.nsamples].float().requires_grad_(True)
        out = self.output[self.nsamples].float().requires_grad_(True)

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        else:
            inp = inp
        tmp = inp.shape[0]
        
        out_ = self.attention(inp, attention_mask=self.attention_mask, position_ids=self.position_ids)[0]
        criterion = nn.MSELoss()
        loss = criterion(out, out_[0])
        self.pre_loss += loss.item()
        loss.backward(retain_graph=True)
        
        this_scalers = {}
        this_scalers['self_attn.q_proj'] = self.attention.q_proj.weight.grad ** 2
        this_scalers['self_attn.k_proj'] = self.attention.k_proj.weight.grad ** 2
        this_scalers['self_attn.v_proj'] = self.attention.v_proj.weight.grad ** 2
        this_scalers['self_attn.o_proj'] = self.attention.o_proj.weight.grad ** 2

        for param in self.attention.parameters():
            param.grad.zero_()
            
        self.output[self.nsamples] = out_[0].clone().detach()

        self.nsamples += tmp
        for key in self.scalers.keys():
            self.scalers[key] *= (self.nsamples - tmp) / self.nsamples
            self.scalers[key] += this_scalers[key] / self.nsamples

    def update_loss(self, out):
        criterion = nn.MSELoss()
        loss = criterion(out, self.output[self.nsamples])
        self.post_loss += loss.item() 
        self.nsamples += 1
    
    def free(self):
        self.attention = None
        self.scalers = None
        self.input = None
        self.output = None
        torch.cuda.empty_cache()

class LlamaMLPGPT:
    def __init__(self, feedforward, block_id=0, block_name="mlp"):
        self.feedforward = copy.deepcopy(feedforward)
        for param in self.feedforward.parameters():
            param.data = param.data.float()
        self.feedforward.train()

        self.scalers = {}
        self.scalers['mlp.gate_proj'] = torch.zeros_like(
            self.feedforward.gate_proj.weight, 
            device=self.feedforward.gate_proj.weight.device, 
            dtype=torch.float32
            )
        self.scalers['mlp.down_proj'] = torch.zeros_like(
            self.feedforward.down_proj.weight, 
            device=self.feedforward.down_proj.weight.device, 
            dtype=torch.float32
            )
        self.scalers['mlp.up_proj'] = torch.zeros_like(
            self.feedforward.up_proj.weight, 
            device=self.feedforward.up_proj.weight.device, 
            dtype=torch.float32)
        self.nsamples = 0

        self.input = []
        self.output = []
        
        self.pre_loss = 0
        self.post_loss = 0

    def get_input(self, inp):
        self.input.append(inp.clone().detach())

    def get_output(self, out):
        self.output.append(out.clone().detach())

    def add_batch(self):
        inp = self.input[self.nsamples].float().requires_grad_(True)
        out = self.output[self.nsamples].float().requires_grad_(True)

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        else:
            inp = inp
        tmp = inp.shape[0]
        
        out_ = self.feedforward(inp)
        criterion = nn.MSELoss()
        loss = criterion(out, out_[0])
        self.pre_loss += loss.item()     
        loss.backward(retain_graph=True)

        this_scalers = {}
        this_scalers['mlp.gate_proj'] = self.feedforward.gate_proj.weight.grad ** 2
        this_scalers['mlp.down_proj'] = self.feedforward.down_proj.weight.grad ** 2
        this_scalers['mlp.up_proj']   = self.feedforward.up_proj.weight.grad ** 2

        for param in self.feedforward.parameters():
            param.grad.zero_()
            
        self.output[self.nsamples] = out_[0].clone().detach()

        self.nsamples += tmp
        for key in self.scalers.keys():
            self.scalers[key] *= (self.nsamples - tmp) / self.nsamples
            self.scalers[key] += this_scalers[key] / self.nsamples

    def update_loss(self, out):
        criterion = nn.MSELoss()
        loss = criterion(out, self.output[self.nsamples])
        self.post_loss += loss.item()
        self.nsamples += 1

    def free(self):
        self.feedforward = None
        self.scalers = None
        self.input = None
        self.output = None
        torch.cuda.empty_cache()

class OPTAttentionGPT:
    def __init__(self, attention, attention_mask, block_id=0, block_name="attention"):
        self.attention = copy.deepcopy(attention)
        for param in self.attention.parameters():
            param.data = param.data.float()
        self.attention.train()
        self.attention_mask = attention_mask

        self.scalers = {}
        self.scalers['self_attn.q_proj'] = torch.zeros_like(
            self.attention.q_proj.weight, 
            device=self.attention.q_proj.weight.device,
            dtype=torch.float32
            )
        self.scalers['self_attn.k_proj'] = torch.zeros_like(
            self.attention.k_proj.weight,
            device=self.attention.k_proj.weight.device, 
            dtype=torch.float32
            )
        self.scalers['self_attn.v_proj'] = torch.zeros_like(
            self.attention.v_proj.weight, 
            device=self.attention.v_proj.weight.device, 
            dtype=torch.float32
            )
        self.scalers['self_attn.out_proj'] = torch.zeros_like(
            self.attention.out_proj.weight, 
            device=self.attention.out_proj.weight.device, 
            dtype=torch.float32
            )
        self.nsamples = 0

        self.input = []
        self.output = []
        
        self.pre_loss = 0
        self.post_loss = 0
    
    def get_input(self, inp):
        self.input.append(inp.clone().detach())

    def get_output(self, out):
        self.output.append(out.clone().detach())

    def add_batch(self):
        inp = self.input[self.nsamples].float().requires_grad_(True)
        out = self.output[self.nsamples].float().requires_grad_(True)

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        else:
            inp = inp
        tmp = inp.shape[0]
        
        out_ = self.attention(inp, attention_mask=self.attention_mask)[0]
        criterion = nn.MSELoss()
        loss = criterion(out, out_[0])
        self.pre_loss += loss.item()
        loss.backward(retain_graph=True)
        
        this_scalers = {}
        this_scalers['self_attn.q_proj'] = self.attention.q_proj.weight.grad ** 2
        this_scalers['self_attn.k_proj'] = self.attention.k_proj.weight.grad ** 2
        this_scalers['self_attn.v_proj'] = self.attention.v_proj.weight.grad ** 2
        this_scalers['self_attn.out_proj'] = self.attention.out_proj.weight.grad ** 2

        for param in self.attention.parameters():
            param.grad.zero_()
            
        self.output[self.nsamples] = out_[0].clone().detach()

        self.nsamples += tmp
        for key in self.scalers.keys():
            self.scalers[key] *= (self.nsamples - tmp) / self.nsamples
            self.scalers[key] += this_scalers[key] / self.nsamples

    def update_loss(self, out):
        criterion = nn.MSELoss()
        loss = criterion(out, self.output[self.nsamples])
        self.post_loss += loss.item() 
        self.nsamples += 1
    
    def free(self):
        self.attention = None
        self.scalers = None
        self.input = None
        self.output = None
        torch.cuda.empty_cache()

class OPTMLPGPT:
    def __init__(self, fc1, activation_fn, fc2, block_id=0, block_name="mlp"):
        self.fc1 = copy.deepcopy(fc1)
        self.activation_fn = copy.deepcopy(activation_fn)
        self.fc2 = copy.deepcopy(fc2)
        for param in self.fc1.parameters():
            param.data = param.data.float()
        self.fc1.train()
        for param in self.fc2.parameters():
            param.data = param.data.float()
        self.fc2.train()

        self.scalers = {}
        self.scalers['fc1'] = torch.zeros_like(
            self.fc1.weight, 
            device=self.fc1.weight.device, 
            dtype=torch.float32
            )
        self.scalers['fc2'] = torch.zeros_like(
            self.fc2.weight, 
            device=self.fc2.weight.device, 
            dtype=torch.float32
            )
        self.nsamples = 0

        self.input = []
        self.output = []
        
        self.pre_loss = 0
        self.post_loss = 0

    def get_input(self, inp):
        self.input.append(inp.clone().detach())

    def get_output(self, out):
        self.output.append(out.clone().detach())

    def add_batch(self):
        inp = self.input[self.nsamples].float().requires_grad_(True)
        out = self.output[self.nsamples].float().requires_grad_(True)

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        else:
            inp = inp
        tmp = inp.shape[0]
        
        out_ = self.fc1(inp)
        out_ = self.activation_fn(out_)
        out_ = self.fc2(out_)
        criterion = nn.MSELoss()
        loss = criterion(out, out_[0])
        self.pre_loss += loss.item()
        loss.backward(retain_graph=True)

        this_scalers = {}
        this_scalers['fc1'] = self.fc1.weight.grad ** 2
        this_scalers['fc2'] = self.fc2.weight.grad ** 2

        for param in self.fc1.parameters():
            param.grad.zero_()
        for param in self.fc2.parameters():
            param.grad.zero_()
            
        self.output[self.nsamples] = out_[0].clone().detach()

        self.nsamples += tmp
        for key in self.scalers.keys():
            self.scalers[key] *= (self.nsamples - tmp) / self.nsamples
            self.scalers[key] += this_scalers[key] / self.nsamples

    def update_loss(self, out):
        criterion = nn.MSELoss()
        loss = criterion(out, self.output[self.nsamples])
        self.post_loss += loss.item()
        self.nsamples += 1
    
    def free(self):
        self.fc1 = None
        self.activation_fn = None
        self.fc2 = None
        self.scalers = None
        self.input = None
        self.output = None
        torch.cuda.empty_cache()