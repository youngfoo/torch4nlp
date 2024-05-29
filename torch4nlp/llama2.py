import torch
from torch import nn


class Llama2:
    def __init__(self):
        # model parameters
        self.num_hidden_layers = 32  # number of decoder layers
        self.vocab_size = 32000
        self.hidden_size = 4096
        self.num_heads = 32 # num_attention_heads
        self.hidden_act = 'silu'
        self.pretraining_tp = 1

        # special tokens
        self.padding_idx = ？

        # layers
        for _ in range(self.num_hidden_layers):
            q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            
        self.layers = nn.ModuleList()
        
        ## embedding layer
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx)
        
        ## rms norm layer
        self.rms_norm_weight = nn.Parameter(torch.ones(self.hidden_size))
        self.rms_variance_epsilon = 1e-5  # rms_norm_eps
        
        ## mlp layer
        

    def rms_norm(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.rms_variance_epsilon)
        return self.rms_norm_weight * hidden_states.to(input_dtype)
    
    def mlp(self, hidden_states):
        
    def forward(self, input_ids, position_ids=None):
        
        batch_size, seq_length = input_ids.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0)  # [1, seq_length]
        
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        for idx, decoder_layer in enumerate(self.layers):
            # decoder layer
            residual = hidden_states
            hidden_states = self.rms_norm(hidden_states)
            # self attn
            
            
            hidden_states = residual + hidden_states
            
            # fully connected
            residual = hidden_states
            hidden_states = self.rms_norm_layer(hidden_states)
            hidden_states = self.
            layer_outputs = decoder_layer(
                hidden_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=output_attentions,
                use_cache=(),
            )
            hidden_states = layer_outputs[0]
            hidden_states = self.rms_norm_layer(hidden_states)


        