import torch
import torch.nn as nn



class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Linear(config.input_size, config.hidden_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.embed(input)
        output = self.relu(output)
        return output

class FullyConnectedLayer(nn.Module):
    def __init__(self, config ):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.dense(input)
        output = self.relu(output)
        return output

class FullyConnectedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_states = config.hidden_states
        self.embedlayer = EmbeddingLayer(config)
        self.layer = nn.ModuleList([FullyConnectedLayer(config) for _ in range(config.num_hidden_layers)])
        self.out = nn.Linear(config.hidden_size, config.num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        all_hidden_states = ()
        output = self.embedlayer(input)

        if self.hidden_states:
            all_hidden_states = all_hidden_states + (output,)
        for i, layer_module in enumerate(self.layer):
            output = layer_module(output)
            if self.hidden_states:
                all_hidden_states = all_hidden_states + (output,)
        output = self.out(output)

        #add last hidden layer representation
        if self.hidden_states:
            all_hidden_states = all_hidden_states + (output,)
            # output : last hidden layer, (intermediate layer)

        output = self.softmax(output)
        if self.hidden_states:
            output= (output) + all_hidden_states
        else: output = (output)
        return output # output, (all_hidden_states)