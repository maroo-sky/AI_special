import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from model.FC import FullyConnectedModel
from config.config_FC import FC_config
import random
import numpy as np
import argparse
import time

class fc_model():
    def __init__(self, model_dict):
        self.embed_weight = model_dict['embedlayer.embed.weight']
        self.embed_bias = model_dict['embedlayer.embed.bias']
        self.layer1_weight = model_dict['layer.0.dense.weight']
        self.layer1_bias = model_dict['layer.0.dense.bias']
        self.layer2_weight = model_dict['layer.1.dense.weight']
        self.layer2_bias = model_dict['layer.1.dense.bias']
        self.out_weight = model_dict['out.weight']
        self.out_bias = model_dict['out.bias']

    def relu(self, input):
        return np.maximum(0, input)

    def forward(self, input):
        output = np.matmul(input, self.embed_weight.T)
        output += self.embed_bias
        output = self.relu(output)

        output = np.matmul(output, self.layer1_weight.T)
        output += self.layer1_bias
        output = self.relu(output)

        output = np.matmul(output, self.layer2_weight.T)
        output += self.layer2_bias
        output = self.relu(output)
        #print(output)

        #output = np.matmul(output, self.out_weight.T)
        #output += self.out_bias

        return output

def convert_model_to_numpy(model):
    model_dict = {}
    for name, param in model.named_parameters():
        model_dict[name] = param.data.cpu().detach().numpy()

    return model_dict


def test(test_loader, model):

    with torch.no_grad():
        correct = 0
        total = 0
        tic = time.perf_counter()
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device).detach().numpy()
            labels = labels.to(device).detach().numpy()
            outputs = model.forward(images)
            print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        toc = time.perf_counter()
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
        print('Processing Time is: {} sec'.format(toc-tic))

# test dataset

batch_size = 64

test_dataset = torchvision.datasets.MNIST(root='./dataset/',
                                              train=False,
                                              transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = FC_config()

checkpoint = "./checkpoint/fc.pt"
model = FullyConnectedModel(config)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.to(device)


model_dict = convert_model_to_numpy(model)
model = fc_model(model_dict)

correct = 0
total = 0


for _ in range(10):
    tic = time.perf_counter()
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device).detach().numpy()
        labels = labels.to(device).detach().numpy()
        outputs = model.forward(images)
        predicted = np.argmax(outputs, axis=1)
        total += labels.shape[0]
        correct += np.sum((predicted == labels))
    toc = time.perf_counter()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    #print('Processing Time is: {} sec'.format(toc-tic))
    print('{}'.format(toc - tic))