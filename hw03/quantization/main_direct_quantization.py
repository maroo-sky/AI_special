import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from model.resnet import ResNet18
import random
import numpy as np
import argparse
import os
from tqdm import tqdm
import pdb

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def direct_quantization(model):
    model_fp32 = torch.quantization.fuse_modules(model,
                                                 [['conv1', 'bn1', 'layer1',
                                                   'layer2', 'layer3', 'layer4', 'linear']])
    model_fp32_prepared = torch.quantization.prepare(model_fp32)
    qmodel = torch.quantization.convert(model_fp32_prepared)
    return qmodel

def test(test_loader, model):
    model.eval()
    acc = 0.0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader, desc="Iteration"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = correct / total
        print(acc)
    return acc

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="/home/s1_u1/projects/quantization/dataset/CIFAR10",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--out_dir",
        default="/home/s1_u1/projects/quantization/checkpoints/resnet18.pt",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--test_batch_size",
        default=1000,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--n_gpu",
        default=1,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int,
        required=True,
    )
    args = parser.parse_args()
    set_seed(args)

    with torch.cuda.device(args.gpu_id):
        test_batch_size = args.test_batch_size

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_set = torchvision.datasets.CIFAR10(
            root=args.dataset, train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=test_batch_size, shuffle=False, num_workers=2)

        model = ResNet18()
        model.to(device)
        model.load_state_dict(torch.load(args.out_dir))

        qmodel = direct_quantization(model)
        _ = test(test_loader, qmodel)

if __name__ == "__main__":
    main()

