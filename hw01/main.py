import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from model.FC import FullyConnectedModel
from config.config_FC import FC_config
import random
import numpy as np
import argparse
import pdb

from estimators.intrinsic_dimensionality import ID
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and optimizer

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# Train the model
def train(train_loader, num_epochs, model, optimizer, criterion, args):
    id = ID()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            if args.do_id:
                id.ID_estimator(outputs, epoch)
                id.Global_ID()

            loss = criterion(outputs[0], labels)

            # Backprpagation and optimization
            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    if args.do_id:
        pdb.set_trace()
        intrinsic_dimensionality = id.Extract_global_ID()
        return intrinsic_dimensionality

def test(test_loader, model):

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="/home/s1_u1/hw01/dataset/",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--out_dir",
        default="/home/s1_u1/hw01/checkpoint/fc.pt",
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
        "--batch_size",
        default=64,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--lr_rate",
        default=0.001,
        type=float,
        required=False,
    )
    parser.add_argument(
        "--num_epoch",
        default=3,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--n_gpu",
        default=1,
        type=int,
        required=False,
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_id", action="store_true", help="Whether to estimate id")
    args = parser.parse_args()

    set_seed(args)

    batch_size = args.batch_size
    learning_rate = args.lr_rate
    num_epochs = args.num_epoch
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root=args.dataset,
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=False)

    test_dataset = torchvision.datasets.MNIST(root=args.dataset,
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    config = FC_config()
    model = FullyConnectedModel(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    id = []
    if args.do_train:
        if args.do_id:
            id.append(train(train_loader, num_epochs, model, optimizer, criterion, args))
        else:
            train(train_loader, num_epochs, model, optimizer, criterion)
        test(test_loader, model)
    else:
        model = FullyConnectedModel(config)
        model.load_state_dict(torch.load(args.out_dir))
        model.to(device)
        model.eval()
        test(test_loader, model)

    torch.save(model.state_dict(), args.out_dir)



if __name__ == "__main__":
    main()