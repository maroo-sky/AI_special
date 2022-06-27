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

def train(train_loader, test_loader, num_epoch, model, args):
    run_file = "epoch:{}_lr:{}_lambda:{}_wd:{}_seed:{}".format(args.num_epoch,
                                                               args.lr_rate,
                                                               args.reg_lambda,
                                                               args.weight_decay,
                                                               args.seed)
    run_file = os.path.join(args.run_file, run_file)
    if args.write:
        tb_writer = SummaryWriter(run_file)

    global_step = 0
    optimizer = None

    learning_rate = args.lr_rate
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=args.weight_decay)

    t_max = len(train_loader) // num_epoch + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    tr_loss, logging_loss = 0.0, 0.0

    for epoch in range(num_epoch):
        iteration = tqdm(train_loader, desc="Iteration")
        for i, (images, labels) in enumerate(iteration):
            images, labels = images.to(device), labels.to(device)

            model.train()
            outputs = model(images)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            global_step += 1
            tr_loss += loss.item()

            if args.write and global_step % len(iteration) == 0:
                training_acc = test(train_loader, model)
                test_acc = test(test_loader, model)
                tb_writer.add_scalar("training__acc", training_acc, global_step)
                tb_writer.add_scalar("test_acc", test_acc, global_step)

                logging_loss = tr_loss

    if args.write:
        tb_writer.close()



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
        "--run_file",
        default="/home/s1_u1/runs/CIFAR10",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Whether write tensorboard or not",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--train_batch_size",
        default=64,
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
        "--lr_rate",
        default=0.1,
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
    parser.add_argument(
        "--logging_steps",
        default=100,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--save_rep",
        action="store_true",
        help="Whether save representation"
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training.")
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--l1_reg",
        action="store_true",
        help="Do L1 regularization",
    )
    parser.add_argument(
        "--l2_reg",
        action="store_true",
        help="Do L2 regularization",
    )
    parser.add_argument(
        "--reg_lambda",
        default=1e-4,
        type=float,
        help="Hyper-parameter for regularizer",
    )
    parser.add_argument(
        "--momentum",
        default=0.0,
        type=float,
        required=False,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        required=False,
    )
    parser.add_argument(
        "--optimizer",
        choices=['sgd', 'adam'],
        default='sgd',
        type=str,
        help="Choose optimizer",
        required=False,
    )
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int,
        required=True,
    )
    args = parser.parse_args()
    set_seed(args)

    with torch.cuda.device(args.gpu_id):
        train_batch_size = args.train_batch_size
        test_batch_size = args.test_batch_size
        num_epochs = args.num_epoch

        #CIFAR-10 dataset
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root=args.dataset, train=True, download=False, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=train_batch_size, shuffle=True, num_workers=2)

        test_set = torchvision.datasets.CIFAR10(
            root=args.dataset, train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=test_batch_size, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

        model = ResNet18()
        model.to(device)

        if args.do_train:
            train(train_loader, test_loader, num_epochs, model, args)
            torch.save(model.state_dict(), args.out_dir)
            _ = test(test_loader, model)
        else:
            model.load_state_dict(torch.load(args.out_dir))
            model.to(device)
            model.eval()
            _ = test(test_loader, model)

if __name__ == "__main__":
    main()
        
        






























