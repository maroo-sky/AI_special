import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import csv
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
from model.grid_args import grid_args

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


def train_classifiar(train_loader, test_loader, num_epoch, model, args, grid_args):
    run_file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_id_acc".format(
        grid_args.optimizer,
        grid_args.num_epoch,
        grid_args.lr_rate,
        args.seed,
        grid_args.weight_decay,
        grid_args.train_batch_size)

    run_file = os.path.join(args.run_file, run_file)
    if args.write:
        tb_writer = SummaryWriter(run_file)

    optimizer = None
    global_step = 0
    learning_rate = grid_args.lr_rate
    if grid_args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=0.9,
                                    weight_decay=grid_args.weight_decay)
    elif grid_args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=grid_args.weight_decay)

    t_max = len(train_loader) // num_epoch + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    criteria = nn.CrossEntropyLoss()

    tr_ce_loss, logging_ce_loss = 0.0, 0.0

    for epoch in range(num_epoch):
        iteration = tqdm(train_loader, desc="Iteration")
        for i, (images, labels) in enumerate(iteration):
            images = images.to(device)
            labels = labels.to(device)
            model.train()
            outputs = model(images)

            ce_loss = criteria(outputs, labels)

            ce_loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            tr_ce_loss += ce_loss.item()

            if args.write and global_step % len(iteration) == 0:
                training_acc = test(train_loader, model)
                test_acc = test(test_loader, model)
                #tb_writer.add_scalar("training_acc", training_acc, global_step)
                #tb_writer.add_scalar("test_acc", test_acc, global_step)
                #tb_writer.add_scalar("ce_loss", (tr_ce_loss - logging_ce_loss) / args.logging_steps,
                #                     global_step)

                #logging_ce_loss = tr_ce_loss

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

def write_results(filename, opt, num_epoch, lr, seed, weight_decay, num_batch_size, acc):

    file_exists = os.path.isfile(filename)

    info = {}
    fieldnames = []
    info["opt"], info["epoch"], info["lr"], \
    info["seed"], info["weight_decay"], info["batch_size"], info["ACC"] = \
        opt, num_epoch, lr, seed, weight_decay, num_batch_size, acc

    for key in info:
        fieldnames.append(str(key))

    with open(filename, 'a+', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(info)

    return

def grid(args):
    set_seed(args)

    for lr_rate in args.lr_rate:
        for epoch in args.num_epoch:
            for batch_size in args.train_batch_size:
                for wd in args.weight_decay:
                    for opt in args.optimizer:
                        set_seed(args)
                        g_args = grid_args(optimizer=opt,
                                          num_epoch=epoch,
                                          lr_rate=lr_rate,
                                          weight_decay=wd,
                                          train_batch_size=batch_size)

                        train_batch_size = batch_size
                        test_batch_size = args.test_batch_size
                        num_epochs = epoch

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
                            train_classifiar(train_loader, test_loader, num_epochs, model, args, g_args)
                            torch.save(model.state_dict(), args.out_dir)
                            acc = test(test_loader, model)
                            write_results(
                                args.eval_results, opt, epoch, lr_rate, args.seed, wd, batch_size, acc
                            )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_results",
        default="/home/s1_u1/projects/quantization/checkpoints/resnet_cifar10_result.csv",
        type=str,
        required=False,
    )
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
        default="/home/s1_u1/runs/CIFAR10/resnet18_grid",
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
        default=[256],
        type=int,
        required=False,
        nargs="*",
    )
    parser.add_argument(
        "--test_batch_size",
        default=1000,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--lr_rate",
        default=[0.001],
        type=float,
        required=False,
        nargs="*",
    )
    parser.add_argument(
        "--num_epoch",
        default=[50],
        type=int,
        required=False,
        nargs="*",
    )
    parser.add_argument(
        "--n_gpu",
        default=1,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--logging_steps",
        default=500,
        type=int,
        required=False,
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
        "--weight_decay",
        default=[0.0],
        type=float,
        required=False,
        nargs="*",
    )
    parser.add_argument(
        "--optimizer",
        default=['sgd', 'adam'],
        type=str,
        help="Choose optimizer",
        required=False,
        nargs="*",
    )
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int,
        required=True,
    )
    args = parser.parse_args()

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    with torch.cuda.device(args.gpu_id):
        grid(args)

if __name__=="__main__":

    main()


















