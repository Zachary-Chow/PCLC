import os
import numpy as np
import torch
import torchvision
import argparse
from modules import network, contrastive_loss, pclc, transform
from utils import yaml_config_hook, save_model
from torch.utils import data

#CUDA_VISIBVLE_DEVICES=0 python train.py



def train():
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        z_i, z_j, c_i, c_j, z_i_1, z_j_1, z_i_2, z_j_2, z_i_3, z_j_3, c_i_1, c_j_1, c_i_2, c_j_2, c_i_3, c_j_3 = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss_instance_1 = criterion_instance(z_i_1, z_j_1)
        loss_cluster_1 = criterion_cluster(c_i_1, c_j_1)
        loss_instance_2 = criterion_instance(z_i_2, z_j_2)
        loss_cluster_2 = criterion_cluster(c_i_2, c_j_2)
        loss_instance_3 = criterion_instance(z_i_3, z_j_3)
        loss_cluster_3 = criterion_cluster(c_i_3, c_j_3)
        loss = 0.25 * (loss_instance + loss_cluster +  (loss_instance_1 + loss_cluster_1) +  (loss_instance_2 + loss_cluster_2) + (loss_instance_3 + loss_cluster_3))
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    if args.dataset == "ImageNet-dogs":
        dataset = torchvision.datasets.ImageFolder(
            root='./dataset',
            transform=transform.Augmentation(size=args.image_size),
        )
        class_num = 15
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets',
            transform=transform.Augmentation(size=args.image_size),
        )
        class_num = 10
    elif args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root="./datasets",
            download=True,
            train=True,
            transform=transform.Augmentation(size=args.image_size),
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./datasets",
            download=True,
            train=False,
            transform=transform.Augmentation(size=args.image_size),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root="./datasets",
            download=True,
            train=True,
            transform=transform.Augmentation(size=args.image_size),
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root="./datasets",
            download=True,
            train=False,
            transform=transform.Augmentation(size=args.image_size),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    else:
        raise NotImplementedError
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
   
    pclc = pclc.pclc()
    model = network.Network_pclc(pclc, args.feature_dim, class_num)
    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.reload:
        print("reload training.")
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    
    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        if epoch % 50 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(args, model, optimizer, args.epochs)
