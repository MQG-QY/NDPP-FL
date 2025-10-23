import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
# from opacus import PrivacyEngine
from options import parse_args
from data import *
from net import *
from tqdm import tqdm
from utils import compute_noise_multiplier, mi
from tqdm.auto import trange
import copy
import sys
import random
import json
from Computing_Mutual_Information import mutual_information

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
num_clients = args.num_clients
local_epoch = args.local_epoch
global_epoch = args.global_epoch
batch_size = args.batch_size
target_epsilon = args.target_epsilon
target_delta = args.target_delta
clipping_bound = args.clipping_bound
dataset = args.dataset
user_sample_rate = args.user_sample_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.store == True:
    saved_stdout = sys.stdout
    file = open(
        f'./txt/{args.dirStr}/'
        f'dataset {dataset} '
        f'--num_clients {num_clients} '
        f'--user_sample_rate {args.user_sample_rate} '
        f'--local_epoch {local_epoch} '
        f'--global_epoch {global_epoch} '
        f'--batch_size {batch_size} '
        f'--target_epsilon {target_epsilon} '
        f'--target_delta {target_delta} '
        f'--clipping_bound {clipping_bound} '
        f'--fisher_threshold {args.fisher_threshold} '
        f'--lambda_1 {args.lambda_1} '
        f'--lambda_2 {args.lambda_2} '
        f'--lr {args.lr} '
        f'--alpha {args.dir_alpha}'
        f'.txt'
        , 'a'
    )
    sys.stdout = file


def local_update(model, dataloader, noise_multiplier):
    model.train()
    model = model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(local_epoch):
        # noise_stddev = (torch.sqrt(torch.tensor((clipping_bound ** 2) * (noise_multiplier ** 2) * math.sqrt(2) /num_clients))) * (math.exp(-0.5 * epoch))
        noise_stddev = (torch.sqrt(
            torch.tensor((clipping_bound ** 2) * (noise_multiplier ** 2) * math.sqrt(2) / num_clients))) / (
                                   1 + epoch * 0.5)

        print(noise_stddev)
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # 梯度裁剪
            norm = torch.sqrt(sum([torch.sum(param.data ** 2) for param in model.parameters()]))
            average_track_list = []
            average_track_dict = {}
            item = 0
            for param in model.parameters():
                item += 1
                param_data = (clipping_bound * param.data) / (norm + (0.1 / norm))
                if param_data.dim() == 4:
                    hessian = torch.matmul(param_data, param_data.permute(0, 1, 3, 2))
                if param_data.dim() == 3:
                    hessian = torch.matmul(param_data, param_data.permute(0, 2, 1))
                if param_data.dim() == 2:
                    hessian = torch.matmul(param_data, param_data.permute(1, 0))
                hessian_2d = hessian.view(-1, hessian.size(-1))
                diag_hessian = torch.diag(hessian_2d)  # 将hessian 矩阵转换为对角阵
                average_track = abs(torch.sum(diag_hessian)) / (diag_hessian.size()[0])
                if average_track.is_cuda:
                    average_track = average_track.cpu()
                average_track_list.append(average_track.detach().numpy())
                average_track_dict["parm" + str(item)] = average_track.detach().numpy()
            item2 = 0
            with torch.no_grad():
                for param2 in model.parameters():
                    item2 += 1
                    param2_data = (clipping_bound * param2.data) / (norm + (0.1 / norm))
                    proportion = average_track_dict["parm" + str(item2)] / max(average_track_list)
                    noise = torch.randn_like(param2_data) * noise_stddev * proportion
                    param2.data.add_(noise)

    model = model.to('cpu')
    return model


def test(client_model, client_testloader):
    client_model.eval()
    client_model = client_model.to(device)

    num_data = 0

    correct = 0
    with torch.no_grad():
        for data, labels in client_testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = client_model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            num_data += labels.size(0)

    accuracy = 100.0 * correct / num_data

    client_model = client_model.to('cpu')

    return accuracy


log_file = "fedsam_training_log_cifar100_e_600.jsonl"  # 每轮结果保存到该文件
# 如果之前有旧文件，选择是否清空
if os.path.exists(log_file):
    print(f"Log file '{log_file}' already exists, appending new records.")
else:
    with open(log_file, "w") as f:
        f.write("")  # 创建空文件


def main():
    mean_acc_s = []
    acc_matrix = []
    if dataset == 'MNIST':
        train_dataset, test_dataset = get_mnist_datasets()
        clients_train_set = get_clients_datasets(train_dataset, num_clients)
        client_data_sizes = [len(client_dataset) for client_dataset in clients_train_set]
        clients_train_loaders = [DataLoader(client_dataset, batch_size=batch_size) for client_dataset in
                                 clients_train_set]
        clients_test_loaders = [DataLoader(test_dataset) for i in range(num_clients)]
        clients_models = [mnistNet() for _ in range(num_clients)]
        global_model = mnistNet()
    elif dataset == 'FashionMNIST':
        train_dataset, test_dataset = get_fashion_mnist_datasets()
        clients_train_set = get_clients_datasets(train_dataset, num_clients)
        client_data_sizes = [len(client_dataset) for client_dataset in clients_train_set]
        clients_train_loaders = [DataLoader(client_dataset, batch_size=batch_size) for client_dataset in
                                 clients_train_set]
        clients_test_loaders = [DataLoader(test_dataset) for i in range(num_clients)]
        clients_models = [fashion_mnistNet() for _ in range(num_clients)]
        global_model = fashion_mnistNet()
    elif dataset == 'CIFAR10':
        # no-iid
        # clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR10(args.dir_alpha, num_clients)
        # iid
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_iid_cifar10(num_clients)
        clients_models = [cifar10Net() for _ in range(num_clients)]
        global_model = cifar10Net()

    elif dataset == 'CIFAR100':
        # no-iid
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR100(args.dir_alpha, num_clients)
        clients_models = [ResNet18_CIFAR100() for _ in range(num_clients)]
        global_model = ResNet18_CIFAR100()
    elif dataset == 'EMNIST':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_EMNIST(num_clients)
        clients_models = [EMNISTNet() for _ in range(num_clients)]
        global_model = EMNISTNet()
    elif dataset == 'FEMNIST':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_FEMNIST(num_clients)
        clients_models = [femnistNet() for _ in range(num_clients)]
        global_model = femnistNet()
    elif dataset == 'SVHN':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_SVHN(args.dir_alpha, num_clients)
        clients_models = [SVHNNet() for _ in range(num_clients)]
        global_model = SVHNNet()
    else:
        print('undifined dataset')
        assert 1 == 0
    #  全局模型
    for client_model in clients_models:
        client_model.load_state_dict(global_model.state_dict())

    mi_data = mi()
    e = global_epoch * (args.target_epsilon)
    e_c = 0
    for epoch in trange(global_epoch):
        chunk_mi = mi_data[
                   epoch * int(num_clients * user_sample_rate):(epoch + 1) * int(num_clients * user_sample_rate)]
        float_list = [float(i) for i in chunk_mi]
        avg = (sum(float_list)) / len(float_list)
        if epoch == 0:
            target_epsilon = e / global_epoch
        else:
            target_epsilon = math.exp(-avg) * ((e - e_c) / (global_epoch - epoch))
        e_c += target_epsilon
        noise_multiplier = compute_noise_multiplier(target_epsilon, target_delta, global_epoch, local_epoch,
                                                    batch_size, client_data_sizes)
        sampled_client_indices = random.sample(range(num_clients), max(1, int(user_sample_rate * num_clients)))
        sampled_clients_models = [clients_models[i] for i in sampled_client_indices]
        sampled_clients_train_loaders = [clients_train_loaders[i] for i in sampled_client_indices]
        sampled_clients_test_loaders = [clients_test_loaders[i] for i in sampled_client_indices]
        clients_model_updates = []
        clients_accuracies = []

        for idx, (client_model, client_trainloader, client_testloader) in enumerate(
                zip(sampled_clients_models, sampled_clients_train_loaders, sampled_clients_test_loaders)):
            if not args.store:
                tqdm.write(f'client:{idx + 1}/{args.num_clients}')
            local_model = local_update(client_model, client_trainloader,
                                       noise_multiplier * (math.sqrt(float(chunk_mi[idx]) / float(min(chunk_mi)))))
            client_update = [param.data - global_weight for param, global_weight in
                             zip(local_model.parameters(), global_model.parameters())]
            clients_model_updates.append(client_update)
            accuracy = test(local_model, client_testloader)
            clients_accuracies.append(accuracy)
        print(clients_accuracies)
        mean_acc_s.append(sum(clients_accuracies) / len(clients_accuracies))
        print(mean_acc_s)
        acc_matrix.append(clients_accuracies)
        sampled_client_data_sizes = [client_data_sizes[i] for i in sampled_client_indices]
        sampled_client_weights = [
            sampled_client_data_size / sum(sampled_client_data_sizes)
            for sampled_client_data_size in sampled_client_data_sizes
        ]
        aggregated_update = [
            torch.sum(
                torch.stack(
                    [
                        noisy_update[param_index] * sampled_client_weights[idx]
                        for idx, noisy_update in enumerate(clients_model_updates)
                    ]
                ),
                dim=0,
            )
            for param_index in range(len(clients_model_updates[0]))
        ]
        with torch.no_grad():
            for global_param, update in zip(global_model.parameters(), aggregated_update):
                global_param.add_(update)
        for client_model in clients_models:
            client_model.load_state_dict(global_model.state_dict())

        log_record = {
            "epoch": epoch,
            "mean_accuracy": mean_acc_s,
            "client_accuracies": clients_accuracies,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_record) + "\n")  # 一行一轮结果，便于断点恢复
    char_set = '1234567890abcdefghijklmnopqrstuvwxyz'
    ID = ''
    for ch in random.sample(char_set, 5):
        ID = f'{ID}{ch}'
    print(
        f'===============================================================\n'
        f'task_ID : '
        f'{ID}\n'
        f'main_base\n'
        f'noise_multiplier : {noise_multiplier}\n'
        f'mean accuracy : \n'
        f'{mean_acc_s}\n'
        f'acc matrix : \n'
        f'{torch.tensor(acc_matrix)}\n'
        f'===============================================================\n'
    )


if __name__ == '__main__':
    main()
