#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet18
import torch
from tensorboardX import SummaryWriter
from sklearn.feature_selection import mutual_info_regression
# from options import args_parser
# from client_update import LocalUpdate, test_inference
# from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
# from utils import get_dataset, average_weights, exp_details
import warnings
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore")


def mutual_information(train_loader, client_update):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    correlations = []
    mi_scores = []
    """
    计算训练集与梯度之间的互信息
    """
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        for i in range(data.size(0)):
            sample_data = data[i].view(1, -1).cpu().numpy().flatten()
            sample_gradients = []

            for param in client_update:
                if param is not None:
                    sample_grad = param.detach().numpy()
                    sample_gradients.append(sample_grad.flatten())

            sample_gradients = np.concatenate(sample_gradients)

            # 确保 sample_data 和 sample_gradients 长度一致
            min_length = min(len(sample_data), len(sample_gradients))

            sample_data = sample_data[:min_length]
            sample_gradients = sample_gradients[:min_length]

            # 计算样本整体的相关性
            if len(sample_data) >= 2 and len(sample_gradients) >= 2:
                corr, _ = pearsonr(sample_data, sample_gradients)
                correlations.append(corr)
            # 计算样本整体的互信息
            if len(sample_data) >= 2 and len(sample_gradients) >= 2:
                mi = mutual_info_regression(sample_data.reshape(-1, 1), sample_gradients)
                mi_scores.append(np.mean(mi))

    # 计算平均互信息
    avg_mi = np.mean(mi_scores) if mi_scores else 0
    return avg_mi
