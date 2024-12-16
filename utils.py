# -*- coding: utf-8 -*-
# @Time    : 2024/11/4 20:05
# @Author  : lil louis
# @Location: Beijing
# @File    : utils.py

import pandas as pd
import os
from sklearn.model_selection import KFold
from dataset import OSADataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CO_Single, CO_Both, block
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import torch


def get_loaders(excel_dir, batch_size, oral_dir, cranio_dir,
                oral_transform, cranio_transform, fold_index, n_splits=10):
    id_df = pd.read_excel(excel_dir)
    id_map = {row['idx']: {
        'front_area': row['front_area'],
        'front_length': row['front_length'],
        'front_width': row['front_width'],
        'profile_area': row['profile_area'],
        'profile_length': row['profile_length'],
        'profile_thickness': row['profile_thickness'],
        'profile_curvature': row['profile_curvature'],
        'label': row['label']
    } for index, row in id_df.iterrows()}

    all_indices = list(range(len(os.listdir(oral_dir)) // 2))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(kf.split(all_indices))
    train_indices, val_indices = splits[fold_index]

    train_dataset = OSADataset(oral_root=oral_dir,
                               cranio_root=cranio_dir,
                               id_map=id_map,
                               cranio_transform=cranio_transform,
                               oral_transform=oral_transform,
                               indices=[all_indices[i] for i in train_indices])
    val_dataset = OSADataset(oral_root=oral_dir,
                             cranio_root=cranio_dir,
                             id_map=id_map,
                             cranio_transform=cranio_transform,
                             oral_transform=oral_transform,
                             indices=[all_indices[i] for i in val_indices])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader


def get_model(in_channels, num_situation):
    model = None
    if num_situation in {1, 2, 4, 5}:
        model = CO_Single(in_channels=in_channels,
                          block=block,
                          layers=[3, 4, 6, 3])
    elif num_situation in {3, 6}:
        model = CO_Both(in_channels=in_channels,
                        block=block,
                        layers=[3, 4, 6, 3])
    else:
        raise ValueError(f"Unsupported num_situation value: {num_situation}")
    return model


def train_fn(epoch_index, loader, model, loss_fn, optimizer, scaler, num_situation, device):
    model.train()
    loop = tqdm(loader)
    total_loss = 0.0
    y_true, y_pred = [], []
    for index, ((cranio_front, cranio_profile, oral_front, oral_profile),
                (physio_dim, oral_dim, label)) in enumerate(loop):
        cranio_front = cranio_front.to(device, dtype=torch.float32)
        cranio_profile = cranio_profile.to(device, dtype=torch.float32)
        oral_front = oral_front.to(device, dtype=torch.float32)
        oral_profile = oral_profile.to(device, dtype=torch.float32)
        physio_dim = physio_dim.to(device).float()
        oral_dim = oral_dim.to(device).float()
        label = label.to(device).float()
        with torch.cuda.amp.autocast():
            inputs_map = {
                1: (cranio_front, physio_dim, oral_dim),
                2: (cranio_profile, physio_dim, oral_dim),
                3: (cranio_front, cranio_profile, physio_dim, oral_dim),
                4: (oral_front, physio_dim, oral_dim),
                5: (oral_profile, physio_dim, oral_dim),
                6: (oral_front, oral_profile, physio_dim, oral_dim)
            }
            if num_situation in inputs_map:
                outputs = torch.sigmoid(model(*inputs_map[num_situation]))
            else:
                raise ValueError("Invalid situation number")
            outputs = outputs.squeeze(1).float()
            loss = loss_fn(outputs, label)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        y_true.extend(label.cpu().detach().numpy())
        y_pred.extend(outputs.cpu().detach().numpy())
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    result_cal(epoch_index, y_true, y_pred, num_situation, "train")
    return None



def check_accuracy(epoch_index, loader, model, loss_fn, num_situation, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for (cranio_front, cranio_profile, oral_front, oral_profile), (physio_dim, oral_dim, label) in loader:
            cranio_front = cranio_front.to(device, dtype=torch.float32)
            cranio_profile = cranio_profile.to(device, dtype=torch.float32)
            oral_front = oral_front.to(device, dtype=torch.float32)
            oral_profile = oral_profile.to(device, dtype=torch.float32)
            physio_dim = physio_dim.to(device).float()
            oral_dim = oral_dim.to(device).float()
            label = label.to(device).float()
            with torch.cuda.amp.autocast():
                inputs_map = {
                    1: (cranio_front, physio_dim, oral_dim),
                    2: (cranio_profile, physio_dim, oral_dim),
                    3: (cranio_front, cranio_profile, physio_dim, oral_dim),
                    4: (oral_front, physio_dim, oral_dim),
                    5: (oral_profile, physio_dim, oral_dim),
                    6: (oral_front, oral_profile, physio_dim, oral_dim)
                }
                if num_situation in inputs_map:
                    outputs = torch.sigmoid(model(*inputs_map[num_situation]))
                else:
                    raise ValueError("Invalid situation number")
                outputs = outputs.squeeze(1).float()
                loss = loss_fn(outputs, label)
            y_true.extend(label.cpu().detach().numpy())
            y_pred.extend(outputs.cpu().detach().numpy())
            total_loss += loss.item()
    result_cal(epoch_index, y_true, y_pred, num_situation, "val")
    return None


def result_cal(epoch_index, y_true, y_pred, num_situation, status):
    print(y_true)
    print(y_pred)
    y_true = np.array(y_true)
    y_pred_p = np.array(y_pred).ravel()
    y_pred_label = (y_pred_p > 0.5).astype(np.float32)

    cm = confusion_matrix(y_true, y_pred_label)
    accuracy = accuracy_score(y_true, y_pred_label)
    precision = precision_score(y_true, y_pred_label)
    sensitivity = recall_score(y_true, y_pred_label)

    print(f"Epoch: {epoch_index}, Confusion Matrix:\n{cm}")
    print(f"Epoch: {epoch_index}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Sensitivity: {sensitivity:.4f}")

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_p)
    roc_auc = auc(fpr, tpr)
    Roc_Save(epoch_index, fpr, tpr, roc_auc, num_situation, status)
    CI_Roc(epoch_index, y_true, y_pred_p)
    return None


def CI_Roc(epoch_index, y_true, y_pred):
    assert len(y_true) == len(y_pred)
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    bootstrapped_aucs = []

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_aucs.append(score)

    alpha = 0.95
    lower = np.percentile(bootstrapped_aucs, (1-alpha) / 2 * 100)
    upper = np.percentile(bootstrapped_aucs, (alpha + (1-alpha) / 2) * 100)
    mean_auc = np.mean(bootstrapped_aucs)

    print(f"Epoch: {epoch_index}, Mean AUC: {mean_auc:.2f}, 95% Confidence Interval：[{lower:.2f} - {upper:.2f}]")
    return None


def Roc_Save(epoch_index, fpr, tpr, roc_auc, num_situation, status):
    situation = ["Cranio_Front", "Cranio_Profile", "Cranio_FAP",
                 "Oral_Front", "Oral_Profile", "Oral_FAP"]
    file_save_path = os.path.join("./result", situation[num_situation - 1], status, f"{epoch_index}_{roc_auc:.2f}")

    if not os.path.isdir(file_save_path):
        os.makedirs(file_save_path)

    np.save(os.path.join(file_save_path, "fpr.npy"), fpr)
    np.save(os.path.join(file_save_path, "tpr.npy"), tpr)
    return None