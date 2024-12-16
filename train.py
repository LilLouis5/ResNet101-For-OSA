# -*- coding: utf-8 -*-
# @Time    : 2024/11/5 16:54
# @Author  : lil louis
# @Location: Beijing
# @File    : train.py


from utils import get_loaders, get_model, train_fn, check_accuracy

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torch import optim

def main():
    num_situation = 6  # [1 ~ 6]
    batch_size = 4
    n_splits = 10
    epochs = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 256
    in_channels = 3
    learning_rate = 1e-6
    adam_weight_decay = 0
    adam_betas = (0.9, 0.999)
    loss_weight = torch.tensor([3.0, 2.0], device=device)

    label_excel_dir = "./tongue_clear.xlsx"
    oral_dir = "./dataset_oral"
    cranio_dir = "./dataset_cranio"

    oral_transform = A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ], is_check_shapes=False)

    cranio_transform = A.Compose(
        [A.Resize(height=image_size, width=image_size),
         A.Rotate(limit=35, p=1.0),
         A.HorizontalFlip(p=0.1),
         A.VerticalFlip(p=0.1),
         ToTensorV2(), ]
    )

    model = get_model(in_channels=in_channels,
                      num_situation=num_situation).to(device)
    pos_weight = torch.tenosr([2.0, 1.0], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), betas=adam_betas, lr=learning_rate,weight_decay=adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    total_epoch = 1
    for fold_index in range(n_splits):
        train_dataloader, val_dataloader = get_loaders(label_excel_dir, batch_size, oral_dir, cranio_dir, oral_transform,
                                                       cranio_transform, fold_index, n_splits)
        for epoch in range(epochs):
            train_fn(total_epoch, train_dataloader, model, loss_fn, optimizer, scaler,
                     num_situation, device)
            check_accuracy(total_epoch, val_dataloader, model, loss_fn,
                           num_situation, device)
            total_epoch += 1
    print("Over! Happy Ending!")
    return None


if __name__ == "__main__":
    main()