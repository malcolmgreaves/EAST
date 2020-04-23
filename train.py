from pathlib import Path
from typing import Optional

import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm

from dataset import custom_dataset
from model import EAST
from loss import Loss
import os
import time
import numpy as np

from reusable import load_east_model, get_torch_device


def train(
    train_img_path,
    train_gt_path,
    test_img_path,
    test_gt_path,
    pths_path,
    batch_size,
    lr,
    num_workers,
    epoch_iter,
    interval,
    model_checkpoint: Optional[Path] = None,
):
    file_num = len(os.listdir(train_img_path))
    trainset = custom_dataset(train_img_path, train_gt_path)
    train_loader = data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    data_parallel = torch.cuda.device_count() > 1
    if model_checkpoint:
        print(f"Starting from existing model checkpoint '{model_checkpoint}'")
        model, device = load_east_model(model_checkpoint, set_eval=False)
    else:
        model = EAST()
        device = get_torch_device()
    if data_parallel:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[epoch_iter // 2], gamma=0.1
    )

    best_loss = 99999999999.0

    optimizer.step()
    print("=" * 80)
    for epoch in tqdm(range(epoch_iter), "Training", epoch_iter):
        model.train()
        scheduler.step()
        sum_epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in tqdm(
            enumerate(train_loader), "Batches", int(file_num / batch_size)
        ):
            # start_time = time.time()
            img, gt_score, gt_geo, ignored_map = (
                img.to(device),
                gt_score.to(device),
                gt_geo.to(device),
                ignored_map.to(device),
            )
            pred_score, pred_geo = model(img)
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

            sum_epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(
            #     "Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}".format(
            #         epoch + 1,
            #         epoch_iter,
            #         i + 1,
            #         int(file_num / batch_size),
            #         time.time() - start_time,
            #         loss.item(),
            #     )
            # )

        epoch_loss = sum_epoch_loss / int(file_num / batch_size)
        print("\n\n" + "- " * 40)
        print(
            "epoch_loss is {:.8f}, epoch_time is {:.8f}".format(
                epoch_loss, time.time() - epoch_time
            )
        )
        print(time.asctime(time.localtime(time.time())))
        print("=" * 80)

        # should calculate TEST loss for early stopping :-/

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"* * *  FOUND LOWEST LOSS ON EPOCH {epoch+1}: {best_loss}")
            torch.save(
                model.module.state_dict() if data_parallel else model.state_dict(),
                os.path.join(pths_path, f"best-loss_{best_loss}-epoch_{epoch+1}.pth"),
            )

        if (epoch + 1) % interval == 0:
            torch.save(
                model.module.state_dict() if data_parallel else model.state_dict(),
                os.path.join(pths_path, "model_epoch_{}.pth".format(epoch + 1)),
            )

    torch.save(
        model.module.state_dict() if data_parallel else model.state_dict(),
        os.path.join(pths_path, f"final_model.pth"),
    )

    return model, device


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_checkpoint = Path(sys.argv[1]).absolute()
    else:
        model_checkpoint = None
    train_img_path = os.path.abspath("../ICDAR_2015/train_img")
    train_gt_path = os.path.abspath("../ICDAR_2015/train_gt")
    test_img_path = os.path.abspath("../ICDAR_2015/test_img")
    test_gt_path = os.path.abspath("../ICDAR_2015/test_gt")
    pths_path = "./pths"
    batch_size = 24
    lr = 1e-3
    num_workers = 4
    epoch_iter = 600
    save_interval = 5
    train(
        train_img_path,
        train_gt_path,
        test_img_path,
        test_gt_path,
        pths_path,
        batch_size,
        lr,
        num_workers,
        epoch_iter,
        save_interval,
        model_checkpoint=model_checkpoint,
    )
