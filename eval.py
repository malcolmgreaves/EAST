import time
from pathlib import Path

import torch
import subprocess
import os
from model import EAST
from detect import detect_dataset, default_name_filter, default_draw_for_name
import shutil

from reusable import load_east_model


def eval_model(
    model_name: Path,
    test_img_path: Path,
    submit_path: Path,
    save_flag: bool = True,
    clear_submit: bool = True,
):
    if submit_path.exists():
        if clear_submit:
            shutil.rmtree(str(submit_path.absolute()))
        else:
            raise ValueError(
                f"Submit path exists and clearing is set to false: '{submit_path.absolute()}'"
            )
    os.mkdir(str(submit_path))

    model, device = load_east_model(model_name, pretrained=False)

    start_time = time.time()

    detect_dataset(
        model, device, test_img_path, submit_path, name_filter=default_name_filter
    )

    os.chdir(str(submit_path))
    try:
        subprocess.getoutput("zip -q submit.zip *.txt")
        subprocess.getoutput("mv submit.zip ../")
    finally:
        os.chdir("../")
    res = subprocess.getoutput(
        "python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip"
    )
    print("-" * 80)
    print(res)
    print("-" * 80)
    os.remove("./submit.zip")

    print(f"Evaluation time: {time.time() - start_time}")

    if not save_flag:
        shutil.rmtree(submit_path)


if __name__ == "__main__":
    model_name = Path("./pths/east_vgg16.pth")
    test_img_path = Path("../ICDAR_2015/test_img")
    submit_path = Path("./submit")
    eval_model(model_name, test_img_path, submit_path)
