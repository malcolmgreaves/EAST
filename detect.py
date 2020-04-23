import sys
from pathlib import Path
from typing import (
    Optional,
    Tuple,
    Callable,
    Union,
    Sequence,
    NamedTuple,
    Mapping,
)

import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from PIL.Image import Image as Img
from tqdm import tqdm

from model import EAST
import os
from dataset import get_rotate_mat
import numpy as np
import lanms

Num = Union[int, float]
"""A number.
"""

RGB = Tuple[int, int, int]
"""Values for (red, green, blue) channels: each has value 0 to 255.
"""


def resize_img(img: Img, denominator: int = 32) -> Tuple[Img, float, float]:
    """Resize image to be divisible by `denominator`, which defaults to 32.

    Output:
        resized image, height ratio, width ratio
    """
    w, h = img.size
    assert 1 < denominator < w and denominator < h
    resize_w = w
    resize_h = h

    resize_h = (
        resize_h
        if resize_h % denominator == 0
        else int(resize_h / denominator) * denominator
    )
    resize_w = (
        resize_w
        if resize_w % denominator == 0
        else int(resize_w / denominator) * denominator
    )
    img = img.resize((resize_w, resize_h), Image.BILINEAR)
    ratio_h = resize_h / h
    ratio_w = resize_w / w

    return img, ratio_h, ratio_w


def mk_torchify_img() -> Callable[[Img], torch.Tensor]:
    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return lambda img: t(img).unsqueeze(0)


torchify_image = mk_torchify_img()


def load_pil(img: Img) -> torch.Tensor:
    """Convert PIL Image to torch.Tensor
    """
    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return t(img).unsqueeze(0)


def is_valid_poly(res: np.ndarray, score_shape: Tuple[int, ...], scale: Num) -> bool:
    """Check if the poly is in the image scope.
    Input:
        res        : restored poly in original image
        score_shape: score map shape
        scale      : feature map -> image
    Output:
        True if valid
    """
    cnt = 0
    for i in range(res.shape[1]):
        if (
            res[0, i] < 0
            or res[0, i] >= score_shape[1] * scale
            or res[1, i] < 0
            or res[1, i] >= score_shape[0] * scale
        ):
            cnt += 1
            # TODO - optimization: if cnt > 1: return False
    return cnt <= 1


def restore_polys(
    valid_pos: np.ndarray,
    valid_geo: np.ndarray,
    score_shape: Tuple[int, ...],
    scale: int = 4,
) -> Tuple[np.ndarray, Sequence[int]]:
    """Restore polygons from feature maps in the given positions.

    Input:
        valid_pos  : potential text positions <numpy.ndarray, (n,2)>
        valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
        score_shape: shape of score map
        scale      : Img / feature map
    Output:
        restored polygos <numpy.ndarray, (n,8)>, index
    """
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :]  # 4 x N
    angle = valid_geo[4, :]  # N,

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordidates)
        res[0, :] += x
        res[1, :] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append(
                # format: x0,y0,x1,y1,x2,y2,x3,y3 (origin is upper-left of image)
                # semantics: bottom left (xy), top left, top right, bottom right (clockwise from bottom left corner)
                [
                    res[0, 0],
                    res[1, 0],
                    res[0, 1],
                    res[1, 1],
                    res[0, 2],
                    res[1, 2],
                    res[0, 3],
                    res[1, 3],
                ]
            )

    return np.array(polys), index


def boxes_from_feature_map(
    score: np.ndarray,
    geo: np.ndarray,
    score_thresh: float = 0.9,
    nms_thresh: float = 0.2,
) -> Optional[np.ndarray]:
    """Extract detected bounding boxes from the feature maps.

    Input:
        score       : score map from model <numpy.ndarray, (1,row,col)>
        geo         : geometry map from model <numpy.ndarray, (5,row,col)>
        score_thresh: threshold to segment score map
        nms_thresh  : threshold in nms
    Output:
        boxes       : final polygons <numpy.ndarray, (n,9)>
    """
    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh)  # n x 2, format is [r, c]
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype("float32"), nms_thresh)
    return boxes


def adjust_ratio(boxes: np.ndarray, ratio_w: float, ratio_h: float) -> np.ndarray:
    """Refine boxes.

    Input:
        boxes  : detected polys <numpy.ndarray, (n,9)>
        ratio_w: ratio of width
        ratio_h: ratio of height
    Output:
        refined boxes
    """
    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)


@torch.no_grad()
def detect(img: Img, model: EAST, device: torch.device) -> Optional[np.ndarray]:
    """Detect text regions of img using model; output detected polygons.

    Output:
        Detected polygons <numpy.ndarray, (n,9)>.
        - The first dimension indexes unique text detections.
        - The second dimension contains the exterior quadrilateral points, defining its geometry.
          The points start at the top-left, moving clockwise, in (x,y) pairs as a sequence.
        If the result is `None`, then no text boxes were detected in the image.
    """
    img, ratio_h, ratio_w = resize_img(img)
    score, geo = model(torchify_image(img).to(device))
    boxes = boxes_from_feature_map(
        score=score.squeeze(0).cpu().numpy(), geo=geo.squeeze(0).cpu().numpy(),
    )
    if boxes is None or boxes.size == 0:
        return None
    return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img: Img, boxes: np.ndarray, outline_color: RGB = (0, 255, 0)) -> None:
    """Plot `boxes` on the image `img`.

    NOTE: MUTATION: The original image is mutated, having the polygons defined
                    by `boxes` drawn on it.
    """
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon(
            [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]],
            outline=(0, 255, 0),
        )


def poly_to_str(box: np.ndarray) -> str:
    return ",".join([str(int(b)) for b in box[:-1]])


class Box(NamedTuple):
    x0: int
    y0: int
    x1: int
    y1: int
    x2: int
    y2: int
    x3: int
    y3: int
    rot: int

    def toseq(self) -> Sequence[int]:
        return (
            self.x0,
            self.y0,
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.x3,
            self.y3,
        )

    def todict(self) -> Mapping[str, int]:
        return {
            "top": self.y0,
            "left": self.x0,
            "bottom": self.y3,
            "right": self.x2,
        }


def poly_to_bb(box: np.ndarray) -> Box:
    # return Box(*[int(b) for b in box[:-1]])
    return Box(*[int(b) for b in box])
    # return Box(box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7])


def files_from(
    path: Path, name_filter: Optional[Callable[[str], bool]] = None
) -> Sequence[Path]:
    if path.is_dir():
        img_files = os.listdir(str(path))
        if len(img_files) == 0:
            raise ValueError(f"No files in '{path}'")

        if name_filter:
            img_files = list(filter(name_filter, img_files))
            if len(img_files) == 0:
                raise ValueError(f"Name filter excluded all files in '{path}'")
        return sorted((path / img_file for img_file in img_files), key=str)

    elif path.is_file():
        if name_filter and not name_filter(path.name):
            raise ValueError(f"Rejected single file by name '{path}'")
        return [path]

    else:
        raise ValueError(f"Supplied path is invalid '{path}'.")


def detect_dataset(
    model: EAST,
    device: torch.device,
    test_img_path: Path,
    submit_path: Path,
    name_filter: Optional[Callable[[str], bool]] = None,
    submit_namer: Callable[[str], str] = lambda fi: f'res_{fi.replace(".jpg", ".txt")}',
    name_for_drawn: Optional[Callable[[str], str]] = None,
) -> None:
    """Detection on whole dataset, save .txt results in `submit_path`.

    Input:
        model          : detection model
        device         : gpu if gpu is available
        test_img_path  : dataset path
        submit_path    : submit result for evaluation
        name_filter    : determines which filenames in `test_img_path` to detect on, defaults to all
        submit_namer   : determines the name of the result file for a given image file name
        name_for_drawn : if present, draws bounding boxes & saves the images using this function to name the file
    """
    if not submit_path.is_dir():
        raise ValueError(f"Output path is not a valid directory '{submit_path}'")

    if name_for_drawn:
        print(
            "Saving bounding boxes drawn on each input image to the submission directory."
        )

    img_files = files_from(test_img_path, name_filter)

    for img_fi in tqdm(img_files, "Detecting Text"):
        img = Image.open(img_fi)
        detected_boxes = detect(img, model, device)
        if detected_boxes is not None:
            boxes_as_coords = [poly_to_str(box) + "\n" for box in detected_boxes]
        else:
            boxes_as_coords = []

        out_fi = submit_path / submit_namer(img_fi.name)
        # print(f"Writing {len(boxes_as_coords)} detected boxes from '{img_fi.name}' to '{out_fi.name}'")
        with open(str(out_fi), "wt") as wt:
            wt.writelines(boxes_as_coords)

        if name_for_drawn and detected_boxes is not None:
            plot_boxes(img, detected_boxes)
            img_with_boxes = submit_path / name_for_drawn(img_fi.name)
            img.save(str(img_with_boxes))


def default_draw_for_name(name: str) -> str:
    return f"draw_{name}"


def default_name_filter(name: str) -> bool:
    return not name.startswith(".")


if __name__ == "__main__":
    from reusable import load_east_model

    model, device = load_east_model(
        # Path("./pths/east_vgg16.pth"),
        # Path("./pths/model_epoch_600.pth"),
        Path("./pths/best-loss_0.16445676346377627-epoch_493.pth"),
        set_eval=True,
    )
    test_img_path = Path("../ICDAR_2015/test_img")

    if len(sys.argv) > 1:
        img = Image.open(str(test_img_path / "img_2.jpg"))
        boxes = detect(img, model, device)
        plot_boxes(img, boxes)
        outname = "./res.bmp"
        img.save(outname)
        print(f"Wrote detection to '{outname}'")

    else:
        submit = Path(".") / "submit"
        if not submit.is_dir():
            os.mkdir(str(submit))
        print(f"Running on images in: {test_img_path}'")
        print(f"Writing output to:    {submit}")
        detect_dataset(
            model,
            device,
            test_img_path,
            submit,
            name_filter=default_name_filter,
            name_for_drawn=default_draw_for_name,
        )
