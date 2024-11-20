# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 segment model on a segment dataset

Usage:
    $ bash data/scripts/get_coco.sh --val --segments  # download COCO-segments val split (1G, 5000 images)
    $ python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640-  # validate COCO-segments

Usage - formats:
    $ python segment/val.py --weights yolov5s-seg.pt                 # PyTorch
                                      yolov5s-seg.torchscript        # TorchScript
                                      yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s-seg.xml                # OpenVINO
                                      yolov5s-seg.engine             # TensorRT
                                      yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                      yolov5s-seg_saved_model        # TensorFlow SavedModel
                                      yolov5s-seg.pb                 # TensorFlow GraphDef
                                      yolov5s-seg.tflite             # TensorFlow Lite
                                      yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import json
import os
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import pprint
from torchvision.utils import save_image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch.nn.functional as F

from models.common import DetectMultiBackend
from models.yolo import SegmentationModel
from utils.callbacks import Callbacks
from utils.general import (
    LOGGER,
    NUM_THREADS,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
    xywh2xyxy,
    xyxy2xywh,
    clip_coords,
)
from utils.metrics import ConfusionMatrix, box_iou
from utils.plots import output_to_target, plot_val_study
from utils.segment.dataloaders import create_dataloader
from utils.segment.general import (
    mask_iou,
    process_mask,
    process_mask_upsample,
    scale_masks,
    expand_and_shift_masks,
)
from utils.segment.metrics import Metrics, ap_per_class_box_and_mask
from utils.segment.plots import plot_images_and_masks
from utils.torch_utils import de_parallel, select_device, smart_inference_mode
from utils.augmentations import letterbox_tensor


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (
            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
        )  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map, pred_masks):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    from pycocotools.mask import encode

    def single_encode(x):
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    pred_masks = np.transpose(pred_masks, (2, 0, 1))
    with ThreadPool(NUM_THREADS) as pool:
        rles = pool.map(single_encode, pred_masks)
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
                "segmentation": rles[i],
            }
        )


def process_batch(
    detections, labels, iouv, pred_masks=None, gt_masks=None, overlap=False, masks=False
):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    if masks:
        if overlap:
            nl = len(labels)
            index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
            gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(
                gt_masks[None],
                pred_masks.shape[1:],
                mode="bilinear",
                align_corners=False,
            )[0]
            gt_masks = gt_masks.gt_(0.5)
        iou = mask_iou(
            gt_masks.view(gt_masks.shape[0], -1),
            pred_masks.view(pred_masks.shape[0], -1),
        )
    else:  # boxes
        iou = box_iou(labels[:, 1:], detections[:, :4])

    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where(
            (iou >= iouv[i]) & correct_class
        )  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def pad_or_truncate(tensor, x_fixed, device):
    # If the tensor's first dimension is greater than x_fixed, truncate it to size x_fixed.
    if tensor.shape[0] > x_fixed:
        return tensor[:x_fixed]
    elif tensor.shape[0] < x_fixed:
        zero_tensors = torch.zeros(
            (x_fixed - tensor.shape[0],) + tuple(tensor[0].shape)
        ).to(device, non_blocking=True)
        return torch.cat([tensor, zero_tensors], dim=0)

    return tensor


@smart_inference_mode()
def run(
    data,
    weights_stage1=None,  # model.pt path(s)
    weights_stage2=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val-seg",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    overlap=False,
    mask_downsample_ratio=1,
    compute_loss=None,
    callbacks=Callbacks(),
):
    if save_json:
        check_requirements(["pycocotools"])
        process = process_mask_upsample  # more accurate
    else:
        process = process_mask  # faster

    # Initialize/load model and set device
    device = select_device(device, batch_size=batch_size)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Load model
    model_stage1 = DetectMultiBackend(
        weights_stage1, device=device, dnn=dnn, data=data, fp16=half
    )
    stride, pt, jit, engine = (
        model_stage1.stride,
        model_stage1.pt,
        model_stage1.jit,
        model_stage1.engine,
    )
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    half = model_stage1.fp16  # FP16 supported on limited backends with CUDA
    nm = (
        de_parallel(model_stage1).model.model[-1].nm
        if isinstance(model_stage1, SegmentationModel)
        else 32
    )  # number of masks
    if engine:
        batch_size = model_stage1.batch_size
    else:
        device = model_stage1.device
        if not (pt or jit):
            batch_size = 1  # export.py models default to batch-size 1
            LOGGER.info(
                f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models"
            )

    # Load model
    model_stage2 = DetectMultiBackend(
        weights_stage2, device=device, dnn=dnn, data=data, fp16=half
    )

    # Data
    data = check_dataset(data)  # check

    # Configure
    model_stage1.eval()
    model_stage2.eval()
    cuda = device.type != "cpu"
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(
        f"coco{os.sep}val2017.txt"
    )  # COCO dataset
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    model_stage1.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    model_stage2.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    pad = 0.0 if task in ("speed", "benchmark") else 0.5
    rect = False if task == "benchmark" else pt  # square inference for benchmarks
    task = (
        task if task in ("train", "val", "test") else "val"
    )  # path to train/val/test images
    dataloader = create_dataloader(
        data[task],
        imgsz,
        batch_size,
        stride,
        single_cls,
        pad=pad,
        rect=rect,
        workers=workers,
        prefix=colorstr(f"{task}: "),
        overlap_mask=overlap,
        mask_downsample_ratio=mask_downsample_ratio,
    )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {0: "stomata", 1: "outer_line"}
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ("%22s" + "%11s" * 10) % (
        "Class",
        "Images",
        "Instances",
        "Box(P",
        "R",
        "mAP50",
        "mAP50-95)",
        "Mask(P",
        "R",
        "mAP50",
        "mAP50-95)",
    )
    dt = Profile(), Profile(), Profile()
    metrics = Metrics()
    loss = torch.zeros(4, device=device)
    jdict, stats = [], []
    # callbacks.run('on_val_start')
    pbar = tqdm(
        dataloader, desc=s, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    )  # progress bar
    for batch_i, (im, targets, paths, shapes, masks) in enumerate(pbar):
        # plot stage 2
        plot_stage_2_list = []
        # callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
                masks = masks.to(device)
            masks = masks.float()
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            out_stage1, train_out_stage1 = model_stage1(
                im
            )  # if training else model(im, augment=augment, val=True)  # inference, loss

        # Loss
        if compute_loss:
            loss += compute_loss(train_out_stage1, targets, masks)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor(
            (width, height, width, height), device=device
        )  # to pixels
        lb = (
            [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
        )  # for autolabelling
        with dt[2]:
            out_stage1 = non_max_suppression(
                out_stage1,
                conf_thres,
                iou_thres,
                labels=lb,
                multi_label=True,
                agnostic=True,
                max_det=100,
                nm=nm,
            )

        # Metrics
        plot_masks = []  # masks for plotting
        for si, pred in enumerate(out_stage1):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct_masks = torch.zeros(
                npr, niou, dtype=torch.bool, device=device
            )  # init
            correct_bboxes = torch.zeros(
                npr, niou, dtype=torch.bool, device=device
            )  # init
            seen += 1

            # ============================================================================================
            # =======================================Stage2===============================================
            # ============================================================================================
            pred_c = pred.clone()
            clip_coords(pred_c[:, :4], (height, width))

            pred_stage2_list = []
            plot_dets = []
            pred_mask_stage2_list = []
            for box in pred_c[:, :4]:
                x1, y1, x2, y2 = torch.round(box).to(torch.int)
                cropped_image_original = im[si][:, y1:y2, x1:x2]
                cropped_image, ratio_stage2, pad_stage2 = letterbox_tensor(
                    im=cropped_image_original,
                    new_shape=im[si].shape[1:],
                    auto=False,
                )

                # save_image(im[si], "/home/heiru/projects/instill/StomaVision/ogg.jpg")
                # save_image(cropped_image_original, "/home/heiru/projects/instill/StomaVision/og.jpg")
                # save_image(cropped_image, "/home/heiru/projects/instill/StomaVision/scale.jpg")

                cropped_image = cropped_image[None]
                out_stage2, train_out_stage2 = model_stage2(cropped_image)
                out_stage2 = non_max_suppression(
                    out_stage2,
                    conf_thres,
                    iou_thres,
                    labels=lb,
                    multi_label=True,
                    agnostic=True,
                    max_det=100,
                    nm=nm,
                )

                for sii, pred_stage2 in enumerate(out_stage2):
                    proto_out_stage2 = train_out_stage2[1][sii]
                    if pred_stage2.shape[0] == 0:
                        continue

                    # Predictions
                    ## to stage 1 inference space pred
                    pred_stage2n = pred_stage2.clone()
                    scale_coords(
                        cropped_image[sii].shape[1:],
                        pred_stage2n[:, :4],
                        cropped_image_original.shape[1:],
                        ratio_pad=[ratio_stage2, pad_stage2],
                    )
                    ## transform pred
                    offsets = torch.tensor([x1, y1, x1, y1]).to(
                        device, non_blocking=True
                    )
                    pred_stage2n[:, :4] += offsets
                    plot_dets.append(pred_stage2n.clone())
                    ## to stage 1 native-space pred
                    scale_coords(
                        im[si].shape[1:],
                        pred_stage2n[:, :4],
                        shape,
                        shapes[si][1],
                    )

                    # Masks
                    pred_masks_stage2 = process(
                        proto_out_stage2,
                        pred_stage2[:, 6:],
                        pred_stage2[:, :4],
                        shape=cropped_image[sii].shape[1:],
                        upsample=True,
                    )

                    pred_masks_stage2 = scale_masks(
                        cropped_image[sii].shape[1:],
                        pred_masks_stage2.permute(1, 2, 0).contiguous().cpu().numpy(),
                        [
                            cropped_image_original.shape[1],
                            cropped_image_original.shape[2],
                            3,
                        ],
                        # ratio_pad=[ratio_stage2, pad_stage2],
                    )
                    pred_masks_stage2 = torch.tensor(
                        pred_masks_stage2, device=device
                    ).permute(2, 0, 1)
                    # transform mask
                    pred_masks_stage2 = expand_and_shift_masks(
                        pred_masks_stage2,
                        x_offset=x1,
                        y_offset=y1,
                        target_shape=cropped_image[sii].shape[1:],
                    )

                    pred_stage2_list.append(pred_stage2n)
                    pred_mask_stage2_list.append(pred_masks_stage2)

            pred_stage2_list = [
                pad_or_truncate(t, int(200 / pred_c.shape[0]), device)
                for t in pred_stage2_list
            ]
            pred_stage2n = torch.cat(pred_stage2_list, dim=0)
            pred_mask_stage2_list = [
                pad_or_truncate(t, int(200 / pred_c.shape[0]), device)
                for t in pred_mask_stage2_list
            ]
            pred_mask_stage2n = torch.cat(pred_mask_stage2_list, dim=0)

            plot_dets = [
                pad_or_truncate(t, int(200 / pred_c.shape[0]), device)
                for t in plot_dets
            ]
            plot_dets = torch.cat(plot_dets, dim=0)
            # ============================================================================================
            # =======================================Stage2===============================================
            # ============================================================================================

            if npr == 0:
                if nl:
                    stats.append(
                        (
                            correct_masks,
                            correct_bboxes,
                            *torch.zeros((2, 0), device=device),
                            labels[:, 0],
                        )
                    )
                    if plots:
                        confusion_matrix.process_batch(
                            detections=None, labels=labels[:, 0]
                        )
                continue

            # Masks
            midx = [si] if overlap else targets[:, 0] == si
            gt_masks = masks[midx]
            proto_out = train_out_stage1[1][si]
            pred_masks = process(
                proto_out,
                pred[:, 6:],
                pred[:, :4],
                shape=im[si].shape[1:],
                upsample=True,
            )

            # Predictions
            pred[:, 5] = 1
            predn = pred.clone()
            scale_coords(
                im[si].shape[1:], predn[:, :4], shape, shapes[si][1]
            )  # native-space pred

            # concat
            # print(pred_masks.shape)
            # print(pred_masks_stage2.shape)
            final_pred_masks = torch.cat((pred_masks, pred_mask_stage2n), dim=0)
            final_predn = torch.cat((predn, pred_stage2n), dim=0)

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(
                    im[si].shape[1:], tbox, shape, shapes[si][1]
                )  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct_bboxes = process_batch(final_predn, labelsn, iouv)
                correct_masks = process_batch(
                    final_predn,
                    labelsn,
                    iouv,
                    final_pred_masks,
                    gt_masks,
                    overlap=overlap,
                    masks=True,
                )
                if plots:
                    confusion_matrix.process_batch(final_predn, labelsn)
            stats.append(
                (
                    correct_masks,
                    correct_bboxes,
                    final_predn[:, 4],
                    final_predn[:, 5],
                    labels[:, 0],
                )
            )  # (conf, pcls, tcls)

            final_pred_masks = torch.as_tensor(final_pred_masks, dtype=torch.uint8)
            if plots and batch_i < 3:
                plot_masks.append(final_pred_masks.cpu())  # filter top 15 to plot

            # Save/log
            if save_txt:
                save_one_txt(
                    final_predn,
                    save_conf,
                    shape,
                    file=save_dir / "labels" / f"{path.stem}.txt",
                )
            if save_json:
                final_pred_masks = scale_masks(
                    im[si].shape[1:],
                    final_pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                    shape,
                    shapes[si][1],
                )
                save_one_json(
                    final_predn, jdict, path, class_map, final_pred_masks
                )  # append to COCO-JSON dictionary
            # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])
            plot_stage_2_list.append(plot_dets)

        for i in range(len(out_stage1)):
            out_stage1[i] = torch.cat((out_stage1[i], plot_stage_2_list[i]), dim=0)
        plot_stage_2_list = [pad_or_truncate(t, 300, device) for t in out_stage1]
        plot_stage_2_list = torch.stack(plot_stage_2_list, dim=0)
        # Plot images
        if plots and batch_i < 3:
            if len(plot_masks):
                plot_masks = torch.cat(plot_masks, dim=0)
            plot_images_and_masks(
                im,
                targets,
                masks,
                paths,
                save_dir / f"val_batch{batch_i}_labels.jpg",
                names,
            )
            plot_images_and_masks(
                im,
                output_to_target(plot_stage_2_list, max_det=300),
                plot_masks,
                paths,
                save_dir / f"val_batch{batch_i}_pred.jpg",
                names,
            )  # pred

        # callbacks.run('on_val_batch_end')

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        results = ap_per_class_box_and_mask(
            *stats, plot=plots, save_dir=save_dir, names=names
        )
        metrics.update(results)
    nt = np.bincount(stats[4].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 8  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), *metrics.mean_results()))
    if nt.sum() == 0:
        LOGGER.warning(
            f"WARNING: no labels found in {task} set, can not compute metrics without labels ⚠️"
        )

    # Print results per class
    if (verbose or (nc < 50)) and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            LOGGER.info(pf % (names[c], seen, nt[c], *metrics.class_result(i)))

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    shape = (batch_size, 3, imgsz, imgsz)
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}"
        % t
    )

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    # callbacks.run('on_val_end')

    mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask = (
        metrics.mean_results()
    )

    # Return results
    s = (
        f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
        if save_txt
        else ""
    )
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    final_metric = (
        mp_bbox,
        mr_bbox,
        map50_bbox,
        map_bbox,
        mp_mask,
        mr_mask,
        map50_mask,
        map_mask,
    )
    return (
        (*final_metric, *(loss.cpu() / len(dataloader)).tolist()),
        metrics.get_maps(nc),
        t,
    )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=ROOT / "data/coco128-seg.yaml",
        help="dataset.yaml path",
    )
    parser.add_argument(
        "--weights_stage1",
        nargs="+",
        type=str,
        default=ROOT / "yolov5s-seg.pt",
        help="model path(s)",
    )
    parser.add_argument(
        "--weights_stage2",
        nargs="+",
        type=str,
        default=ROOT / "yolov5s-seg.pt",
        help="model path(s)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        type=int,
        default=640,
        help="inference size (pixels)",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.001, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.6, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max-det", type=int, default=300, help="maximum detections per image"
    )
    parser.add_argument(
        "--task", default="val", help="train, val, test, speed or study"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="max dataloader workers (per RANK in DDP mode)",
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="treat as single-class dataset"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-hybrid",
        action="store_true",
        help="save label+prediction hybrid results to *.txt",
    )
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-json", action="store_true", help="save a COCO-JSON results file"
    )
    parser.add_argument(
        "--project", default=ROOT / "runs/val-seg", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    # opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(
                f"WARNING: confidence threshold {opt.conf_thres} > 0.001 produces invalid results ⚠️"
            )
        if opt.save_hybrid:
            LOGGER.info(
                "WARNING: --save-hybrid will return high mAP from hybrid labels, not from predictions alone ⚠️"
            )
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                x, y = (
                    list(range(256, 1536 + 128, 128)),
                    [],
                )  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            os.system("zip -r study.zip study_*.txt")
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
