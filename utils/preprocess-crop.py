import os
import json
from multiprocessing import Manager, Pool
from PIL import Image

Image.MAX_IMAGE_PIXELS = 933120000

from helper import transform_bbox, transform_polygons

suffix = ["train", "test", "val"]


def process_obj(obj, new_dataset):
    annos = obj["annotations"]
    file_name: str = obj["file_name"]
    height: str = obj["height"]
    width: str = obj["width"]

    img = Image.open(file_name)

    for outer_line in annos:
        # outer_line
        if outer_line["category_id"] != 1:
            continue
        outer_line_bbox = outer_line["bbox"]

        stomata_exists = False
        for stomata in annos:
            if stomata["category_id"] != 0:
                continue
            stomata_bbox = stomata["bbox"]
            stomata_segmentation = stomata["segmentation"]
            if (
                stomata_bbox[0] >= outer_line_bbox[0]
                and stomata_bbox[1] >= outer_line_bbox[1]
                and stomata_bbox[2] <= outer_line_bbox[2]
                and stomata_bbox[3] <= outer_line_bbox[3]
            ):
                stomata_exists = True
                break

        if stomata_exists:
            cropped_img = img.crop(outer_line_bbox)
            new_ann = {
                "category_id": 0,
                "bbox": transform_bbox(
                    stomata_bbox,
                    outer_line_bbox[0],
                    outer_line_bbox[1],
                ),
                "segmentation": transform_polygons(
                    stomata_segmentation,
                    outer_line_bbox[0],
                    outer_line_bbox[1],
                ),
            }

            name, ext = file_name.split(".", 1)
            filename = name.split("/")[-1]
            image_output_path = (
                f"{output_image_path}/{filename}_{len(new_dataset)}.{ext}"
            )
            cropped_img.save(image_output_path)
            new_dataset.append(
                {
                    "file_name": image_output_path,
                    "height": outer_line_bbox[3] - outer_line_bbox[1],
                    "width": outer_line_bbox[2] - outer_line_bbox[0],
                    "annotations": [new_ann],
                }
            )


for s in suffix:
    annFile = (
        f"/mnt/local/abrc/coco_dataset/StomaVision-multilabel/labels/labels_{s}.json"
    )
    output_path = "/mnt/local/abrc/coco_dataset/StomaVision-cropped"
    output_image_path = os.path.join(output_path, "images", s)
    output_label_path = os.path.join(output_path, "labels")
    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(output_label_path, exist_ok=True)
    new_dataset = Manager().list()
    with open(annFile, "r") as f:
        dataset = json.load(f)
        with Pool(28) as pool:
            futures = [
                pool.apply_async(process_obj, [obj, new_dataset]) for obj in dataset
            ]

            [r.get() for r in futures]

        output_dataset = list(new_dataset)
        for i in range(len(output_dataset)):
            output_dataset[i]["image_id"] = i

        with open(f"{output_label_path}/labels_{s}.json", "w") as f:
            json.dump(list(output_dataset), f)
