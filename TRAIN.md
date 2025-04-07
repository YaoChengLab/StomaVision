# Train StomaVision: Instance Segmentation with Yolov7

## Step 1: Install dependencies

Install all the dependencies using pip.

```shell
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

## Step 2: Prepare datasets

Dataset structure here is identical to [Yolov5](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data). Read for further details.

### Preprocessed dataset
You can download our preprocesesed dataset to fine-tune YoloV7 model. And skip to [Step 4](#step-4-prepare-configuration-yaml-files)

- [multilabel dataset](https://drive.google.com/file/d/1kjNbFXuDsFbnY7DLG_O-ytBnugtyFIW3/view?usp=drive_link)
  - for fine-tuning model that detects both outer-line and pore
- [outer-line dataset](https://drive.google.com/file/d/1_D-kOylGTpYnxAfBpEdQSo8U7wRLWUIf/view?usp=drive_link)
  - for fine-tuning ensemble model outer-line detection
- [cropped dataset](https://drive.google.com/file/d/1_D-kOylGTpYnxAfBpEdQSo8U7wRLWUIf/view?usp=drive_link)
  - for fine-tuning ensemble model pore ddetection

### Folder Structure

The dataset must be prepared in a specific way. The training script will first fetch images using the path defined in the `{custom_dataset}.yaml` stored under `data`. The script finds the corresponding labels replace train:

- images: `{dataset_dir}/{dataset_name}/images/train/{images}`
- labels: `{dataset_dir}/{dataset_name}/labels/train/{images}`

Each image should have a corresponding label file consisting of the annotations.

```
{dataset_dir}
├── {dataset_name}
│   ├── images           < images must be stored in this dir
│   │   ├──train
│   │   ├──val
│   │   ├──test
│   ├── labels
│   │   ├──train
│   │   ├──val
│   │   ├──test
│   │   ├──labels.json
├── {dataset_name_2}     < you can store multiple dataset under the directory
└── README.md
```

### Label file formats

- BBOX: `class center_x center_y width height`
- POLYGON: `class x1 y1 x2 y2 x3 y3 .. xn yn`

## Step 3: Convert from Detectron2 JSON to Yolov7 compatible

We currently prepare two scripts under the `utils` folder for users to prepare dataset and convert annotations to YOLOv7 compatible format.

### 3-1. Convert from Detectron2 datasets

Note that if the data is annoted with label studio, users need to prepare their dataset with script `data-prep-lablestudio.ipynb` in advance.

`data-prep-detectron2.ipynb` converts labels from `Detectron` JSON format to Yolov7 compatible.

> **NOTE**: Data suffling (i.e. split dataset into train, val and test) is done by the `data-prep-lablestudio.ipynb`. `data-prep-detectron2.ipynb` only convert and prepare annotation files accordindly.

### 3-2. Convert from COCO-data format

`data-prep-coco.ipynb` converts labels from `COCO` JSON format to Yolov7 compatible. In addition to converting annotation formants as the script for Detectron2, this script also help to shuffle image and annotation. `SAI` dataset are prepared using this script. Further information about COCO-data format, please see:

- [create coco annotation from scratch](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch)
- [coco dataset official](https://cocodataset.org/#home)

### 3-3. Visulase annotations

To test the correctness of converted annotations, we prepare `draw-mark.ipynb` for users to visulise annotations on images. User can also use this script to generate ground truth examples for desmonstration.

> **NOTE**: For further information about how to prepare you dataset, see references below:
>
> - [Author's instruction](https://github.com/WongKinYiu/yolov7/issues/752)
> - [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

### 3-4. Create cropped dataset
For fine-tuning ensemble model's second stage(pore detection), you will need to prepare an additional datset that crops out the stomata label and image. Please refer to [this script](./utils/preprocess-crop.py) to preprocess the original dataset and create a cropped dataset.
- You may need to change the `catalog_id` in the script depending on your label

## Step 4: Prepare configuration YAML files

There are three YAML files need to be configured before training:

- `data/{custom_data}.yaml`: a YAML file defining the location of datasets and labels.
- `cfg/{custom_model}.yaml`: a YAML file defining the model structure and class number.
- `hyp/{custom_hypterparamaters}.yaml`: a YAML file defining training hyper-parameters.

You can use the [predefined data files](data) provided if you downloaded our [preprocessed dataset](#preprocessed-dataset), or define a new one for yourself.

> **NOTE**  
> Your probabaly don't want to change `cfg` as it defines the network architecture. However, remember to update the class number to meet your application.

## Step 5: Download the pre-trained model (optional for transfer learning)

You can train a customised model based on pre-trained models to benefit from transferring knowledge from the established models.

- [`yolov7-seg.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-seg.pt)
- [`yolov7x-seg.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x-seg.pt)

They can be download with

```bash
# download `yolov7-seg.pt`
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-seg.pt -O model/yolov7-seg.pt
# download `yolov7x-seg.pt`
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x-seg.pt -O model/yolov7x-seg.pt
```

## Step 6: Modify the training script

We prepare a shell script `train.sh` for better visulisation. A couple of basic settings need to be updated as to your development environment before training.

- `--device`: set to `cpu` if you use CPU on M1, set it to `0` (i.e. GPU index) if you are using cuda-enabled GPU.
- `--data`: a YAML file defining the location of datasets and labels.
- `--cfg`: a YAML file defining the model structure and class number.
- `--hyp`: a YAML file defining training hyper-parameters.
- `--weight`: a `.py` file storing the pre-trained model. Leave it blank (i.e., `""`) if you wish to train the model from scratch.

```shell
python seg/segment/train.py \
--workers 8 \
--device cpu \
--batch-size 16 \
--data data/stomavision.yaml \
--img 640 \
--cfg cfg/yolov7-seg.yaml \
--weights 'models/yolov7-seg.pt' \
--name yolov7-abrc-stomavision \
--hyp data/hyp.scratch.abrc.yaml

```

We provide a couple of training scripts for stomata detection tasks below:

- `train.sh`: This is a script to train from a pretrained model `model/yolov7-seg.pt` using **single** and **multiple** GPUs.
- `train-seq.sh` This is a script to train from a pretrained model `model/yolov7-seg.pt` with different 1) data-augmentation strategies and 2) batch sizes using **single** and **multiple** GPUs.
- `train-x-seq.sh` This is a script to fraom from a pretrained model `model/yolov7x-seg.pt` with different 1) data-augmentation strategies and 2) batch sizes using **single** and **multiple** GPUs.

> **NOTE**  
> We observed that Yolov7 employs intense data augmentation strategy to inhance model generality. However, this strategy also results in significant data and annotation fragmentation, causing **out of memory error** during training. When enounter this error, we highly recommend users to check the number and the correctness of annotations per image using the `util/draw-mask.ipynb` script.

The outputs will be stored under `/seg/runs/train-seg/`, where `/weights/best.pt` is the model yields the best performance, `/weight/last.pt` is the model generated in the last training iteration.

## Step 7. Validation (Evaluation)

YOLOv7 provides a Python script model validation (evaluation). It can be triggered with

```shell
data="dataset_name"
model=/path/to/your/trained/model/weights
device= 0 # set to `cpu` for infernce with cpu, 0,1,2,3 for inference with GPU.
python seg/segment/val.py \
--workers 8 \
--device $device \
--batch-size 16 \
--data data/$data.yaml \
--img 640 \
--weights $model/weights/best.pt \
--conf 0.3 \
--name $model-$data
```

We also provide a script `val.sh` for users to evaluate their models on different datasets. This will produce some metric plots under `plot` folder.

## Step 8. Inference

YOLOv7 provides a Python script for prediction with trained models. Below demonstrates an example. Ensure you assign image directory `image` and trained model `model` in advance.

```shell
image=/path/to/your/dataset/images/val
model=/path/to/your/weights/best.pt
device=cpu \ # cpu when infernce with cpu, 0,1,2,3 for inference with GPU.
python seg/segment/predict.py \
--device $device \
--weights $model \
--conf 0.3 \
--img-size 640 \
--source $image
```

We also provide a script `inference.sh` for users to inference with models.

## Step 9. Evaluation

Given the inference results, We have created a script `utils/evaluation.ipynb` for performance evaluation. In addition to the required libraries and and helper functions, we provide two blocks of scripts for evluate on inference file or iterate through the entire validation set.
