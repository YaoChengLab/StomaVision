import io
import os
import time
import base64
import cv2
import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
from streamlit_image_select import image_select
from pycocotools import mask as cocomask
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from instill.configuration import global_config
from instill.clients import InstillClient
from instill.resources import Pipeline
from utils import (
    binary_mask_to_polygon,
    fit_polygons_to_rotated_bboxes,
    calc_polygon_area,
    random_color,
)


def cv2_base64(image):
    buffer_img = cv2.imencode(".jpg", image)[1]
    base64_str = base64.b64encode(buffer_img).decode("utf-8")
    return base64_str


def sample_input():
    img = image_select(
        label="Select a sample image to inference",
        use_container_width=False,
        images=[
            "samples/c_tr_56_cut_want.jpg",
            "samples/c_tr_68_cut_want.jpg",
            "samples/u_tr_26.png",
        ],
    )
    img = cv2.imread(img)
    image_dict = {"sample.jpg": img}
    preprocess_and_render_layout(image_dict)


def image_input():
    list_of_img_bytes = st.sidebar.file_uploader(
        "Upload one or more images",
        type=["png", "jpeg", "jpg"],
        accept_multiple_files=True,
    )

    image_dict = {}
    if len(list_of_img_bytes) > 0:
        for img_bytes in list_of_img_bytes:
            file_bytes = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
            image_dict[img_bytes.name] = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        preprocess_and_render_layout(image_dict)


def preprocess_and_render_layout(image_dict):
    if len(image_dict) > 0:
        # inference
        outputs = infer_image(image_dict)

        # visualize
        col1, col2, col3 = st.columns(3)
        col1.header("Original image")
        col2.header("Image with prediction")
        col3.header("Stomata prediction metrics")
        for (file_name, orig_img), output in zip(image_dict.items(), outputs):
            col1, col2, col3 = st.columns(3)
            predictions = output[0]["objects"]
            predicted_image = output[0]["vis"]
            predicted_image = np.array(
                Image.open(io.BytesIO(base64.b64decode(predicted_image.split(",")[1])))
            )

            measurements = []  # measurement = (id, long_axis, short_axis, ratio, area)
            # post process
            for idx, pred in enumerate(predictions):
                bbox = pred["bounding_box"]
                category = pred["category"]
                rle = pred["rle"]
                score = pred["score"]

                rle = {
                    "counts": rle.split(","),
                    "size": [bbox["height"], bbox["width"]],
                }

                # mask = rle2mask(mask_rle=rle, shape=orig_img.shape[:2])
                compressed_rle = cocomask.frPyObjects(
                    rle, rle.get("size")[0], rle.get("size")[1]
                )
                mask = cocomask.decode(compressed_rle)
                polygons = binary_mask_to_polygon(mask)
                fitted_rbbox = fit_polygons_to_rotated_bboxes(polygons)
                if len(fitted_rbbox) < 1:
                    continue
                fitted_rbbox = fitted_rbbox[0]
                area = calc_polygon_area(polygons)
                if len(area) < 1:
                    continue
                area = area[0]

                # Get long and short axis
                if fitted_rbbox[1][0] > fitted_rbbox[1][1]:
                    long_axis, short_axis = fitted_rbbox[1][0], fitted_rbbox[1][1]
                else:
                    long_axis, short_axis = fitted_rbbox[1][1], fitted_rbbox[1][0]

                ratio = short_axis / long_axis
                measurements.append(
                    (
                        predicted_image.shape[0],
                        predicted_image.shape[1],
                        long_axis,
                        short_axis,
                        ratio,
                        area,
                    )
                )

                rb = (
                    (
                        bbox["left"] + fitted_rbbox[0][0],
                        bbox["top"] + fitted_rbbox[0][1],
                    ),
                    (fitted_rbbox[1][0], fitted_rbbox[1][1]),
                    fitted_rbbox[2],
                )
                c = random_color()
                box = cv2.boxPoints(rb)
                box = np.intp(box)
                predicted_image = cv2.drawContours(
                    predicted_image, [box], 0, color=c, thickness=2
                )
                t_size = cv2.getTextSize(f"idx:{idx}", 0, fontScale=0.5, thickness=1)[0]
                pt = np.amax(box, axis=0)
                c1 = (pt[0], pt[1])
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(
                    predicted_image, c1, c2, color=c, thickness=-1, lineType=cv2.LINE_AA
                )  # filled
                cv2.putText(
                    predicted_image,
                    f"idx:{idx}",
                    (c1[0], c1[1] - 2),
                    0,
                    0.5,
                    [255, 255, 255],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            df = pd.DataFrame(
                measurements,
                columns=[
                    "img_height",
                    "img_width",
                    "long_axis",
                    "short_axis",
                    "ratio",
                    "area",
                ],
            )

            with col1:
                st.image(orig_img, channels="BGR", caption=file_name)
            with col2:
                st.image(predicted_image, caption=file_name)
            with col3:
                st.dataframe(df)


def video_input():
    vid_file = None
    vid_bytes = st.sidebar.file_uploader("Upload a video", type=["mp4", "mpv", "avi"])
    if vid_bytes:
        vid_file = "./input." + vid_bytes.name.split(".")[-1]
        with open(vid_file, "wb") as out:
            out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        cap_fps = cap.get(cv2.CAP_PROP_FPS)

        st.markdown("---")
        vid = st.empty()
        frame_num = 0
        image_dict = {}
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_dict[f"frame{frame_num}"] = frame
            frame_num += 1
        cap.release()

        outputs = infer_image(image_dict, vid)
        vis_vid_file = "./vis." + vid_bytes.name.split(".")[-1]
        vid_writer = cv2.VideoWriter(
            vis_vid_file,
            cv2.VideoWriter_fourcc(*"MJPG"),
            cap_fps,
            (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )
        for o in outputs:
            predicted_image = o[0]["vis"]
            predicted_image = np.array(
                Image.open(io.BytesIO(base64.b64decode(predicted_image.split(",")[1])))
            )
            vid_writer.write(predicted_image)

        with open(vid_file, "rb") as v:
            vid_bytes = v.read()
            vid.video(vid_bytes)

        os.remove(vid_file)


def infer_image(images_dict: dict, process_field=None):
    operations = []
    for idx, img in images_dict.items():
        if process_field is not None:
            process_field.text(
                f"operation started for video frame: {idx}/{len(images_dict)}"
            )
        i = Struct()
        i.update({"input": cv2_base64(img)})
        operation = stomata_pipeline.trigger_async([i])
        operations.append(operation)

    responses = []
    for idx, op in enumerate(operations):
        if process_field is not None:
            process_field.text(
                f"operation done for video frame: {idx}/{len(operations)}"
            )
        while not stomata_pipeline.get_operation(op).done:
            time.sleep(0.1)
        latest_op = stomata_pipeline.get_operation(op)
        response_dict = MessageToDict(latest_op.response)
        if len(response_dict) > 0:
            responses.append(response_dict["outputs"])

    return responses

global_config.set_default(
    url="localhost:8080",
    token=st.secrets["instill_api_key"],
    secure=False,
)

st.set_page_config(layout="wide")

st.title("Stomata detection with YoloV7")
st.sidebar.title("Settings")

lang = st.toggle("Toggle for Chinese")
if not lang:
    st.markdown(
        Path("markdowns/readme_EN.md").read_text("utf-8"), unsafe_allow_html=True
    )
else:
    st.markdown(
        Path("markdowns/readme_CH.md").read_text("utf-8"), unsafe_allow_html=True
    )

client = InstillClient()
pipeline_service = client.pipeline_service
stomata_pipeline = Pipeline(client=client, name=st.secrets["stomata_pipeline_name"])

# input options
input_option = st.sidebar.radio(
    "Select input type: ", ["sample image", "self-upload image"]
)

if input_option == "self-upload image":
    image_input()
elif input_option == "sample image":
    sample_input()
# else:
# video_input()
