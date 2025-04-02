# pylint: disable=no-member,wrong-import-position,no-name-in-module
import io
import time
import base64
import cv2
import pandas as pd
import streamlit as st
import numpy as np
import requests
import json
from tempfile import NamedTemporaryFile
from pathlib import Path
from PIL import Image
from streamlit_image_select import image_select
from pycocotools import mask as cocomask
from run_pipeline import run_stomata_detection_async, get_operation
from utils import (
    binary_mask_to_polygon,
    fit_polygons_to_rotated_bboxes,
    calc_polygon_area,
    random_color,
)
from download import download_modal

# Set Instill Cloud API token from Streamlit secrets
instill_api_token = (
    st.secrets["INSTILL_API_TOKEN"] if "INSTILL_API_TOKEN" in st.secrets else None
)
instill_org_uid = None


def cv2_base64(image):
    buffer_img = cv2.imencode(".jpg", image)[1]
    base64_str = base64.b64encode(buffer_img).decode("utf-8")
    return base64_str


def sample_input():
    img = image_select(
        label="Select a sample image to inference",
        use_container_width=False,
        images=[
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


def preprocess_and_render_layout(image_dict, process_field=None):
    if len(image_dict) > 0:
        # inference
        outputs = batch_infer_image(image_dict, process_field)

        color_1 = random_color()
        color_2 = random_color()

        # visualize
        if process_field is None:
            col1, col2, col3 = st.columns(3)
            col1.header("Original image")
            col2.header("Image with prediction")
            col3.header("Stomata prediction metrics")
            dfs = []
            p_images = []
            error_fns = []
        else:
            process_field.text("Average measurements per frame")
            t_measurements = []
        for file_name, output in outputs.items():
            orig_img = image_dict[file_name]
            if process_field is None:
                col1, col2, col3 = st.columns(3)
            # measurement = (id, long_axis, short_axis, ratio, area)
            measurements = []
            error = None
            if "error" in output[0]:
                error = output[0]
                error_fns.append(file_name)
            else:
                predictions = output[0]["objects"]
                vis_output: str = output[0]["vis"]
                if vis_output.startswith("https") or vis_output.startswith("http"):
                    resp = requests.get(vis_output, timeout=180)
                    resp.raise_for_status()
                    predicted_image = np.array(Image.open(io.BytesIO(resp.content)))
                else:
                    predicted_image = np.array(
                        Image.open(
                            io.BytesIO(base64.b64decode(vis_output.split(",")[1]))
                        )
                    )

                # post process
                for idx, pred in enumerate(predictions):
                    bbox = pred["bounding-box"]
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
                            category,
                            long_axis,
                            short_axis,
                            ratio,
                            area,
                        )
                    )

                    if process_field is None:
                        rb = (
                            (
                                bbox["left"] + fitted_rbbox[0][0],
                                bbox["top"] + fitted_rbbox[0][1],
                            ),
                            (fitted_rbbox[1][0], fitted_rbbox[1][1]),
                            fitted_rbbox[2],
                        )
                        if category == "stomata":
                            c = color_1
                        else:
                            c = color_2
                        box = cv2.boxPoints(rb)
                        box = np.intp(box)
                        predicted_image = cv2.drawContours(
                            predicted_image, [box], 0, color=c, thickness=2
                        )
                        t_size = cv2.getTextSize(
                            f"idx:{idx}", 0, fontScale=0.5, thickness=1
                        )[0]
                        pt = np.amax(box, axis=0)
                        c1 = (pt[0], pt[1])
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                        cv2.rectangle(
                            predicted_image,
                            c1,
                            c2,
                            color=c,
                            thickness=-1,
                            lineType=cv2.LINE_AA,
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
                    "category",
                    "long_axis",
                    "short_axis",
                    "ratio",
                    "area",
                ],
            )

            if process_field is None:
                if error is None:
                    dfs.append(df)
                    p_images.append(predicted_image)
                    fn, ext = file_name.rsplit(".", 1)
                with col1:
                    st.image(orig_img, channels="BGR", caption=file_name)
                with col2:
                    if error is None:
                        st.image(predicted_image, channels="RGB", caption=file_name)
                        img_bytes = cv2.imencode(f".{ext}", predicted_image)[
                            1
                        ].tobytes()
                        st.download_button(
                            label="Download Predicted Image",
                            data=img_bytes,
                            file_name=f"{fn}_predicted.{ext}",
                            mime="image/png",
                            on_click="ignore",
                        )
                    else:
                        st.text_area(
                            "Error Component in Instill Pipeline:", error["component"]
                        )
                with col3:
                    if error is None:
                        st.dataframe(df)
                        st.download_button(
                            label="Download CSV",
                            data=df.to_csv().encode("utf-8"),
                            file_name=f"{fn}.csv",
                            mime="text/csv",
                            on_click="ignore",
                        )
                    else:
                        st.text_area("Error Message:", error["error"])
            else:
                long_axis_list = [t[2] for t in measurements]
                short_axis_list = [t[3] for t in measurements]
                ratio_list = [t[4] for t in measurements]
                area_axis_list = [t[5] for t in measurements]
                avg_long_axis = sum(long_axis_list) / len(long_axis_list)
                avg_short_axis = sum(short_axis_list) / len(short_axis_list)
                avg_ratio = sum(ratio_list) / len(ratio_list)
                avg_area = sum(area_axis_list) / len(area_axis_list)
                t_measurements.append(
                    (
                        measurements[0][0],
                        measurements[0][1],
                        avg_long_axis,
                        avg_short_axis,
                        avg_ratio,
                        avg_area,
                        len(measurements),
                    )
                )

        if process_field is not None:
            df = pd.DataFrame(
                t_measurements,
                columns=[
                    "img_height",
                    "img_width",
                    "avg_long_axis",
                    "avg_short_axis",
                    "avg_ratio",
                    "avg_area",
                    "total_stomata_in_frame",
                ],
            )
            st.dataframe(df)
        else:
            with st.sidebar:
                keys = [k for k in outputs.keys() if k not in error_fns]
                res = pd.concat(dfs, keys=keys)
                res.index.names = ["filename", "index"]
                download_modal(res, outputs.keys(), p_images)


def video_input():
    vid_bytes = st.sidebar.file_uploader("Upload a video", type=["mp4", "mpv", "avi"])
    if vid_bytes:
        with NamedTemporaryFile(
            suffix=f".{vid_bytes.name.split('.')[-1]}"
        ) as input_video:
            input_video.write(vid_bytes.read())

            if input_video:
                cap = cv2.VideoCapture(input_video.name)
                # cap_fps = cap.get(cv2.CAP_PROP_FPS)

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

                preprocess_and_render_layout(image_dict, vid)

                # TODO: if needed to support video playback
                # with NamedTemporaryFile(
                #     suffix=f".{vid_bytes.name.split('.')[-1]}"
                # ) as vis_video:
                #     vid_writer = cv2.VideoWriter(
                #         vis_video.name,
                #         cv2.VideoWriter_fourcc(*'MJPG'),
                #         cap_fps,
                #         (
                #             int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                #             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                #         ),
                #     )
                #     print(input_video.name)
                #     print(vis_video.name)
                #     print(vid_writer.isOpened())
                #     for _, output in outputs.items():
                #         predicted_image = output[0]["vis"]
                #         predicted_image = np.array(
                #             Image.open(
                #                 io.BytesIO(
                #                     base64.b64decode(predicted_image.split(",")[1])
                #                 )
                #             )
                #         )
                #         vid_writer.write(predicted_image)

                #     vid_writer.release()
                #     print(os.path.getsize(vis_video.name))
                #     vis_video.seek(0)

                #     vid_bytes = vis_video.read()
                #     vid.video(vid_bytes)


def batch_infer_image(images_dict: dict, process_field=None) -> dict:
    operations = {}
    for idx, img in images_dict.items():
        if process_field is not None:
            process_field.text(
                f"operation started for video frame: {idx}/{len(images_dict)}"
            )

        resp = run_stomata_detection_async(
            instill_api_token, cv2_base64(img), instill_org_uid
        )
        operations[idx] = resp["operation"]

    responses = {}
    for idx, op in operations.items():
        if process_field is not None:
            process_field.text(
                f"operation done for video frame: {idx}/{len(operations)}"
            )

        resp = get_operation(instill_api_token, op["name"], instill_org_uid)
        operation = resp["operation"]

        while operation is not None and not operation["done"]:
            time.sleep(0.5)
            resp = get_operation(instill_api_token, op["name"], instill_org_uid)
            operation = resp["operation"]

        if operation is not None:
            response_dict = operation["response"]
            error = None
            for component, trace in response_dict["metadata"]["traces"].items():
                if "STATUS_ERROR" in trace["statuses"]:
                    error = {
                        "component": component,
                        "error": trace["error"]["message"],
                    }
            # print([json.loads(t.replace("'", "\""))["error"] for t in response_dict["metadata"]["traces"]])
            if len(response_dict) > 0 and error is None:
                responses[idx] = response_dict["outputs"]
            else:
                responses[idx] = [error]

    return responses


st.set_page_config(layout="wide")

st.title("Stomata detection with YoloV7")
st.sidebar.title("Settings")

lang = st.toggle("Toggle for Chinese")
if not lang:
    st.markdown(
        Path("markdowns/readme_EN.md").read_text("utf-8"),
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        Path("markdowns/readme_CH.md").read_text("utf-8"),
        unsafe_allow_html=True,
    )

# input options
input_option = st.sidebar.radio(
    "Select input type: ",
    [
        "sample image",
        "self-upload image",
        "self-upload video",
    ],
)

if input_option == "self-upload image":
    image_input()
elif input_option == "sample image":
    sample_input()
elif input_option == "self-upload video":
    video_input()
