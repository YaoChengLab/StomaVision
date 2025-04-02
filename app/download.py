import zipfile
import streamlit as st
import cv2
from io import BytesIO


@st.fragment()
def download_modal(df, filenames, images):
    st.download_button(
        label="Download Combined CSV",
        data=df.to_csv().encode("utf-8"),
        file_name="all.csv",
        mime="text/csv",
        on_click="ignore",
    )

    buf = BytesIO()
    with zipfile.ZipFile(buf, "x") as z:
        for filename, image in zip(filenames, images):
            fn, ext = filename.rsplit(".", 1)
            img_bytes = cv2.imencode(f".{ext}", image)[1].tobytes()
            z.writestr(f"{fn}_predicted.{ext}", img_bytes)

    st.download_button(
        label="Download Predicted Images ZIP",
        data=buf.getvalue(),
        file_name="predicted_images.zip",
        mime="application/zip",
    )
