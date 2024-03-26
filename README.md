StomaVision: stomatal trait analysis through deep learning
==============

# About StomaVision

For an efficient analysis of stomatal traits—like number, size, and closure rate—which are crucial for understanding plant responses to the environment, we introduce StomaVision. This automated tool simplifies stomatal detection and measurement on images from various plant species, including field samples. It enhances research efficiency by processing images quickly, even from videos, increasing the data available for robust analysis.

This tool is developed and maintained by the [Bioinformatics and Computational Genomics Group](https://www.southerngenomics.org/) at [Academia Sinica](https://www.sinica.edu.tw/), in collaboration with engineers from [Instill AI](https://www.instill.tech/).

To ensure maximum portability and ease of use, all necessary software packages and modules have been encapsulated within Docker containers.

## How to cite this tool
Ting-Li Wu, Po-Yu Chen, Xiaofei Du, Heiru Wu, Jheng-Yang Ou, Po-Xing Zheng, Yu-Lin Wu, Ruei-Shiuan Wang, Te-Chang Hsu, Chen-Yu Lin, Wei-Yang Lin,  Ping-Lin Chang, Chin-Min Kimmy Ho, Yao-Cheng Lin. StomaVision: stomatal trait analysis through deep learning


# Table of Contents

- [Users' guide](#usersguide)
  - [Online StomaVision](#web)
  - [Deploy StomaVision locally](#local)
    - [Launch local instance of Instill Core](#instillcore)
    - [Launch local instance of StomaVision streamlit app](#streamlit)
  - [Train your own model](#train)
- [Citation](#publication)

<a id="usersguide"></a>
# Users' guide
<a id="web"></a>
## For the impatient users - [**online app**](https://stomavision.streamlit.app)

We have set up a public Streamlit app powered by [Instill AI](https://www.instill.tech/) for you to try out, you can access it [**here**](https://stomavision.streamlit.app)

This public online version of [StomaVision](https://stomavision.streamlit.app) is hosted on  [Streamlit](https://streamlit.io/) and powered by [Instill Cloud](https://www.instill.tech/). If the instance remains unused for an extended period, it may be shut down. Please note that relaunching the instance could take approximately 5 minutes.

**NOTE**: The recommended image size is a long axis of less than **2000 pixels** (e.g., 1920x1080 or 1280x960 pixels). A magnification of **20X** is recommended for the objective lens. Other magnifications may affect prediction accuracy. Multiple photos can be uploaded at once.

<a id="local"></a>
## Deploy StomaVision locally

StomaVision was developed based on YOLOv7 in combination with the [Instill Core](https://www.instill.tech/) and [Streamlit](https://streamlit.io/). The whole environment is packaged in two main containers. To setup your local instance of StomaVision app, follow the steps below

<a id="instillcore"></a>
### Launch local instance of [**Instill Core**](https://github.com/instill-ai/instill-core)

First you will need a local instance of `Instill Core` to host and serve the `StomaVision` model and utilize the versatile AI pipeline, refer to [Local Instill Core guide](SERVE.md) to set it up

After you have `Instill Core` running locally, you can go to [https://localhost:3000](https://localhost:3000)s pipeline builder page to play around with the new created pipeline, you can learn more about it [here](https://www.instill.tech/docs/v0.25.0-beta/vdp/build)

<a id="streamlit"></a>
### Launch local instance of StomaVision streamlit app

Now we can launch the local instance of StomaVision streamlit app that connects to your local `Instill Core` instance, refer to [Local Stremalit guide](DEPLOY.md) to set it up

<a id="train"></a>
## Train your own version of Stomata Detection Model
You can also fine-tune and serve your own stomata detection model, please refer to [TRAIN guide](TRAIN.md)

<a id="publication"></a>
# Citation
Ting-Li Wu, Po-Yu Chen, Xiaofei Du, Heiru Wu, Jheng-Yang Ou, Po-Xing Zheng, Yu-Lin Wu, Ruei-Shiuan Wang, Te-Chang Hsu, Chen-Yu Lin, Wei-Yang Lin,  Ping-Lin Chang, Chin-Min Kimmy Ho, Yao-Cheng Lin. StomaVision: stomatal trait analysis through deep learning
