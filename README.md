# StomaVision: a stomata detection and measurement based on YOLOv7

Welcome to the StomaVision

StomaVision is a stomata detection and measurement based on YOLOv7.

All required software packages and modules were packed in Docker containers to make it portable.

## For the impatient users

We have set up a public Streamlit app powered by [`Instill AI`](https://www.instill.tech/) for you to try out, you can access it here

### [**StomaVision**](https://stomavision.streamlit.app/)

## Deploy StomaVision locally

To setup your local instance of StomaVision app, follow the steps below

### Launch local instance of [**Instill Core**](https://github.com/instill-ai/instill-core)

First you will need a local instance of `Instill Core` to host and serve the `StomaVision` model and utilize the versatile AI pipeline, refer to [SERVE guide](SERVE.md) to set it up

After you have `Instill Core` running locally, you can go to `localhost:3000`'s pipeline builder page to play around with the new created pipeline, you can learn more about it [here](https://www.instill.tech/docs/v0.25.0-beta/vdp/build)

### Launch local instance of StomaVision app

Now we can launch the local instance of StomaVision streamlit app that connects to your local `Instill Core` instance, refer to [DEPLOY guide](DEPLOY.md) to set it up

### Train your own version of Stomata Detection Model
You can also fine-tune and serve your own stomata detection model, please refer to [TRAIN guide](TRAIN.md)
