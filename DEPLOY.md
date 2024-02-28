## Deploy an streamlit app for quick inference

To quickly deploy a streamlit app that utilize the `Instill VDP` pipeline, we provide a `dockerfile`. If you haven't have a `Instill Core` instance running, please refer to [this guide](SERVE.md) to set it up

1. first we need to config the `config.toml` file located under the path `app/config.toml`, with the corresponding input and outputs key defined in the pipeline's recipe, for example
```toml
stomata_pipeline_name = "stomata"                   # the name of the pipeline you defined on console
stomata_pipeline_input_key = "input"                # the key of the image field of the pipeline's start operator
stomata_pipeline_output_objects_key = "objects"     # the key of the model objects output field of the pipeline's end operator
stomata_pipeline_output_visualization_key = "vis"   # the key of the image operator output field of the pipeline's end operator
```
2. copy the API_TOKEN you've created on the console's settings page
3. execute the command `docker build --build-arg API_TOKEN={your_api_token} -t {your_desired_app_name} .`
4. execute the command `docker run -d --network instill-network -p 8501:8501 {your_desired_app_name}`
> [!NOTE]  
> **Streamlit APP Port**  
> If you want to change the default port, you can directly edit the `dockerfile`
5. now you can go to `localhost:8501` and start using the app
