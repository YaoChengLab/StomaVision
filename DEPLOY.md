## Deploy an streamlit app for quick inference

To quickly deploy a streamlit app that utilize the `Instill VDP` pipeline, we provide a `dockerfile`. If you haven't have a `Instill Core` instance running, please refer to [this guide](SERVE.md) to set it up

1. copy the API_TOKEN you've created on the console's settings page
2. execute the command `docker build --build-arg API_TOKEN={your_api_token} -t {your_desired_app_name} .`
3. execute the command `docker run -d --network instill-network -p 8501:8501 {your_desired_app_name}`
> [!NOTE]  
> **Streamlit APP Port**  
> If you want to change the default port, you can directly edit the `dockerfile`
4. now you can go to `localhost:8501` and start using the app
