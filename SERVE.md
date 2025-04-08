## Serving StomaVision on [Instill Core](https://github.com/instill-ai/instill-core)

### How to pack your own model weight to be served on [Instill Core](https://github.com/instill-ai/instill-core)

You will need to have `docker` installedYou will need to have `docker` installed. You can read more about how model serving works on Instill Core [here](https://github.com/instill-ai/models?tab=readme-ov-file).

1. Put your `outerline model` and `pore model` under the path `deploy/` and rename them to be `outerlinebest.pt` and `porebest.pt` respectively, or you can download our pretrained model with

```bash
curl -o ./deploy/outerlinebest.pt https://artifacts.instill.tech/model/yolov7-stomata/outerlinebest.pt
curl -o ./deploy/porebest.pt https://artifacts.instill.tech/model/yolov7-stomata/porebest.pt
```

2. Install Instill SDK

```bash
pip install instill-sdk==0.16.2
```

3. Build model image
  - for system with NVIDIA GPU
```bash
cd deploy
instill build admin/stomavision
```
  - for CPU system
```bash
cd deploy
mv instill-cpu.yaml instill.yaml
mv model-cpu.py model.py
instill build admin/stomavision
```

4. Clone and launch `Instill Core`.

```bash
git clone -b v0.50.2-beta https://github.com/instill-ai/instill-core.git && cd instill-core
# Launch all services
make all
```

5. On your browser go to `localhost:3000` and login with default password: `password`
6. Go to `Settings` -> `API Tokens` and create a new API token
7. In terminal, `docker login`

```bash
docker login localhost:8080
Username: admin
Password: {your-api-token}
```

8. Back on the browser, go to `Model` page and create a new model with

- Model ID: stomavision
- AI Task: Instance Segmentation
- Hardware:
  - GPU if your host machine has NVIDIA GPU
  - CPU if your host machine doe not have NVIDIA GPU

9. In terminal, push the model onto instill-core

```bash
instill push admin/stomavision -u localhost:8080
```

10. After the model is pushed and the status is `online` on model page, go to pipeline page and create a pipeline with name `stomavision`
11. Copy and paste the recipe content from [here](utils/pipeline_recipe.yaml) into the editor
12. Now you can trigger the pipeline with an image to see the result!

Or you can use API to trigger the model like this

```bash
curl --location 'http://localhost:8080/v1alpha/namespaces/admin/models/stomavision/trigger' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer {your_token_here}' \
--data '{
    "task_inputs": [
        {
            "data": {
                "image-url": "https://microscopyofnature.com/sites/default/files/2022-03/Mais-stomata-ZW10.jpg",
                "type": "image-url"
            }
        }
    ]
}'
```
