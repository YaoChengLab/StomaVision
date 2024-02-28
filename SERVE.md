## Serving StomaVision on [Instill Core](https://github.com/instill-ai/core)

### How to pack your own model weight to be served on [Instill Model](https://github.com/instill-ai/model)

First name your newly trained weight to `model.pt` and move it under the path `deploy/model.pt`

And zip the whole `/deploy` folder
```bash
cd deploy
zip -r "stomavision.zip" .
mv stomavision.zip ../
cd ..
```

Next clone and launch `Instill Core`, it will automatically deploy [Instill Model](https://github.com/instill-ai/model) and [Instill VDP](https://github.com/instill-ai/vdp) for you. You will need to have `docker` installed
```bash
git clone https://github.com/instill-ai/core.git
cd core
make all
```

Now we open a browser and
1. go to `localhost:3000` and login with default password: `password`
2. navigate to `Model Hub` page
3. click `Add Model` button
4. give the model a name in the `ID` section
5. select `Local` in the `Model source` drop down menu
6. select the `stomavision.zip` file we just zipped in the `Upload a file` section
7. click `Set up`
8. now we can wait for awhile and refresh the page until we see the model `Status` become `Online`, shouldn't take longer than 1 minute
9. now the model is served! To verify it, we first need to obtain an API_KEY
10. click the profile icon at the top right corner and click `Settings`
11. go to API Tokens tab
12. click `Create Token`
13. give it a name and click `Create Token`
14. click the copy icon to copy the whole token
15. now in your terminal, replace the `{your_model_name_here}` and `{your_token_here}` and send the request
```bash
curl --location 'http://localhost:8080/model/v1alpha/users/admin/models/{your_model_name_here}/trigger' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer {your_token_here}' \
--data '{
    "task_inputs": [
        {
            "instance_segmentation": {
                "image_url": "https://upload.wikimedia.org/wikipedia/en/d/de/Nail_polish_impression_of_stomata.jpg"
            }
        }
    ]
}'
```
you should get a response like this
```json
{
    "task": "TASK_INSTANCE_SEGMENTATION",
    "task_outputs": [
        {
            "instance_segmentation": {
                "objects": [
                    {
                        "rle": "679,6,32,8,30,10,28,12,26,14,25,15,24,15,24,15,24,15,24,15,25,13,27,11,28,10,31,7,33,4,330",
                        "category": "stomata",
                        "score": 0.81463146,
                        "bounding_box": {
                            "top": 216,
                            "left": 1766,
                            "width": 40,
                            "height": 39
                        }
                    },
                    ...
                    ...
                    ...
                ]
            }
        }
    ]
}
```
### How to visualize the inference result with [Instill VDP](https://github.com/instill-ai/vdp)
Now we have our model served on Instill Core, we can create a pipeline to easily visualize the inference result
1. go to the `Pipelines` page
2. click `Create Pipeline` button
3. give the pipeline a name and choose `Public` or `Private` base on your needs
4. first we click the `Add Field` button in the `start` component
5. select `Image`, give it a title and key and click save, they could be the same, you will reference the key later in other component,
6. we will come back to `end` component later, now click `Add Component` on the top left corner
7. select `Instill Model` under the `AI` category, this is to create a component that allow the pipeline to connect to the model we just served
8. click the `Create Connector` within the newly created `Instill Model` component
9. give it a name and make sure you select `Internal Mode`
10. now back in the canvas, select `TASK_INSTANCE_SEGMENTATION` in the newly created `Instill Model` component
11. you should find the model we've created earlier in the `Model Name` dropdown menu, select it
12. type `${` in the `Image` field, and you will be hinted with the `key` name for our image input from our start component
13. click `Add Component` again, and select `Image` from `Operators`
14. select `TASK_DRAW_INSTANCE_SEGMENTATION`
15. type `${` in the Objects field, and you will be hinted with the output objects from our `Instill Model` component
16. type `${` in the `Image` field, and you will be hinted with the `key` name for our image input from our start component
> [!IMPORTANT]  
> **End operator fields with streamlit app**  
> If you are going to deploy streamlit app with this pipeline, the end operator must have the following two fields, 1. a field that reference the `instill model` component `output.objects`, and 2. a field that reference the `image` operator `output.image` field. You can define the key name as you like. For more information please refer to [DEPLOY.md](DEPLOY.md)
17. now we move on to end component, click `Add Field` and give it a `Title` and `Key`, and type `${` in the `Value` field, you will be hinted with some valid options, select the `image` output from our `image` operator, it should be something like `{image_operator_name}.output.image`, and we click save
18. click `Add Field` and give it a `Title` and `Key`, and type `${` in the `Value` field, you will be hinted with some valid options, select the `objects` output from our `instill model` component, it should be something like `{instill_model_name}.output.objects`, and we click save
19. click `Save` on the top right bar
20. if you see all components are linked with solid blue lines, we are ready to trigger the pipeline!
21. click the `Upload image` in the `start` component and select an image with stomata, click `Run` on the top right bar
22. now in the `end` component, you will see the visualized output image
