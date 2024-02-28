## StomaVision User Guide

- **Data Type Selection**: First, decide whether to use videos or photos.
- **Photos**: The recommended size is a long axis of less than 2000 pixels (e.g., 1920x1080 or 1280x960 pixels). A magnification of 20X is recommended for the objective lens. Other magnifications may affect prediction accuracy. Multiple photos can be uploaded at once.
- **Videos**: The recommended size is 1080p (HD): 1920x1080 pixels. The frame rate (FPS) should be between 15-30 FPS, depending on the speed of microscope field movement and focusing. Considering network speed, it is recommended to upload videos of 30-60 seconds in length. (Yet to be implemented)
- **Output Results**:
  - **Images**:
    - The original image (left side) and the prediction result (right side) will be displayed separately, facilitating users to judge the accuracy of the model's predictions.
    - In the right-side image, boxes indicate the position of stomata, with labels (e.g., “idx:123”) corresponding to numbers in the table. This can be used to filter out undesirable data.
    - Each predicted stomata is surrounded by a box, used for extracting the long and short axes of the stoma, corresponding to the “long_axis” and “short_axis” fields in the table.
    - In addition to the box, a polygonal color block indicates the predicted stomatal pore, with the area of the block (the “area” field) representing the size of the stomata.
    - A two-way arrow button on the image can be used to zoom in for closer inspection.
  - **Table Explanation**:
    - There are seven fields in total:
      - "ID": Corresponds to the stomata number in the image, used for tracing and counting.
      - "img_height" and "img_width": The dimensions of the image in pixels, used to calculate the number and proportion of stomata opening.
      - "long_axis" and "short_axis": The pixel values of the long and short axes of the stomata, which need to be converted to actual lengths based on the scale. Conversion example: If the image scale is 50 μm = 10 pixels, then 57 pixels = (50/10) x 57 = 285 μm.
      - "ratio": Calculated from “short_axis” / “long_axis”, used to assess the degree of stomatal opening.
      - "area": The area represented by the polygonal color block, in pixels, which needs to be converted to actual size.
    - Three buttons in the upper right corner of the table: Download results, search values, and zoom function. (Yet to be implemented)