# DarkNetV3

The DarkNetV3 model is based on the ["YOLOv3: An Incremental Improvement"](https://pjreddie.com/media/files/papers/YOLOv3.pdf) paper.

## Architecture overview

This paper makes a more powerful version than its predecedors by increasing depth and using ResNet tricks.

The key takeaways from the paper are the following:

- adds residual connection compared to DarkNetV2


## Model builders

The following model builders can be used to instantiate a DarknetV3 model, with or
without pre-trained weights. All the model builders internally rely on the
[`DarknetV3`][holocron.models.DarknetV3] base class.

::: holocron.models.classification
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - DarknetV3
            - darknet53
