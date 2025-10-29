# DarkNetV2

The DarkNetV2 model is based on the ["YOLO9000: Better, Faster, Stronger"](https://pjreddie.com/media/files/papers/YOLO9000.pdf) paper.

## Architecture overview

This paper improves its version version by adding more recent gradient flow facilitators.

The key takeaways from the paper are the following:

- adds batch normalization layers compared to DarkNetV1


## Model builders

The following model builders can be used to instantiate a DarknetV2 model, with or
without pre-trained weights. All the model builders internally rely on the
[`DarknetV2`][holocron.models.DarknetV2] base class.

::: holocron.models.classification
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - DarknetV2
            - darknet19
