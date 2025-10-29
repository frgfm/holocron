# DarkNet

The DarkNet model is based on the ["You Only Look Once: Unified, Real-Time Object Detection"](https://pjreddie.com/media/files/papers/yolo_1.pdf) paper.

## Architecture overview

This paper introduces a highway network with powerful feature representation abilities.

The key takeaways from the paper are the following:

- improves the Inception architecture by using conv1x1
- replaces ReLU by LeakyReLU


## Model builders

The following model builders can be used to instantiate a DarknetV1 model, with or
without pre-trained weights. All the model builders internally rely on the
[`DarknetV1`][holocron.models.DarknetV1] base class.

::: holocron.models.classification
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - DarknetV1
            - darknet24
