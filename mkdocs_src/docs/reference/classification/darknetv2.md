# DarkNetV2

The DarkNetV2 model is based on the ["YOLO9000: Better, Faster, Stronger"](https://pjreddie.com/media/files/papers/YOLO9000.pdf) paper.

## Architecture overview

This paper improves its version version by adding more recent gradient flow facilitators.

The key takeaways from the paper are the following:

- adds batch normalization layers compared to DarkNetV1


## Model builders

The following model builders can be used to instantiate a DarknetV2 model, with or
without pre-trained weights. All the model builders internally rely on the
`holocron.models.classification.darknetv2.DarknetV2` base class. Please refer to the [source
code](https://github.com/frgfm/Holocron/blob/main/holocron/models/classification/darknetv2.py) for
more details about this class.

::: holocron.models.classification.darknetv2
    options:
        heading_level: 3
        show_root_heading: false
        members:
            - darknet19
