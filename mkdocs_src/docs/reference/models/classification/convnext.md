# ConvNeXt

The ConvNeXt model is based on the ["A ConvNet for the 2020s"](https://arxiv.org/pdf/2201.03545.pdf) paper.

## Architecture overview

This architecture compiles tricks from transformer-based vision models to improve a pure convolutional model.

![ConvNeXt architecture](https://github.com/frgfm/Holocron/releases/download/v0.2.1/convnext.png)

The key takeaways from the paper are the following:

- update the stem convolution to act like a patchify layer of transformers
- increase block kernel size to 7
- switch to depth-wise convolutions
- reduce the amount of activations and normalization layers


## Model builders

The following model builders can be used to instantiate a ConvNeXt model, with or
without pre-trained weights. All the model builders internally rely on the [`ConvNeXt`][holocron.models.ConvNeXt] base class.

::: holocron.models.classification
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - ConvNeXt
            - convnext_atto
            - convnext_femto
            - convnext_pico
            - convnext_nano
            - convnext_tiny
            - convnext_small
            - convnext_base
            - convnext_large
            - convnext_xl
