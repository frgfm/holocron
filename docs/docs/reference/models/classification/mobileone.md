# MobileOne

The ResNet model is based on the ["An Improved One millisecond Mobile Backbone"](https://arxiv.org/pdf/2206.04040.pdf) paper.

## Architecture overview

This architecture optimizes the model for inference speed at inference time on mobile device.

![MobileOne architecture](https://github.com/frgfm/Holocron/releases/download/v0.2.1/mobileone.png)

The key takeaways from the paper are the following:

- reuse the reparametrization concept of RepVGG while adding overparametrization in the block branches.
- each block is composed of two consecutive reparametrizeable blocks (in a similar fashion than RepVGG): a depth-wise convolutional block, a point-wise convolutional block.


## Model builders

The following model builders can be used to instantiate a MobileOne model, with or
without pre-trained weights. All the model builders internally rely on the
[`MobileOne`][holocron.models.MobileOne] base class.

::: holocron.models.classification
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - MobileOne
            - mobileone_s0
            - mobileone_s1
            - mobileone_s2
            - mobileone_s3
