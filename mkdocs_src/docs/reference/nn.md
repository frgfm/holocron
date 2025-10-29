# holocron.nn

An addition to the `torch.nn` module of Pytorch to extend the range of neural networks building blocks.

## Non-linear activations

::: holocron.nn
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - HardMish
            - NLReLU
            - FReLU

## Loss functions

::: holocron.nn
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - Loss
            - FocalLoss
            - MultiLabelCrossEntropy
            - ComplementCrossEntropy
            - MutualChannelLoss
            - DiceLoss
            - PolyLoss

## Loss wrappers

::: holocron.nn
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - ClassBalancedWrapper

## Convolution layers

::: holocron.nn
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - NormConv2d
            - Add2d
            - SlimConv2d
            - PyConv2d
            - Involution2d

## Regularization layers

::: holocron.nn
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - DropBlock2d

## Downsampling

::: holocron.nn
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - ConcatDownsample2d
            - GlobalAvgPool2d
            - GlobalMaxPool2d
            - BlurPool2d
            - SPP
            - ZPool

## Attention

::: holocron.nn
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - SAM
            - LambdaLayer
            - TripletAttention
