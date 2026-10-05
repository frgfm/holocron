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

Check each loss's input contract before using it. `N` is the batch size and `K`
is the number of classes; `...` represents optional spatial dimensions.

| Loss | Input | Target |
|---|---|---|
| `FocalLoss` | Raw logits, shape `(N, K, ...)` | Class indices, shape `(N, ...)`, dtype `torch.int64` |
| `PolyLoss` | Raw logits, shape `(N, K, ...)` | Class indices with dtype `torch.int64`, or soft class probabilities with the same shape as the logits |
| `DiceLoss` | Class probabilities, shape `(N, K, ...)` | One-hot or soft targets with the same shape as the input |

For hard targets, class indices must be in `[0, K)`, except for ignored values
supported by the selected loss. A segmentation target has shape `(N, H, W)`,
without a class channel. Casting a soft target to `torch.int64` does not turn it
into class indices.

`FocalLoss(ignore_index=-100)` supports ignored hard targets outside the class
range and excludes them from the loss and gradient. `PolyLoss` currently does
not: a hard target containing `-100` or `255` outside `[0, K)` raises an error,
even when it matches `ignore_index`. Its in-range ignore behavior and
`reduction="none"` limits are documented below.

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
