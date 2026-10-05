# holocron.ops

`holocron.ops` implements operators that are specific for Computer Vision.

!!! note
    Those operators currently do not support TorchScript.

## Boxes

![Four-panel comparison of IoU overlap, GIoU enclosing area, DIoU center distance, and CIoU aspect-ratio geometry using the same target and prediction boxes.](../img/box-iou-family.svg)

::: holocron.ops
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - box_giou
            - diou_loss
            - ciou_loss
