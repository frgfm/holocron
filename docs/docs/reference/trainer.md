# holocron.trainer

`holocron.trainer` provides some basic objects for training purposes.

::: holocron.trainer.Trainer
    options:
        heading_level: 3
        show_object_full_path: false
        members:
            - set_device
            - to_cuda
            - save
            - load
            - fit_n_epochs
            - find_lr
            - plot_recorder
            - check_setup


## Image classification

::: holocron.trainer
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - ClassificationTrainer
            - BinaryClassificationTrainer

## Semantic segmentation

::: holocron.trainer
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - SegmentationTrainer

## Object detection

::: holocron.trainer
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - DetectionTrainer

## Miscellaneous

::: holocron.trainer
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - freeze_bn
            - freeze_model
