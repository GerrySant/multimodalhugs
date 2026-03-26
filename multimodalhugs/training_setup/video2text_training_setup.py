"""
Backward-compatible entry point for the video-to-text setup path.

Delegates entirely to general_training_setup.main with
default_dataset_type="video2text" so that configs that predate the
data.dataset_type field continue to work when --modality video2text is given.

For new projects, use multimodalhugs-setup without --modality and add
  data:
    dataset_type: video2text
to your config file instead.
"""
from typing import Optional
from multimodalhugs.training_setup.general_training_setup import main as _general_main


def main(
    config_path: str,
    do_dataset: bool,
    do_processor: bool,
    do_model: bool,
    output_dir: Optional[str] = None,
    update_config: Optional[bool] = None,
    rebuild_dataset_from_scratch: bool = False,
):
    _general_main(
        config_path=config_path,
        do_dataset=do_dataset,
        do_processor=do_processor,
        do_model=do_model,
        output_dir=output_dir,
        update_config=update_config,
        rebuild_dataset_from_scratch=rebuild_dataset_from_scratch,
        default_dataset_type="video2text",
    )
