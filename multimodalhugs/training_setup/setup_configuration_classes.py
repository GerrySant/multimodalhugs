from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class SetupArguments:
    modality: Literal["pose2text","signwriting2text","image2text","text2text","features2text","video2text"] = field(
        metadata={"help": "Training setup modality."}
    )
    config_path: str = field(
        metadata={"help": "Path to YAML configuration file"}
    )
    do_dataset: bool = field(
        default=False, metadata={"help": "Prepare the dataset."}
        )
    do_processor: bool = field(
        default=False, metadata={"help": "Set up the processor."}
        )
    do_model: bool = field(
        default=False, metadata={"help": "Build the model."}
        )
    output_dir: Optional[str] = field(
        default=None, metadata={"help": "Base output directory."}
        )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    update_config: Optional[bool] = field(
        default=None,
        metadata={"help": "Write created artifact paths back into the config file."}
    )