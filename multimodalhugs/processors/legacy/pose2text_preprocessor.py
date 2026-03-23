import logging
from typing import Any, Optional

from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor, ProcessorSlot
from multimodalhugs.processors.pose_modality_processor import PoseModalityProcessor
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor

logger = logging.getLogger(__name__)


class Pose2TextTranslationProcessor(MultimodalMetaProcessor):
    name = "pose2text_processor"
    attributes = ["tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        reduce_holistic_poses: bool = True,
        skip_frames_stride: Optional[int] = None,
        **kwargs,
    ):
        # Pass-through for from_pretrained, which calls cls(slots=..., tokenizer=...)
        if "slots" in kwargs:
            super().__init__(tokenizer=tokenizer, **kwargs)
            return
        self.reduce_holistic_poses = reduce_holistic_poses
        self.skip_frames_stride = skip_frames_stride
        super().__init__(
            slots=[
                ProcessorSlot(
                    processor=PoseModalityProcessor(
                        reduce_holistic_poses=reduce_holistic_poses,
                        skip_frames_stride=skip_frames_stride,
                    ),
                    output_data_key="input_frames",
                    output_mask_key="attention_mask",
                    column_map={
                        "signal": "signal",
                        "signal_start": "signal_start",
                        "signal_end": "signal_end",
                    },
                ),
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role="label"),
                    output_data_key="labels",
                    is_label=True,
                    column_map={"decoder_prompt": "decoder_prompt", "output": "output"},
                ),
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role="encoder"),
                    output_data_key="encoder_prompt",
                    output_mask_key="encoder_prompt_length_padding_mask",
                    column_map={"encoder_prompt": "signal"},
                ),
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role="prompt"),
                    output_data_key="decoder_input_ids",
                    output_mask_key="decoder_attention_mask",
                    column_map={"decoder_prompt": "signal"},
                ),
            ],
            tokenizer=tokenizer,
        )
