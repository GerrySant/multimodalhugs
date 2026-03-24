import logging
from typing import Any, Optional

from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor, ProcessorSlot
from multimodalhugs.processors.features_modality_processor import FeaturesModalityProcessor
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor

logger = logging.getLogger(__name__)


class Features2TextTranslationProcessor(MultimodalMetaProcessor):
    name = "features2text_processor"
    model_input_names = ["input_frames", "attention_mask"]

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        use_cache: bool = True,
        skip_frames_stride: Optional[int] = None,
        temporal_dimention_position: int = 0,
        **kwargs,
    ):
        if "slots" in kwargs:
            super().__init__(tokenizer=tokenizer, **kwargs)
            return
        self.use_cache = use_cache
        self.skip_frames_stride = skip_frames_stride
        self.temporal_dimention_position = temporal_dimention_position
        super().__init__(
            slots=[
                ProcessorSlot(
                    processor=FeaturesModalityProcessor(
                        skip_frames_stride=skip_frames_stride,
                        temporal_dimention_position=temporal_dimention_position,
                        use_cache=use_cache,
                    ),
                    output_data_key="input_frames",
                    output_mask_key="attention_mask",
                ),
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role="label"),
                    output_data_key="labels",
                    is_label=True,
                    column_map={"decoder_prompt": "target_prefix", "output": "target"},
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
