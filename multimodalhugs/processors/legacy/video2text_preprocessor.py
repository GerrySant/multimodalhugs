import logging
from typing import Any, Optional

from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor, ProcessorSlot
from multimodalhugs.processors.video_modality_processor import VideoModalityProcessor
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor, TextRole

logger = logging.getLogger(__name__)


class Video2TextTranslationProcessor(MultimodalMetaProcessor):
    name = "video2text_processor"
    model_input_names = ["input_frames", "attention_mask"]

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        custom_preprocessor_path: Optional[str] = None,
        skip_frames_stride: Optional[int] = None,
        join_chw: bool = False,
        use_cache: bool = True,
        **kwargs,
    ):
        if "slots" in kwargs:
            super().__init__(tokenizer=tokenizer, **kwargs)
            return
        self.custom_preprocessor_path = custom_preprocessor_path
        self.skip_frames_stride = skip_frames_stride
        self.join_chw = join_chw
        self.use_cache = use_cache
        super().__init__(
            slots=[
                ProcessorSlot(
                    processor=VideoModalityProcessor(
                        custom_preprocessor_path=custom_preprocessor_path,
                        skip_frames_stride=skip_frames_stride,
                        join_chw=join_chw,
                        use_cache=use_cache,
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
                    processor=TextModalityProcessor(tokenizer=tokenizer, role=TextRole.TARGET),
                    output_data_key="labels",
                    is_label=True,
                    column_map={"decoder_prompt": "target_prefix", "output": "target"},
                ),
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role=TextRole.INPUT),
                    output_data_key="encoder_prompt",
                    output_mask_key="encoder_prompt_length_padding_mask",
                    column_map={"encoder_prompt": "signal"},
                ),
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role=TextRole.INPUT),
                    output_data_key="decoder_input_ids",
                    output_mask_key="decoder_attention_mask",
                    column_map={"decoder_prompt": "signal"},
                ),
            ],
            tokenizer=tokenizer,
        )
