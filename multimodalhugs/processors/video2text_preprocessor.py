import logging
from typing import Any, Optional

from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor, ProcessorSlot
from multimodalhugs.processors.video_modality_processor import VideoModalityProcessor
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor

logger = logging.getLogger(__name__)


class Video2TextTranslationProcessor(MultimodalMetaProcessor):
    name = "video2text_processor"
    attributes = ["tokenizer"]
    model_input_names = ["inputs_embeds", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        custom_preprocessor_path: Optional[str] = None,
        skip_frames_stride: Optional[int] = None,
        join_chw: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # Pass-through for from_pretrained, which calls cls(encoder_slots=..., ...)
        if "encoder_slots" in kwargs:
            super().__init__(tokenizer=tokenizer, **kwargs)
            return
        self.custom_preprocessor_path = custom_preprocessor_path
        self.skip_frames_stride = skip_frames_stride
        self.join_chw = join_chw
        self.use_cache = use_cache
        super().__init__(
            encoder_slots=[
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
                )
            ],
            label_slot=ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="label"),
                output_data_key="labels",
            ),
            encoder_prompt_slot=ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="encoder"),
                output_data_key="encoder_prompt",
                output_mask_key="encoder_prompt_length_padding_mask",
                column_map={"encoder_prompt": "signal"},
            ),
            decoder_prompt_slot=ProcessorSlot(
                processor=TextModalityProcessor(tokenizer=tokenizer, role="prompt"),
                output_data_key="decoder_input_ids",
                output_mask_key="decoder_attention_mask",
                column_map={"decoder_prompt": "signal"},
            ),
            tokenizer=tokenizer,
        )
