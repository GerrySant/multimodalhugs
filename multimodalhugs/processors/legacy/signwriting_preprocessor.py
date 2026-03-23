import logging
from typing import Any, Optional

from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor, ProcessorSlot
from multimodalhugs.processors.signwriting_modality_processor import SignwritingModalityProcessor
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor

logger = logging.getLogger(__name__)


class SignwritingProcessor(MultimodalMetaProcessor):
    name = "signwritting_processor"
    attributes = ["tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        custom_preprocessor_path: Optional[str] = None,
        width: int = 224,
        height: int = 224,
        channels: int = 3,
        invert_frame: bool = False,
        **kwargs,
    ):
        if "slots" in kwargs:
            super().__init__(tokenizer=tokenizer, **kwargs)
            return
        self.custom_preprocessor_path = custom_preprocessor_path
        self.width = width
        self.height = height
        self.channels = channels
        self.invert_frame = invert_frame
        super().__init__(
            slots=[
                ProcessorSlot(
                    processor=SignwritingModalityProcessor(
                        custom_preprocessor_path=custom_preprocessor_path,
                        width=width,
                        height=height,
                        channels=channels,
                        invert_frame=invert_frame,
                    ),
                    output_data_key="input_frames",
                    output_mask_key="attention_mask",
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
