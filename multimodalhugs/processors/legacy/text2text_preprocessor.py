import logging
from typing import Any, Optional

from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor, ProcessorSlot
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor

logger = logging.getLogger(__name__)


class Text2TextTranslationProcessor(MultimodalMetaProcessor):
    name = "text2text_processor"
    attributes = ["tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        **kwargs,
    ):
        # Pass-through for from_pretrained, which calls cls(encoder_slots=..., ...)
        if "encoder_slots" in kwargs:
            super().__init__(tokenizer=tokenizer, **kwargs)
            return
        super().__init__(
            encoder_slots=[
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role="encoder"),
                    output_data_key="input_ids",
                    output_mask_key="attention_mask",
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
