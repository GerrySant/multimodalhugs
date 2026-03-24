import logging
from typing import Any, Optional

from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor, ProcessorSlot
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor, TextRole

logger = logging.getLogger(__name__)


class Text2TextTranslationProcessor(MultimodalMetaProcessor):
    name = "text2text_processor"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        **kwargs,
    ):
        if "slots" in kwargs:
            super().__init__(tokenizer=tokenizer, **kwargs)
            return
        super().__init__(
            slots=[
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role=TextRole.INPUT),
                    output_data_key="input_ids",
                    output_mask_key="attention_mask",
                    column_map={"signal": "signal"},
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
