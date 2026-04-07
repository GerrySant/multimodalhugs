import logging
from typing import Any, List, Optional

from multimodalhugs.processors.meta_processor import _LegacyMetaProcessorBase, ProcessorSlot
from multimodalhugs.processors.image_modality_processor import ImageModalityProcessor
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor, TextRole

logger = logging.getLogger(__name__)


class Image2TextTranslationProcessor(_LegacyMetaProcessorBase):
    """
    .. deprecated::
        Use ``MultimodalMetaProcessor`` with explicit ``ProcessorSlot`` declarations
        instead.  This task-specific wrapper is kept for backward compatibility and
        will be removed in a future release.  See ``processors/meta_processor.py``
        and the processor configuration docs for the recommended approach.
    """
    name = "image2text_processor"
    model_input_names = ["input_frames", "attention_mask"]

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        font_path: Optional[str] = None,
        width: int = 224,
        height: int = 224,
        normalize_image: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        **kwargs,
    ):
        self.font_path = font_path
        self.width = width
        self.height = height
        self.normalize_image = normalize_image
        self.mean = mean
        self.std = std
        if "slots" in kwargs:
            super().__init__(**kwargs)
            return
        super().__init__(
            slots=[
                ProcessorSlot(
                    processor=ImageModalityProcessor(
                        font_path=font_path,
                        width=width,
                        height=height,
                        normalize_image=normalize_image,
                        mean=mean,
                        std=std,
                    ),
                    output_data_key="input_frames",
                    output_mask_key="attention_mask",
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
        )
