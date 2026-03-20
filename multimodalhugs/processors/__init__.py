from .multimodal_sequence2sequence_processor import MultimodalSequence2SequenceProcessor
from .signwriting_preprocessor import SignwritingProcessor
from .pose2text_preprocessor import Pose2TextTranslationProcessor
from .features2text_preprocessor import Features2TextTranslationProcessor
from .image2text_preprocessor import Image2TextTranslationProcessor
from .text2text_preprocessor import Text2TextTranslationProcessor
from .video2text_preprocessor import Video2TextTranslationProcessor
from .modality_processor import ModalityProcessor
from .pose_modality_processor import PoseModalityProcessor
from .video_modality_processor import VideoModalityProcessor
from .text_modality_processor import TextModalityProcessor
from .features_modality_processor import FeaturesModalityProcessor
from .image_modality_processor import ImageModalityProcessor
from .signwriting_modality_processor import SignwritingModalityProcessor
from .meta_processor import ProcessorSlot, MultimodalMetaProcessor
from .utils import *

from transformers import AutoProcessor

# Register all processors with AutoProcessor so that AutoProcessor.from_pretrained()
# works whenever multimodalhugs.processors is imported, not only inside the CLI scripts.
# register_for_auto_class() writes an "auto_map" into processor_config.json on save,
# enabling class discovery from a saved path.
# AutoProcessor.register() populates the in-memory lookup table for the current process.
Pose2TextTranslationProcessor.register_for_auto_class("AutoProcessor")
AutoProcessor.register("pose2text_translation_processor", Pose2TextTranslationProcessor)

Video2TextTranslationProcessor.register_for_auto_class("AutoProcessor")
AutoProcessor.register("video2text_translation_processor", Video2TextTranslationProcessor)

Features2TextTranslationProcessor.register_for_auto_class("AutoProcessor")
AutoProcessor.register("features2text_translation_processor", Features2TextTranslationProcessor)

SignwritingProcessor.register_for_auto_class("AutoProcessor")
AutoProcessor.register("signwritting_processor", SignwritingProcessor)

Image2TextTranslationProcessor.register_for_auto_class("AutoProcessor")
AutoProcessor.register("image2text_translation_processor", Image2TextTranslationProcessor)

Text2TextTranslationProcessor.register_for_auto_class("AutoProcessor")
AutoProcessor.register("text2text_translation_processor", Text2TextTranslationProcessor)

MultimodalMetaProcessor.register_for_auto_class("AutoProcessor")
AutoProcessor.register("multimodal_meta_processor", MultimodalMetaProcessor)