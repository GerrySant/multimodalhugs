import os
import cv2
import torch
import pyarrow
import numpy as np
import logging

from typing import List, Dict, Any, Optional, Callable, Union

from multimodalhugs.data import pad_and_create_mask, get_images, string_to_list
from multimodalhugs.processors import MultimodalSequence2SequenceProcessor


logger = logging.getLogger(__name__)

class Image2TextTranslationProcessor(MultimodalSequence2SequenceProcessor):  # FeatureExtractionMixin
    name = "image2text_processor"
    attributes = ["tokenizer"]
    model_input_names = ["input_frames", "attention_mask"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        font_path: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        normalize_image: bool = True,
        mean: Optional[Union[str, List[float]]] = None,
        std: Optional[Union[str, List[float]]] = None,
        **kwargs,
    ):
        """
        Initializes the Image2TextTranslationProcessor for converting images into text inputs for multimodal sequence-to-sequence models.

        Args:
            tokenizer (Optional[Any], optional): A tokenizer object used to tokenize the text output.
                Usually loaded via Hugging Face's AutoTokenizer.
            font_path (Optional[str], optional): Path to the `.ttf` file that determines the typography used in the image generation.
                Useful when rendering text or labels over images for OCR-style training.
            width (Optional[int], optional): Target width (in pixels) for images after preprocessing.
                If None, the original image width is preserved.
            height (Optional[int], optional): Target height (in pixels) for images after preprocessing.
                If None, the original image height is preserved.
            normalize_image (bool, optional): If True, normalizes pixel values using the provided mean and standard deviation. Defaults to True.
            mean (Optional[List], optional): Mean pixel values for normalization, specified as a list (e.g., [0.5, 0.5, 0.5]).
                Only used if `normalize_image` is True.
            std (Optional[List], optional): Standard deviation values for normalization, specified as a list (e.g., [0.5, 0.5, 0.5]).
                Only used if `normalize_image` is True.
            **kwargs: Additional keyword arguments passed to the parent class (`MultimodalSequence2SequenceProcessor`), such as `max_seq_length`, `padding`, or modality-specific parameters.
        """
        if normalize_image and (mean is None or std is None):
            raise ValueError("Normalization is enabled (normalize_image=True), but 'mean' and/or 'std' were not provided.")
        if isinstance(mean, str):
            mean = string_to_list(mean)
        if isinstance(std, str):
            std = string_to_list(std)
        self.font_path = font_path
        self.width = width
        self.height = height
        self.normalize_image = normalize_image
        self.mean = mean
        self.std = std
        super().__init__(tokenizer=tokenizer, **kwargs)

    def _image_to_tensor(self, signal: Union[str, np.ndarray, torch.Tensor, pyarrow.lib.StringScalar]) -> torch.Tensor:
        """
        Converts an image or image source (text or path) to a tensor.

        Args:
            signal (Union[str, np.ndarray, torch.Tensor, pyarrow.lib.StringScalar]): 
                Image input as a file path, raw array, tensor, or text string.

        Returns:
            torch.Tensor: Tensor representation of the image.
        """
        if isinstance(signal, torch.Tensor):
            # Case in which the transformation has already been performed during the dataset._get_items_()
            return signal

        if isinstance(signal, pyarrow.lib.StringScalar):
            signal = signal.as_py()

        if isinstance(signal, np.ndarray):
            return torch.from_numpy(signal)

        if isinstance(signal, str):
            if os.path.exists(signal):
                _, ext = os.path.splitext(signal)
                ext = ext.lower()

                if ext == ".npy":
                    image = np.load(signal)
                elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                    image = cv2.imread(signal, cv2.IMREAD_UNCHANGED)
                else:
                    raise ValueError(f"Unsupported file format: {ext}")

                if image is None:
                    raise ValueError(f"Failed to read image from path: {signal}")

                if self.normalize_image:
                    image = (image - self.mean) / self.std

                return torch.from_numpy(image)

            else:
                # Signal is text to render as image
                image = get_images(
                    src_text=signal,
                    font_path=self.font_path,
                    width=self.width,
                    height=self.height,
                    normalize_image=self.normalize_image,
                    mean=self.mean,
                    std=self.std,
                )
                return torch.from_numpy(image)

        raise TypeError(f"Unsupported type for 'signal': {type(signal)}")

    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        tensor_sequences = [self._image_to_tensor(sample["signal"]) for sample in batch]
        padded_inputs, padded_input_masks = pad_and_create_mask(tensor_sequences)
        return {
            "input_frames": padded_inputs,
            "attention_mask": padded_input_masks
        }, kwargs

    def _transform_get_items_output(self, batch):
        """
        Returns a transformation function applied at the dataset level during iteration.

        This method defines a transformation that is applied to each batch **within the dataset iterator**, 
        typically by using `datasets.Dataset.with_transform()`. As a result, the transformation is executed 
        at runtime during `__getitem__()` or `__getitems__()`, which allows it to benefit from prefetching 
        and parallel data loading when using multiple DataLoader workers.

        Unlike the `_obtain_*` methods, which are also executed on-the-fly but within the **processor call 
        (typically inside the DataCollator)**, this transformation occurs **prior to batching and collation**. 
        It is therefore ideal for operations that are expensive and can be parallelized at the sample or batch 
        level, such as decoding signals, loading external files, or converting inputs to intermediate formats.

        Use this method to preprocess inputs early in the pipeline while maintaining a modular design that 
        separates dataset-level and collator-level responsibilities.

        Args:
            batch (Dict[str, List[Any]]): A dictionary representing a batch of dataset examples (not yet collated).

        Returns:
            Dict[str, List[Any]]: The transformed batch, with updated or added fields ready for collation.
        """
        tensor_signals = [self._image_to_tensor(batch["signal"][i]) for i in range(len(batch["signal"]))]
        batch["signal"] = tensor_signals
        return batch