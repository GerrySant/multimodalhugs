import re
import os
import ast
import json
import torch
import pyarrow
import numpy as np
import pandas as pd

from typing import List
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def load_tokenizer_from_vocab_file(vocab_file, special_tokens_dict = None, output_dir = None):
    vocab = {}
    special_tokens = {
        '<s>': 0,   # bos_token
        '</s>': 2,  # eos_token, sep_token
        '<unk>': 3, # unk_token
        '<pad>': 1  # pad_token
    } if special_tokens_dict is None else special_tokens_dict
    
    with open(vocab_file, 'r') as f:
        for line in f:
            token, _ = line.strip().split()
            vocab[token] = len(vocab) + len(special_tokens)  # Start indexing from the last index of the special tokens dict

    combined_vocab = dict(sorted({**special_tokens, **vocab}.items(), key=lambda item: item[1]))

    # Save vocab to JSON file
    vocab_json_path = vocab_file.replace('.txt', '.json')
    vocab_json_path = output_dir + f"/{vocab_json_path.split('/')[-1]}" if output_dir is not None else vocab_json_path
    with open(vocab_json_path, 'w') as f:
        json.dump(combined_vocab, f, indent=4)

    # Initialize a tokenizer
    tokenizer = Tokenizer(WordLevel(vocab=combined_vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    # Save the tokenizer
    tokenizer_path = vocab_json_path.replace('.json', '_tokenizer.json')
    tokenizer.save(tokenizer_path)

    # Load the tokenizer with HuggingFace transformers
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, additional_special_tokens=list(vocab.keys()))
    
    fast_tokenizer.add_special_tokens({
        'bos_token': '<s>',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'sep_token': '</s>',
        'pad_token': '<pad>'
    })

    # Save the HuggingFace tokenizer
    fast_tokenizer.save_pretrained(tokenizer_path.replace('_tokenizer.json', '_tokenizer'))
    return fast_tokenizer

def add_new_special_tokens_from_vocab_file(tokenizer, vocab_file, output_dir=None):
    new_special_tokens = []
    with open(vocab_file, 'r') as f:
        for line in f:
            new_special_tokens.append(line.strip().split()[0])
    tokenizer.add_special_tokens(
        special_tokens_dict={'additional_special_tokens': new_special_tokens}, 
        replace_additional_special_tokens=False
        )
    if output_dir is not None:
        output_dir + "/tokenizer"
    return tokenizer

def _transform(n_px, mean: List[float] = [0.48145466, 0.4578275, 0.40821073], std: List[float] = [0.26862954, 0.26130258, 0.27577711]):
    mean = tuple(mean)
    std = tuple(std)
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize(mean, std),
    ])

def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None

def pad_and_create_mask(tensor_list):
    # Determine the maximum number of frames (t_max)
    max_frames = max(tensor.size(0) for tensor in tensor_list)
    
    # Get the shape of other dimensions (starting from dimension 1)
    other_dimensions = tensor_list[0].shape[1:]
    
    # Create the output tensor with padding
    BATCH_SIZE = len(tensor_list)
    padded_shape = (BATCH_SIZE, max_frames) + other_dimensions
    padded_tensor = torch.zeros(padded_shape)
    
    # Create the mask
    mask = torch.zeros((BATCH_SIZE, max_frames), dtype=torch.int)
    
    # Fill the padded tensor and mask
    for i, tensor in enumerate(tensor_list):
        num_frames = tensor.size(0)
        padded_tensor[i, :num_frames] = tensor
        mask[i, :num_frames] = 1
        
    return padded_tensor, mask


def center_image_on_white_background(original_image, target_width=256, target_height=256, edge_gap=5):
    # Create a new image in white with the target dimensions
    new_image = Image.new("RGB", (target_width, target_height), "white")
    
    # Ensure the image + gap doesn't exceed target dimensions, else resize
    max_width = target_width - 2 * edge_gap
    max_height = target_height - 2 * edge_gap
    original_aspect = original_image.width / original_image.height
    if original_aspect > 1:  # wider than tall
        new_width = min(original_image.width, max_width)
        new_height = int(new_width / original_aspect)
    else:
        new_height = min(original_image.height, max_height)
        new_width = int(new_height * original_aspect)

    resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)

    # Calculate the position to center the original image, accounting for edge gap
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2

    # Adjust x and y to ensure there's a gap from the edge
    # Paste the original image into the center of the new white image
    new_image.paste(resized_original, (x, y), resized_original)

    return new_image

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def grayscale_image(image):
    image_rgb = Image.new('RGB', image.size, (255, 255, 255))
    image_rgb.paste(image, (0, 0), image)
    image_l = image_rgb.convert('L')
    return image_l

def resize_and_center_image(original_image, target_width=256, target_height=256):
    # Calcula el tamaño y posición para centrar la imagen
    original_width, original_height = original_image.size
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = int(original_width * scale)
    resized_height = int(original_height * scale)
    
    try:
        resized_image = original_image.resize((resized_width, resized_height), Image.ANTIALIAS)
    except AttributeError:
        # If there's an AttributeError, it means Image.ANTIALIAS is not available
        resized_image = original_image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    
    # Crea una nueva imagen en blanco
    new_image = Image.new("L", (target_width, target_height), 255)  # 255 para el fondo blanco en modo 'L'
    
    # Calcula la posición para centrar la imagen redimensionada
    x = (target_width - resized_width) // 2
    y = (target_height - resized_height) // 2
    
    # Pega la imagen redimensionada en la nueva imagen, centrada
    new_image.paste(resized_image, (x, y))
    
    return new_image

def check_columns(dataset, required_columns):
    if isinstance(dataset, pd.DataFrame):
        # Handling pandas DataFrame
        return all(column in dataset.columns for column in required_columns)
    else:
        # Handling other types with `column_names` attribute
        return all(column in dataset.column_names for column in required_columns)

def contains_empty(sample):
    return any(v == "" or v is None for v in sample.values())

def sample_source_exists(sample):
    return any(v == "" or v is None for v in sample.values())

def file_exists_filter(column_name, sample):
    """
    Checks if the file specified in sample[column_name] exists.
    """
    return os.path.exists(sample[column_name])

def duration_filter(max_frames, sample):
    """
    Checks if the file specified in sample[column_name] exists.
    """
    return sample["DURATION"] <= max_frames

def split_sentence(sentence):
    if isinstance(sentence, pyarrow.lib.StringScalar):
        sentence = sentence.as_py()
    tokens = re.split(r'(\s+|[.,?\!”":;]+)', sentence)
    return [token for token in tokens if len(token.strip()) > 0]

def create_image(word, font_path, img_size=(224, 224), font_size=48):
    img = Image.new('RGB', img_size, color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0, 0), word, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (img_size[0] - text_width) / 2
    y = (img_size[1] - text_height) / 2
    draw.text((x, y), word, fill="black", font=font)
    return np.array(img)

def normalize_images(images, mean, std):
    normalized_images = []
    for image in images:
        normalized_image = (image / 255.0 - mean) / std
        normalized_images.append(normalized_image)
    return np.array(normalized_images)

def make_image_array(words, font_path, width, height, normalize_image, mean, std):
    images = [create_image(word=word, font_path=font_path, img_size=(width, height)) for word in words]
    images = np.array(images)  # Shape: (T, H, W, C)
    if normalize_image:
        return normalize_images(images, mean, std)
    return images

def get_images(src_text, font_path, width, height, normalize_image, mean, std):
    words = split_sentence(src_text)
    images = make_image_array(words=words, font_path=font_path, width=width, height=height, normalize_image=normalize_image, mean=mean, std=std)
    return np.transpose(images, (0, 3, 1, 2)).astype(np.float32)  # Shape: (T, C, H, W)