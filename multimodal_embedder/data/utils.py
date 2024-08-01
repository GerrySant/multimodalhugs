import ast
import json
import torch
import pandas as pd

from typing import List
from PIL import Image
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


def load_tokenizer_from_vocab_file(vocab_file):
    vocab = {}
    with open(vocab_file, 'r') as f:
        for line in f:
            token, _ = line.strip().split()
            vocab[token] = len(vocab) + 4  # Start indexing from 4

    special_tokens = {
        '<s>': 0,   # bos_token
        '</s>': 2,  # eos_token, sep_token
        '<unk>': 3, # unk_token
        '<pad>': 1  # pad_token
    }

    combined_vocab = {**special_tokens, **vocab}

    # Save vocab to JSON file
    vocab_json_path = vocab_file.replace('.txt', '.json')
    with open(vocab_json_path, 'w') as f:
        json.dump(combined_vocab, f)

    # Initialize a tokenizer
    tokenizer = Tokenizer(WordLevel(vocab=combined_vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    # Save the tokenizer
    tokenizer_path = vocab_file.replace('.txt', '_tokenizer.json')
    tokenizer.save(tokenizer_path)

    # Load the tokenizer with HuggingFace transformers
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
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
    # Determine the maximum number of frames
    max_frames = max(tensor.size(0) for tensor in tensor_list)
    
    # Extract other dimensions from the first tensor
    BATCH_SIZE = len(tensor_list)
    _, Channels, W, H = tensor_list[0].shape
    
    # Create the output tensor with padding
    padded_tensor = torch.zeros((BATCH_SIZE, max_frames, Channels, W, H))
    mask = torch.zeros((BATCH_SIZE, max_frames), dtype=torch.int)

    for i, tensor in enumerate(tensor_list):
        num_frames = tensor.size(0)
        padded_tensor[i, :num_frames, :, :, :] = tensor
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