import os
import pytest
import torch
import torch.nn as nn
import numpy as np
import random
from jiwer import wer
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from multimodalhugs.utils.tokenizer_utils import load_tokenizer_from_vocab_file
from multimodalhugs.models.multimodal_embedder.modeling_multimodal_embedder import MultiModalEmbedderModel

from .global_variables import DEVICE, SAMPLES, INPUTS, LABELS

# Declare global variables
model = None
src_tokenizer = None
tgt_tokenizer = None

# Setup configuration and model initialization
@pytest.fixture(scope="module")
def model_setup(): 

    global model, src_tokenizer, tgt_tokenizer
    # Set a fixed seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config_path = 'tests/test_model_only/test_model_only.yaml'
    cfg = OmegaConf.load(config_path)

    src_tokenizer = load_tokenizer_from_vocab_file(vocab_file=cfg.data.new_vocabulary)
    tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.data.text_tokenizer_path)

    # Convert the model config into a dict and update it with the common arguments
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_kwargs.update({
        "src_tokenizer": src_tokenizer,
        "tgt_tokenizer": tgt_tokenizer,
        "config_path": config_path,
        "new_vocab_tokens": src_tokenizer.additional_special_tokens  # Use an empty list if no extra tokens are needed
    })
    model = MultiModalEmbedderModel.build_model(**model_kwargs).to(DEVICE)
    return model, src_tokenizer, tgt_tokenizer

# Test function for model overfitting
def test_training(model_setup):
    global model, src_tokenizer, tgt_tokenizer
    model, src_tokenizer, tgt_tokenizer = model_setup

    # Define optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    clip_value = 1.0

    # Mover tensores a DEVICE
    for key, value in INPUTS.items():
        if isinstance(value, torch.Tensor):
            INPUTS[key] = value.to(DEVICE)

    # Lista para guardar las losses
    losses = []

    # Training loop
    for epoch in range(500):  # reduced epoch count for testing
        optimizer.zero_grad()
        output = model(**INPUTS)
        logits = output.logits
        loss = criterion(logits.view(-1, logits.size(-1)), LABELS.view(-1))
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        print(f"Epoch {epoch} - Loss: {loss}")
    
    # Comprueba si alguno de los últimos 20 valores de la loss es menor que 0.11
    if not any(l < 0.11 for l in losses[-20:]):
        raise AssertionError(f"Model failed to overfit: none of the last 20 losses is below 0.11 (last loss: {loss.item()})")


# Test function for model prediction accuracy
def test_overfitting_accuracy(model_setup):

    global model, src_tokenizer, tgt_tokenizer
    model.eval()

    predictions = []
    targets = []
    
    model.eval()
    with torch.no_grad():
        for i in range(INPUTS['input_frames'].size(0)):
            lang = SAMPLES[i]['tgt_lang']
            generated_ids = model.generate(
                input_frames=INPUTS['input_frames'][i].unsqueeze(0),
                bos_token_id=tgt_tokenizer.convert_tokens_to_ids(tgt_tokenizer.eos_token),
                eos_token_id=tgt_tokenizer.convert_tokens_to_ids(tgt_tokenizer.eos_token),
                pad_token_id=tgt_tokenizer.convert_tokens_to_ids(tgt_tokenizer.pad_token),
                forced_bos_token_id=tgt_tokenizer.convert_tokens_to_ids(f"__{lang}__"),
                attention_mask=INPUTS['attention_mask'][i].unsqueeze(0),
                encoder_prompt=INPUTS['encoder_prompt'][i].unsqueeze(0),
                encoder_prompt_length_padding_mask=INPUTS['encoder_prompt_length_padding_mask'][i].unsqueeze(0),
                max_length=50,  # Maximum length of the output sequence
                num_beams=5,   # Use beam search with specified beam width
                no_repeat_ngram_size=2,  # Prevent repeating ngrams
                early_stopping=True,  # Stop generating as soon as all beams are finished
                )
            predictions.append(tgt_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
            targets.append([tgt_tokenizer.decode(g, skip_special_tokens=True) for g in LABELS[i].unsqueeze(0)][0])
    error_rate = wer(targets, predictions)

    # Check if predictions are accurate
    assert error_rate == 0, "Model prediction is not accurate"