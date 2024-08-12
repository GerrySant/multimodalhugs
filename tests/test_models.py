import os
import pytest
import torch
import torch.nn as nn
import numpy as np
import random
from jiwer import wer
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerFast
from multimodal_embedder.data import load_tokenizer_from_vocab_file
from multimodal_embedder.models import MultiModalEmbedderModel

from tests.global_variables import DEVICE, SAMPLES, INPUTS, LABELS

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

    config_path = 'tests/tests_other_files/tests_configs/test_model.yaml'
    cfg = OmegaConf.load(config_path)

    src_tokenizer = load_tokenizer_from_vocab_file(cfg.data.src_lang_tokenizer_path)
    tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.data.text_tokenizer_path)

    model = MultiModalEmbedderModel.build_model(
        cfg=cfg.model, 
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer
    ).to(DEVICE)

    return model, src_tokenizer, tgt_tokenizer

# Test function for model overfitting
def test_training(model_setup):
    global model, src_tokenizer, tgt_tokenizer

    model, src_tokenizer, tgt_tokenizer = model_setup

    # Define optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    clip_value = 1.0

    # Define inputs and labels
    # This part needs actual implementation of input tensors and labels

    for key, value in INPUTS.items():
        if isinstance(value, torch.Tensor):
            INPUTS[key] = value.to(DEVICE)

    # Training loop
    for epoch in range(500):  # reduced epoch count for testing
        optimizer.zero_grad()
        output = model(**INPUTS)
        logits = output.logits
        loss = criterion(logits.view(-1, logits.size(-1)), LABELS.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
    
    # Check if model successfully overfitted
    assert loss.item() < 0.11, "Model failed to overfit: loss too high"
    

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
                src_langtoks=INPUTS['src_langtoks'][i].unsqueeze(0),
                max_length=50,  # Maximum length of the output sequence
                num_beams=5,   # Use beam search with specified beam widthÇ
                no_repeat_ngram_size=2,  # Prevent repeating ngrams
                early_stopping=True,  # Stop generating as soon as all beams are finished
                )
            predictions.append(tgt_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
            targets.append([tgt_tokenizer.decode(g, skip_special_tokens=True) for g in LABELS[i].unsqueeze(0)][0])
    error_rate = wer(targets, predictions)

    # Check if predictions are accurate
    assert error_rate == 0, "Model prediction is not accurate"