import pytest
import random
import torch

import torch.nn as nn
import numpy as np

from jiwer import wer
from transformers import PreTrainedTokenizerFast
from multimodalhugs.utils.tokenizer_utils import load_tokenizer_from_vocab_file
from multimodalhugs.models.multimodal_embedder.modeling_multimodal_embedder import MultiModalEmbedderModel
from omegaconf import OmegaConf
from .global_variables import DEVICE, INPUTS, LABELS, SAMPLES


@pytest.fixture(scope="function")
def model_setup(request):

    params = request.param
    config_path = params["config_path"]

    # Set seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    cfg = OmegaConf.load(config_path)
    src_tokenizer = load_tokenizer_from_vocab_file(vocab_file=cfg.processor.new_vocabulary)
    tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.processor.text_tokenizer_path)

    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_kwargs.update({
        "src_tokenizer": src_tokenizer,
        "tgt_tokenizer": tgt_tokenizer,
        "config_path": config_path,
        "new_vocab_tokens": src_tokenizer.extra_special_tokens
    })

    model = MultiModalEmbedderModel.build_model(**model_kwargs).to(DEVICE)

    print("model_setup id:", id(model))

    return (model, src_tokenizer, tgt_tokenizer), params


def _train_model(model):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    clip_value = 1.0

    # Move tensors to DEVICE
    for key, value in INPUTS.items():
        if isinstance(value, torch.Tensor):
            INPUTS[key] = value.to(DEVICE)

    losses = []

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

    return model, losses


@pytest.mark.parametrize(
    "model_setup",
    [
        {
            "id": "default_max_length",
            "config_path": "tests/test_model_only/configs/test_default_max_length.yaml",
            "expected_max_length": 200,
        },
        {
            "id": "nondefault_max_length",
            "config_path": "tests/test_model_only/configs/test_nondefault_max_length.yaml",
            "expected_max_length": 15,
        },
        {
            "id": "use_backbone_max_length",
            "config_path": "tests/test_model_only/configs/test_use_backbone_max_length.yaml",
            "expected_max_length": 20,
        },
    ],
    indirect=True,
)
def test_model_maxlength_is_correct(model_setup):
    (model, src_tokenizer, tgt_tokenizer), params = model_setup
    assert model.max_length == params["expected_max_length"], "Model max length is not correct"


@pytest.mark.parametrize(
    "model_setup",
    [
        {
            "id": "default_setup",
            "config_path": "tests/test_model_only/configs/test_model_only.yaml",
        },
    ],
    indirect=True,
)
def test_training(model_setup):

    (model, src_tokenizer, tgt_tokenizer), params = model_setup

    model, losses = _train_model(model)

    last_loss = losses[-1]

    if not any(l < 0.11 for l in losses[-20:]):
        raise AssertionError(f"Model failed to overfit: none of the last 20 losses is below 0.11 (last loss: {last_loss.item()})")


@pytest.fixture(scope="function")
def prepare_trained_model(model_setup):
    (model, src_tokenizer, tgt_tokenizer), params = model_setup

    model, losses = _train_model(model)

    print("prepare_trained_model received model id:", id(model))

    return model, src_tokenizer, tgt_tokenizer


@pytest.mark.parametrize(
    "model_setup",
    [
        {
            "id": "default_setup",
            "config_path": "tests/test_model_only/configs/test_model_only.yaml",
        },
    ],
    indirect=True,
)
def test_overfitting_accuracy(model_setup, prepare_trained_model):

    model, src_tokenizer, tgt_tokenizer = prepare_trained_model

    print("test_overfitting_accuracy received model id:", id(model))

    model.eval()

    predictions = []
    targets = []

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
                max_length=50,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
            predictions.append(tgt_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
            targets.append([tgt_tokenizer.decode(g, skip_special_tokens=True) for g in LABELS[i].unsqueeze(0)][0])

            print("Target:", repr(targets[-1]))
            print("Prediction:", repr(predictions[-1]))

    error_rate = wer(targets, predictions)

    # this assertion needed to be relaxed (from a strict WER of 0.0) because of non-determinism
    # the strict test succeeds on certain machines and fails on others, with the same libraries installed,
    # and despite maximum effort to fix the seed and other sources of non-determinism

    assert error_rate <= 0.125, "Model prediction is not accurate"
