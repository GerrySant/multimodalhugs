# SignwritingModalityProcessor

`SignwritingModalityProcessor` converts SignWriting ASCII (FSW) strings into sequences
of image tensors — one image per sign symbol.  Each symbol is rendered, resized to a
fixed canvas, optionally inverted, and passed through a HuggingFace image processor.

Requires the `signwriting` package:
```bash
pip install signwriting
# or
pip install "multimodalhugs[signwriting]"
```

---

## Input

The `signal` column in the TSV must contain a valid FSW (Formal SignWriting) string,
e.g. `M530x529S2ff00482x483S1e530494x490`.

| Signal type | What happens |
|---|---|
| FSW string | Normalised, then each symbol rendered to an image via `signwriting_to_image`. |
| `torch.Tensor` | Returned unchanged. |

---

## Output

| Stage | Shape |
|---|---|
| `process_sample` | `[N_signs, C, H, W]` float32 — one image per sign symbol |
| `process_batch` | `[B, N_max, C, H, W]` float32, mask `[B, N_max]` |

The spatial dimensions `C`, `H`, `W` are determined by the `custom_preprocessor_path`
(or by `channels`, `width`, `height` if no preprocessor is used).

---

## `custom_preprocessor_path` — required

`custom_preprocessor_path` is **required**.  The processor raises `ValueError` at
construction time if it is `None`.  Pass a HuggingFace model ID or local path to an
image processor (e.g. `"openai/clip-vit-base-patch32"`):

```yaml
processor_kwargs:
  custom_preprocessor_path: openai/clip-vit-base-patch32
```

Each rendered sign image (PIL, grayscale-inverted if `invert_frame=True`) is passed
through the preprocessor to produce a normalised `[C, H, W]` tensor.

---

## Rendering pipeline

For each FSW symbol:
1. `signwriting.tokenizer.normalize_signwriting` normalises the FSW string.
2. `signwriting.visualizer.visualize.signwriting_to_image` renders the symbol to a PIL image.
3. The image is centred on a white background of size `(width, height)`.
4. If `invert_frame=True` (default): `PIL.ImageOps.invert` inverts pixel values
   (black symbols on white → white symbols on black).
5. The `custom_preprocessor` converts the PIL image to a normalised `[C, H, W]` tensor.

---

## YAML config example

```yaml
processor:
  slots:
    - processor_class: SignwritingModalityProcessor
      processor_kwargs:
        custom_preprocessor_path: openai/clip-vit-base-patch32
        width: 224
        height: 224
        channels: 3
        invert_frame: true
      output_data_key: input_frames
      output_mask_key: attention_mask

```

Or using the `pipeline:` shorthand:

```yaml
processor:
  pipeline: signwriting2text
  tokenizer_path: facebook/m2m100_418M
  modality_kwargs:
    custom_preprocessor_path: openai/clip-vit-base-patch32
```

---

## Parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `custom_preprocessor_path` | `str` | — | **Required.** HuggingFace model ID or local path to an image processor. |
| `width` | `int` | `224` | Canvas width in pixels for each rendered sign symbol. |
| `height` | `int` | `224` | Canvas height in pixels for each rendered sign symbol. |
| `channels` | `int` | `3` | Number of colour channels in the output tensor. |
| `invert_frame` | `bool` | `True` | Invert rendered images (black-on-white → white-on-black). |
