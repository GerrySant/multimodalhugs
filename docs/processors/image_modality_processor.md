# ImageModalityProcessor

`ImageModalityProcessor` loads and preprocesses images for use as encoder input in a
multimodalhugs pipeline.  It accepts image file paths, HTTP/HTTPS URLs, plain text
strings (rendered as typographic images), pre-loaded numpy arrays, and torch tensors.

---

## Accepted inputs

| Signal type | What happens |
|---|---|
| File path — `.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` | Loaded via `transformers.image_utils.load_image()`. Always returns **RGB** with EXIF rotation applied. |
| HTTP/HTTPS URL | Same as file path — loaded directly from the URL. |
| Plain text string (no matching file) | Rendered as typographic images, one `[C, H, W]` image per word, using the font at `font_path`. |
| `.npy` file path | Loaded as a precomputed feature array via numpy. No image preprocessing is applied — the array is returned as-is. Use this when features have already been extracted offline. |
| `np.ndarray` | Converted to a `torch.Tensor` unchanged. |
| `torch.Tensor` | Returned unchanged. |

---

## Output shapes

`process_sample` always returns a float32 tensor of shape `[T, C, H, W]`:

| Input | T | C, H, W |
|---|---|---|
| Image file / URL | 1 | determined by the image on disk (or the preprocessor's output size) |
| Text string | number of words | `(3, height, width)` — or the preprocessor's output size |
| `.npy` / ndarray / tensor | shape unchanged | shape unchanged |

`process_batch` pads along the T dimension and returns `[B, T_max, C, H, W]` with a
`[B, T_max]` attention mask.  For single images (T = 1) this is `[B, 1, C, H, W]`,
compatible with a CLIP `FeatureExtractor` inside the model.

---

## Two preprocessing modes

### Mode 1 — Manual normalisation (`normalize_image + mean + std`)

Pixel values are kept as loaded (0–255 range) and then normalised by:

```
normalised = (pixel_value − mean) / std
```

Use this when the model has no upstream `FeatureExtractor` and consumes normalised
pixels directly as `inputs_embeds`.  You must know the exact `mean` and `std` expected
by the backbone's vision encoder (e.g. CLIP's ImageNet-style values).

```yaml
processor:
  slots:
    - processor_class: ImageModalityProcessor
      processor_kwargs:
        font_path: /path/to/font.ttf   # required for text rendering
        width: 224
        height: 224
        normalize_image: true
        mean: '[0.48145466, 0.4578275, 0.40821073]'   # CLIP mean
        std:  '[0.26862954, 0.26130258, 0.27577711]'  # CLIP std
      output_data_key: input_frames
      output_mask_key: attention_mask
```

### Mode 2 — HuggingFace image processor (`custom_preprocessor_path`)

Loaded / rendered images are passed through the specified HuggingFace image processor
(e.g. `CLIPImageProcessor`), which handles resizing, centre-cropping, channel
reordering, and normalisation.

```yaml
processor:
  slots:
    - processor_class: ImageModalityProcessor
      processor_kwargs:
        custom_preprocessor_path: openai/clip-vit-base-patch32
        font_path: /path/to/font.ttf   # required for text rendering
        width: 224    # used only for text rendering canvas size
        height: 224
        device: cuda  # optional: GPU-accelerated resize + normalisation (transformers 5.x)
      output_data_key: input_frames
      output_mask_key: attention_mask
```

When `custom_preprocessor_path` is set:
- It applies to **both** image file / URL loading **and** text rendering.
- `normalize_image`, `mean`, and `std` are ignored (a warning is emitted if
  `normalize_image=True` is also set).
- `.npy` files, numpy arrays, and tensors are **not** passed through the preprocessor
  — those are assumed to be precomputed features.

---

## Choosing between Mode 1 and Mode 2

| Scenario | Recommended mode |
|---|---|
| Model has a `FeatureExtractor` (e.g. `feature_extractor_type: clip`) | **Mode 2** — use the same model as `custom_preprocessor_path` to produce pixel values that the `FeatureExtractor` expects. |
| Model has no `FeatureExtractor` (images go directly as `inputs_embeds`) | **Mode 1** — manual normalisation. |
| Unknown or custom backbone | **Mode 2** if an image processor is available; otherwise **Mode 1** with the backbone's documented mean/std. |

### How the shapes flow to the model

**With CLIP FeatureExtractor in model + `custom_preprocessor_path`:**

```
processor: ImageModalityProcessor(custom_preprocessor_path="openai/clip-vit-base-patch32")
  → process_sample (image file):     [1, 3, 224, 224]    (CLIP pixel_values)
  → process_batch  (B samples):  [B, 1, 3, 224, 224]    mask [B, 1]

model: feature_extractor_type = clip
  → FeatureExtractor flattens [B, 1, 3, 224, 224] → [B*1, 3, 224, 224]
  → CLIPVisionModelWithProjection → image_embeds [B*1, E]
  → unflatten → [B, 1, E]
  → MultimodalMapper → [B, 1, d_model]
  → Backbone encoder
```

**Without FeatureExtractor + manual normalisation:**

```
processor: ImageModalityProcessor(normalize_image=True, mean=[...], std=[...])
  → process_sample (image file):     [1, 3, 224, 224]
  → process_batch  (B samples):  [B, 1, 3, 224, 224]    mask [B, 1]

model: feature_extractor_type = null
  → inputs_embeds = input_frames  → [B, 1, 3, 224, 224]  (raw normalised pixels)
  → MultimodalMapper               (maps feat_dim=C*H*W or 3 to d_model)
  → Backbone encoder
```

---

## Text-rendered images

When the `signal` column in the TSV contains a plain text string (not a file path),
each word is rendered as a separate `[C, H, W]` image using a TrueType font.  The
sequence of word images becomes the T dimension.

```tsv
signal                  encoder_prompt   decoder_prompt   output
Let's open Access.      lowercase:       __en__           let's open access.
```

For "Let's open Access." (4 words): `process_sample` returns `[4, C, H, W]`.

`font_path`, `width`, and `height` control the rendering canvas.  When
`custom_preprocessor_path` is also set, the rendered frames are resized and normalised
by the preprocessor after rendering.

---

## Parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `custom_preprocessor_path` | `str \| None` | `None` | HuggingFace model ID or local path to an image processor (e.g. `"openai/clip-vit-base-patch32"`). When set, loaded images and rendered text frames are passed through the processor. Takes precedence over `normalize_image`. |
| `font_path` | `str \| None` | `None` | Path to a TrueType font file (`.ttf`). Required when the `signal` column contains text strings. |
| `width` | `int \| None` | `None` | Canvas width in pixels for text rendering. Has no effect when loading from file. |
| `height` | `int \| None` | `None` | Canvas height in pixels for text rendering. Has no effect when loading from file. |
| `normalize_image` | `bool` | `True` | If `True`, normalises pixel values with `mean` and `std`. Requires both to be set. Ignored when `custom_preprocessor_path` is set. |
| `mean` | `list[float] \| str \| None` | `None` | Per-channel mean for normalisation. Accepts a list or a comma-separated string (e.g. `"0.485,0.456,0.406"`). Required when `normalize_image=True` and no preprocessor is set. |
| `std` | `list[float] \| str \| None` | `None` | Per-channel standard deviation. Same format as `mean`. |
| `device` | `str \| None` | `None` | Device on which the `custom_preprocessor` runs (e.g. `"cuda"` or `"cuda:0"`). In transformers 5.x the default `TorchvisionBackend` honours this and runs resize/normalisation on GPU. Ignored when `custom_preprocessor_path` is not set. |

---

## Common mistakes

**Setting `normalize_image=True` together with `custom_preprocessor_path`**

Both normalise the image. The processor's normalisation wins — `normalize_image`,
`mean`, and `std` are silently ignored and a `logger.warning` is emitted.

**Omitting `font_path` when the signal is text**

`font_path` is only needed for text rendering.  If the signal column in the TSV
contains plain text and `font_path` is not set, the processor will raise a
`TypeError` inside Pillow when it tries to open `None` as a font path.

**Using a `.npy` file with `custom_preprocessor_path`**

`.npy` files are treated as precomputed features — the custom preprocessor is NOT
applied.  If your `.npy` files contain raw pixel data that needs preprocessing,
load them as numpy arrays and pre-process outside the processor, or convert them
to image files first.

**Different image sizes in the same batch without a preprocessor**

`pad_and_create_mask` pads along the T dimension only; spatial dimensions (H, W) must
be the same across all samples in a batch.  With `custom_preprocessor_path`, the
preprocessor resizes all images to a fixed size so this is handled automatically.
Without a preprocessor, ensure all images in a batch have the same H and W (e.g. by
setting a fixed `width` and `height` for text rendering, or by pre-resizing files).
