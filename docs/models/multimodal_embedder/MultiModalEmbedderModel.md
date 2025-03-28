# MultiModalEmbedderModel

<p>

**MultiModalEmbedderModel: A Transformer-based multimodal model.**

This model extends `transformers.PreTrainedModel`, integrating visual and textual 
inputs using a feature extractor, a Multimodal Mapper (Multimodal Mapper), and 
a backbone Transformer model.

</p>

<h2>Constructor</h2>
<pre><code>
MultiModalEmbedderModel(self, config)
</code></pre>

<h2>Methods</h2>
<table>
  <thead>
    <tr>
      <th>Method Signature</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>_init_backbone(self, config)</code></td>
      <td><p>

**Initialize the Transformer backbone model.**

**Args:**
- `config` (MultiModalEmbedderConfig): Model configuration.

</p></td>
    </tr>
    <tr>
      <td><code>_init_feature_extractor(self, config)</code></td>
      <td><p>

**Initialize the feature extractor.**

**Args:**
- `config` (MultiModalEmbedderConfig): Model configuration.

</p></td>
    </tr>
    <tr>
      <td><code>_init_multimodal_mapper(self, config)</code></td>
      <td><p>

**Initialize the Visual-Language (VL) Mapper.**

**Args:**
- `config` (MultiModalEmbedderConfig): Model configuration.

</p></td>
    </tr>
    <tr>
      <td><code>_reorder_cache(self, past_key_values, beam_idx)</code></td>
      <td><p>

**Reorders the past key-value cache for beam search decoding.**

During beam search, this method reorders `past_key_values` based on the 
surviving beams (`beam_idx`), ensuring that cached values remain aligned 
with the correct sequences.

### **Args:**
- `past_key_values` (Tuple[Tuple[torch.FloatTensor]]):  
Cached self-attention and cross-attention key-value pairs from previous decoding steps.
- `beam_idx` (torch.LongTensor, shape `(num_beams,)`):  
The indices of the beams that survived the last decoding step.

### **Returns:**
- `Tuple[Tuple[torch.FloatTensor]]`: The reordered past key-value states.

</p></td>
    </tr>
    <tr>
      <td><code>forward(self, input_frames: Optional[torch.LongTensor] = None, encoder_prompt: Optional[torch.LongTensor] = None, encoder_prompt_length_padding_mask: Optional[torch.LongTensor] = None, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None, decoder_input_ids: Optional[torch.LongTensor] = None, decoder_attention_mask: Optional[torch.LongTensor] = None, head_mask: Optional[torch.Tensor] = None, decoder_head_mask: Optional[torch.Tensor] = None, cross_attn_head_mask: Optional[torch.Tensor] = None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, inputs_embeds: Optional[torch.FloatTensor] = None, decoder_inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple[torch.Tensor], transformers.modeling_outputs.Seq2SeqLMOutput]</code></td>
      <td><p>

**Forward pass of the MultiModalEmbedderModel.**

This method performs the forward propagation of the model, processing multimodal 
inputs including textual and video-based features. The method integrates visual 
embeddings, applies the Multimodal Mapper (Multimodal Mapper), and processes text 
tokens through the Transformer backbone.

### **Args:**
- `input_frames` (Optional[torch.LongTensor], shape: `(B, N_frames, C, W, H)`):  
The batch of video input frames, where:
    - `B` = batch size  
    - `N_frames` = number of frames per sample  
    - `C` = number of channels  
    - `W` = frame width  
    - `H` = frame height  

- `encoder_prompt` (Optional[torch.LongTensor], shape: `(B, prompt_n_tokens)`):  
A prompt consisting of tokenized text that is prepended to the model's input.

- `encoder_prompt_length_padding_mask` (Optional[torch.LongTensor], shape: `(B, prompt_n_tokens)`):  
Mask to indicate padding tokens in the encoder prompt.

- `input_ids` (Optional[torch.LongTensor], shape: `(B, S_text)`):  
Tokenized input sequence, where:
    - `S_text` = sequence length (number of tokens)  
Padding tokens will be ignored.

- `attention_mask` (Optional[torch.Tensor], shape: `(B, N_frames)`):  
A mask that indicates which tokens or frames should be attended to (`1`) and 
which should be ignored (`0`).

- `decoder_input_ids` (Optional[torch.LongTensor], shape: `(B, T_text)`):  
Input IDs for the decoder during training or inference.  
- If using teacher forcing, should have the format: `['&lt;/s>', '&lt;tgt_lang>', '&lt;token_a>', '&lt;token_b>', '&lt;token_c>']`.  
- In generation mode: `['&lt;s>', '&lt;tgt_lang>']`.

- `decoder_attention_mask` (Optional[torch.LongTensor], shape: `(B, T_text)`):  
Mask for decoder inputs, where `0` indicates padding elements.

- `head_mask` (Optional[torch.Tensor], shape: `(num_layers, num_heads)`):  
Mask for attention heads in the encoder.

- `decoder_head_mask` (Optional[torch.Tensor], shape: `(num_layers, num_heads)`):  
Mask for attention heads in the decoder.

- `cross_attn_head_mask` (Optional[torch.Tensor], shape: `(num_layers, num_heads)`):  
Mask for cross-attention heads in the decoder.

- `encoder_outputs` (Optional[Tuple[Tuple[torch.FloatTensor]]]):  
Precomputed encoder outputs, useful when using cached values for efficiency.

- `past_key_values` (Optional[Tuple[Tuple[torch.FloatTensor]]]):  
Cached past key-value pairs for decoder self-attention and cross-attention.  
Used to speed up autoregressive generation.

- `inputs_embeds` (Optional[torch.FloatTensor], shape: `(B, S_text, hidden_dim)`):  
Precomputed input embeddings instead of `input_ids`.

- `decoder_inputs_embeds` (Optional[torch.FloatTensor], shape: `(B, T_text, hidden_dim)`):  
Precomputed embeddings for decoder inputs.

- `labels` (Optional[torch.LongTensor], shape: `(B, T_text)`):  
Target text token IDs, required during training.  
Should follow the format: `['&lt;tgt_lang>', '&lt;token_a>', '&lt;token_b>', '&lt;token_c>', '&lt;/s>']`.

- `use_cache` (Optional[bool], default=`None`):  
If `True`, enables the use of `past_key_values` for faster decoding.

- `output_attentions` (Optional[bool], default=`None`):  
If `True`, the model outputs attention scores.

- `output_hidden_states` (Optional[bool], default=`None`):  
If `True`, the model outputs hidden states.

- `return_dict` (Optional[bool], default=`None`):  
If `True`, returns a `Seq2SeqLMOutput` instead of a tuple.

### **Returns:**
- `Union[Tuple[torch.Tensor], Seq2SeqLMOutput]`:  
The model output, which includes:
    - `logits` (torch.Tensor, shape `(B, T_text, vocab_size)`) → Model's output token probabilities.
    - `past_key_values` (Optional[Tuple[Tuple[torch.FloatTensor]]]) → Cached attention states (if `use_cache=True`).
    - `decoder_hidden_states` (Optional[Tuple[torch.FloatTensor]]) → Hidden states of the decoder (if `output_hidden_states=True`).
    - `decoder_attentions` (Optional[Tuple[torch.FloatTensor]]) → Attention scores of the decoder (if `output_attentions=True`).

### **Processing Steps:**
1. **Input Embedding:**  
- If `inputs_embeds` is not provided, compute it using `feature_extractor(input_frames)`.
- If a Multimodal Mapper (`multimodal_mapper`) is present, apply it to the embeddings.

2. **Modality Merging:**  
- Combine `inputs_embeds` with the `encoder_prompt`, if provided.
- Use the `merge_modalities()` function to ensure proper alignment.

3. **Transformer Backbone Processing:**  
- The processed embeddings are fed into the backbone Transformer model.

4. **Output Generation:**  
- The model produces token probabilities (`logits`) and optionally outputs attention states.

### **Example Usage:**
```python
model = MultiModalEmbedderModel(config)
input_frames = torch.randn(4, 16, 3, 224, 224)  # Batch of 4 video clips
input_ids = torch.randint(0, 50265, (4, 20))  # Random token IDs
labels = torch.randint(0, 50265, (4, 20))

outputs = model.forward(input_frames=input_frames, input_ids=input_ids, labels=labels)
print(outputs.logits.shape)  # Output: (4, 20, 50265)
```

</p></td>
    </tr>
    <tr>
      <td><code>get_encoder(self)</code></td>
      <td><p>

**Retrieves the encoder component of the model.**

This method returns an `EncoderWrapper`, which encapsulates the model’s encoder 
for use in downstream tasks like sequence-to-sequence generation.

### **Returns:**
- `EncoderWrapper`: The encoder module of the model.

</p></td>
    </tr>
    <tr>
      <td><code>get_input_embeddings(self)</code></td>
      <td><p>

**Retrieve the input embeddings.**

**Returns:**
- `torch.nn.Module`: Input embedding layer.

</p></td>
    </tr>
    <tr>
      <td><code>get_output_embeddings(self)</code></td>
      <td><p>

**Retrieve the output embedding layer (LM Head).**

**Returns:**
- `torch.nn.Module`: LM head layer.

</p></td>
    </tr>
    <tr>
      <td><code>input_to_encoder_outputs(self, input_frames: Optional[torch.LongTensor] = None, encoder_prompt: Optional[torch.LongTensor] = None, encoder_prompt_length_padding_mask: Optional[torch.LongTensor] = None, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None)</code></td>
      <td><p>

**Encodes the multimodal input and returns encoder outputs.**

This method processes multimodal inputs (video frames, text, and embeddings) 
to obtain `encoder_outputs`. It is primarily used during `model.generate()` 
to retrieve encoder representations before decoding.

### **Args:**
- `input_frames` (Optional[torch.LongTensor], shape: `(B, N_frames, C, W, H)`):  
The batch of video input frames, where:
    - `B` = batch size  
    - `N_frames` = number of frames per sample  
    - `C` = number of channels  
    - `W` = frame width  
    - `H` = frame height  

- `encoder_prompt` (Optional[torch.LongTensor], shape: `(B, prompt_n_tokens)`):  
A prompt consisting of tokenized text that is prepended to the model's input.

- `encoder_prompt_length_padding_mask` (Optional[torch.LongTensor], shape: `(B, prompt_n_tokens)`):  
Mask indicating padding tokens in the encoder prompt.

- `input_ids` (Optional[torch.Tensor], shape: `(B, S_text)`):  
Tokenized text input IDs.  
- If `None`, the model relies on `input_frames` for input embeddings.

- `attention_mask` (Optional[torch.Tensor], shape: `(B, N_frames)`):  
A mask indicating which frames should be attended to (`1`) and which should be ignored (`0`).

- `head_mask` (Optional[torch.Tensor], shape: `(num_layers, num_heads)`):  
Mask for attention heads in the encoder.

- `inputs_embeds` (Optional[torch.Tensor], shape: `(B, S_text, hidden_dim)`):  
Precomputed input embeddings instead of `input_ids`.  
- If `None`, embeddings are computed from `input_frames`.

- `output_attentions` (Optional[bool], default=`None`):  
If `True`, the model returns attention weights.

- `output_hidden_states` (Optional[bool], default=`None`):  
If `True`, the model returns hidden states of all layers.

- `return_dict` (Optional[bool], default=`None`):  
If `True`, returns a `BaseModelOutput` instead of a tuple.

### **Returns:**
- `BaseModelOutput` or `Tuple`:  
The encoder outputs containing:
    - `last_hidden_state` (torch.FloatTensor, shape `(B, S_text, hidden_dim)`) → Final encoder hidden states.
    - `hidden_states` (Optional[Tuple[torch.FloatTensor]]) → Hidden states from all layers (if `output_hidden_states=True`).
    - `attentions` (Optional[Tuple[torch.FloatTensor]]) → Attention scores (if `output_attentions=True`).

### **Processing Steps:**
1. **Compute Input Embeddings:**  
- If `inputs_embeds` is not provided, extract features using `feature_extractor(input_frames)`.
- If a Multimodal Mapper (`multimodal_mapper`) is available, apply it to the embeddings.

2. **Merge Modalities:**  
- Combine `inputs_embeds` with `encoder_prompt`, if available.
- Use `merge_modalities()` to align visual and text inputs before passing them to the encoder.

3. **Encode Input Representations:**  
- The processed embeddings are passed to the Transformer encoder to generate `encoder_outputs`.

### **Example Usage:**
```python
model = MultiModalEmbedderModel(config)
input_frames = torch.randn(2, 16, 3, 224, 224)  # Batch of 2 videos
encoder_prompt = torch.randint(0, 50265, (2, 5))  # Random tokenized prompt

encoder_outputs = model.input_to_encoder_outputs(input_frames=input_frames, encoder_prompt=encoder_prompt)
print(encoder_outputs.last_hidden_state.shape)  # Output: (2, sequence_length, hidden_dim)
```

</p></td>
    </tr>
    <tr>
      <td><code>prepare_inputs_for_generation(self, *args, **kwargs)</code></td>
      <td><p>

**Prepares model inputs for autoregressive text generation.**

This method adapts the inputs before passing them to the `backbone` model 
during text generation (e.g., beam search or greedy decoding). It ensures 
stability by handling empty `past_key_values` and properly structuring the 
inputs for multimodal generation.

### **Args:**
- `*args`: Positional arguments passed to the backbone model.
- `**kwargs`: Keyword arguments containing:
    - `past_key_values` (Optional[Tuple[Tuple[torch.FloatTensor]]]):  
    Cached key-value states from previous decoding steps.  
    - If empty (`()`), it is set to `None` to prevent errors in final autoregression steps.
    - `input_frames` (Optional[torch.LongTensor], shape: `(B, N_frames, C, W, H)`):  
    Video input frames.
    - `inputs_embeds` (Optional[torch.Tensor], shape: `(B, S_text, hidden_dim)`):  
    Precomputed input embeddings instead of `input_ids`.
    - `encoder_prompt` (Optional[torch.LongTensor], shape: `(B, prompt_n_tokens)`):  
    Prompt prepended to the input sequence.
    - `encoder_prompt_length_padding_mask` (Optional[torch.LongTensor], shape: `(B, prompt_n_tokens)`):  
    Padding mask for the encoder prompt.

### **Returns:**
- `dict`: A dictionary containing all required inputs for the `backbone.generate()` function.

### **Processing Steps:**
1. **Handle Empty `past_key_values`:**  
- If `past_key_values` is an empty tuple (`()`), it is replaced with `None`.

2. **Retrieve Backbone Model Inputs:**  
- Calls `self.backbone.prepare_inputs_for_generation(*args, **kwargs)` to get base model inputs.

3. **Add Multimodal Inputs:**  
- If `input_frames`, `inputs_embeds`, `encoder_prompt`, or `encoder_prompt_length_padding_mask` 
    are present in `kwargs`, they are added to the model input dictionary.

### **Example Usage:**
```python
model = MultiModalEmbedderModel(config)
input_frames = torch.randn(2, 16, 3, 224, 224)
past_key_values = None  # First decoding step

model_inputs = model.prepare_inputs_for_generation(
    past_key_values=past_key_values, input_frames=input_frames
)
print(model_inputs.keys())  # Output: dict_keys(['input_frames', 'past_key_values'])
```

</p></td>
    </tr>
    <tr>
      <td><code>set_input_embeddings(self, value)</code></td>
      <td><p>

**Set new input embeddings.**

**Args:**
- `value` (torch.nn.Module): New embedding module.

</p></td>
    </tr>
  </tbody>
</table>