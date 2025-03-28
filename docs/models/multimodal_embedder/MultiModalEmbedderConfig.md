# MultiModalEmbedderConfig

<p>This class extends transformers.PretrainedConfig to configure the MultiModalEmbedderModel model class.

This configuration includes parameters for the feature extractor, visual-language mapping, and backbone model.

Refer to the [transformers.PretrainedConfig documentation](https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/configuration#transformers.PretrainedConfig) to specify arguments of the parent class.</p>

<h2>Configuration Fields for MultiModalEmbedderConfig</h2>
<table>
  <thead>
    <tr>
      <th>Field</th>
      <th>Type</th>
      <th>Default</th>
      <th>Description</th>
      <th>Extra Info</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>model_type</strong></td>
      <td><code>str</code></td>
      <td><code>multimodal_embedder</code></td>
      <td>Name of the model to be used.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>d_model</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>Dimention of the model</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>feat_dim</strong></td>
      <td><code>int</code></td>
      <td><code>512</code></td>
      <td>Dimention of the Feature Extractor output. If features are extracted off-line, the dimentionality of features.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>feature_extractor_type</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>Feature Extractor type to be used.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>feature_extractor_config</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>Hyperparameters of the model class specified in feature_extractor_type. Those not specified are assumed to be the default values of the model class.</td>
      <td>In case of initializing the feature_extractor from a pre-trained model, the feature_extractor parameters will be defined automatically, modifying only those specified under this field.</td>
    </tr>
    <tr>
      <td><strong>pretrained_feature_extractor</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>Pretrained Feature Extractor or path to the Pretrained Feature Extractor checkpoint.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>freeze_feature_extractor</strong></td>
      <td><code>bool</code></td>
      <td><code>False</code></td>
      <td>if True, the feature_extractor parameters are frozen during training.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>multimodal_mapper_type</strong></td>
      <td><code>str</code></td>
      <td><code>linear</code></td>
      <td>Chose the Multimodal Mapper type. Options: 'linear', 'adapter'</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>multimodal_mapper_layer_norm_before</strong></td>
      <td><code>bool</code></td>
      <td><code>False</code></td>
      <td>if True, adds a LayerNorm before the multimodal_mapper</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>multimodal_mapper_layer_norm</strong></td>
      <td><code>bool</code></td>
      <td><code>False</code></td>
      <td>if True, adds a LayerNorm inside the multimodal_mapper</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>multimodal_mapperl_mapper_activation</strong></td>
      <td><code>bool</code></td>
      <td><code>False</code></td>
      <td>if True, applies a ReLu at the multimodal_mapperl_mapper output</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>vl_factor</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>If specified, use an adapter as V-L mapper whose overparameterization is given by the given factor</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>multimodal_mapper_dropout</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>Dropout probabilty for the multimodal_mapper</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>freeze_multimodal_mapper</strong></td>
      <td><code>bool</code></td>
      <td><code>False</code></td>
      <td>if True, the multimodal_mapper parameters are frozen during training.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>backbone_used_vocab_size</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>Original vocab_size of the backbone excluding garbage embeddings</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>backbone_type</strong></td>
      <td><code>str</code></td>
      <td><code>m2m_100</code></td>
      <td>Type of the model to be used as a backbone</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>backbone_config</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>Hyperparameters of the model class specified in backbone_type. Those not specified are assumed to be the default values of the model class.</td>
      <td>In case of initializing the backbone from a pre-trained model, the backbone parameters will be defined automatically, modifying only those specified under this field.</td>
    </tr>
    <tr>
      <td><strong>pretrained_backbone</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>Pretrained Backbone or path to the Pretrained Backbone checkpoint.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>backbone_tied_weights_keys</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>Keys of the model parameters that are tied to each other.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>freeze_backbone</strong></td>
      <td><code>bool</code></td>
      <td><code>False</code></td>
      <td>if True, the backbone parameters are frozen during training.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>freeze_encoder_embed_tokens</strong></td>
      <td><code>bool</code></td>
      <td><code>False</code></td>
      <td>if True, the encoder.embed_tokens parameters are frozen during training.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>freeze_decoder_embed_tokens</strong></td>
      <td><code>bool</code></td>
      <td><code>False</code></td>
      <td>if True, the decoder.embed_tokens parameters are frozen during training.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>freeze_lm_head</strong></td>
      <td><code>bool</code></td>
      <td><code>False</code></td>
      <td>if True, the lm_head parameters are frozen during training.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>is_encoder_decoder</strong></td>
      <td><code>bool</code></td>
      <td><code>True</code></td>
      <td>Whether the model is used as an encoder/decoder or not.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>decoder_start_token_id</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>If an encoder-decoder model starts decoding with a different token than _bos_, the id of that token.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>pad_token_id</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>The id of the _padding_ token.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>bos_token_id</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>The id of the _beginning-of-stream_ token.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>eos_token_id</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>The id of the _end-of-stream_ token.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>max_length</strong></td>
      <td><code>int</code></td>
      <td><code>1024</code></td>
      <td>The maximum target length to use when predicting with the generate method.</td>
      <td></td>
    </tr>
  </tbody>
</table>