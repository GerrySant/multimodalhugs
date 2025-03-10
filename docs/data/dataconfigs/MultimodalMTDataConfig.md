# MultimodalMTDataConfig

<p>

**MultimodalMTDataConfig: Configuration for multimodal machine translation datasets.**

This class defines parameters for handling dataset metadata, preprocessing, 
tokenization, and data shuffling.</p>

<h2>Configuration Fields for MultimodalMTDataConfig</h2>
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
      <td><strong>name</strong></td>
      <td><code>str</code></td>
      <td><code>MultimodalMTDataConfig</code></td>
      <td>No description provided.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>version</strong></td>
      <td><code>Union</code></td>
      <td><code>0.0.0</code></td>
      <td>No description provided.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>data_dir</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>No description provided.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>data_files</strong></td>
      <td><code>Union</code></td>
      <td><code>None</code></td>
      <td>No description provided.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>description</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>No description provided.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>train_metadata_file</strong></td>
      <td><code>Union</code></td>
      <td><code>None</code></td>
      <td>Path to the training dataset metadata file.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>validation_metadata_file</strong></td>
      <td><code>Union</code></td>
      <td><code>None</code></td>
      <td>Path to the validation dataset metadata file.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>test_metadata_file</strong></td>
      <td><code>Union</code></td>
      <td><code>None</code></td>
      <td>Path to the test dataset metadata file.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>shuffle</strong></td>
      <td><code>bool</code></td>
      <td><code>True</code></td>
      <td>If True, shuffles the dataset samples.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>new_vocabulary</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>Path to a file containing new tokens for the tokenizer.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>text_tokenizer_path</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>Path to the pre-trained text tokenizer.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>remove_unused_columns</strong></td>
      <td><code>bool</code></td>
      <td><code>True</code></td>
      <td>If True, removes unused columns from the dataset for efficiency.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>preprocess</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>Configuration for dataset-level preprocessing (e.g., resizing, normalization).</td>
      <td>Check <a href="others/PreprocessArguments.md">PreprocessArguments documentation</a> to see which arguments are accepted.</td>
    </tr>
  </tbody>
</table>