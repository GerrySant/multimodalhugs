# BilingualImage2textMTDataConfig

<p>

**BilingualImage2textMTDataConfig: Configuration for Bilingual Image-to-Text Machine Translation datasets.**

This configuration class extends `MultimodalMTDataConfig` to support datasets 
where the signal input is an **image representation of text**, rather than raw text. 
It includes additional parameters for font selection and image generation mode.</p>

<h2>Configuration Fields for BilingualImage2textMTDataConfig</h2>
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
      <td><code>BilingualImage2textMTDataConfig</code></td>
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
    <tr>
      <td><strong>font_path</strong></td>
      <td><code>Optional</code></td>
      <td><code>None</code></td>
      <td>Path to the '.ttf' file that determines the Path to the .tff file which determines the typography used in the image generation</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>as_numpy</strong></td>
      <td><code>Optional</code></td>
      <td><code>False</code></td>
      <td>If True, it creates the images when creating the dataset. If False, the image are created in an online manner.</td>
      <td></td>
    </tr>
  </tbody>
</table>