# BilingualImage2TextDataset

<p>

**BilingualImage2TextDataset: Dataset for Bilingual Image-to-Text Translation.**

This dataset class extends `BilingualText2TextDataset`, where the source input 
is an image instead of raw text. It supports different configurations for handling images.

Go to [BilingualImage2textMTDataConfig documentation](/docs/data/dataconfigs/BilingualImage2textMTDataConfig.md) to find out what arguments to put in the config.</p>

<h2>Constructor</h2>
<pre><code>
BilingualImage2TextDataset(self, config: multimodalhugs.data.datasets.bilingual_image2text.BilingualImage2textMTDataConfig, *args, **kwargs)
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
      <td><code>_generate_examples(self, **kwargs)</code></td>
      <td><p>

**Generate dataset examples as (key, example) tuples.**

This method:
- Loads metadata from a `.csv` metafile.
- Generates images dynamically if `as_numpy` is enabled.
- Yields processed dataset examples.

**Args:**
- `**kwargs`: Dictionary containing:
    - `metafile_path` (str): Path to the metadata file.
    - `split` (str): Dataset split (`train`, `validation`, or `test`).

**Yields:**
- `Tuple[int, dict]`: Index and dictionary containing processed sample data.</p></td>
    </tr>
    <tr>
      <td><code>_info(self)</code></td>
      <td><p>

**Get dataset information and feature structure.**

Defines the dataset structure and feature types.

**Returns:**
- `DatasetInfo`: Object containing dataset metadata, including:
    - `description`: General dataset information.
    - `features`: Dictionary defining dataset schema.
    - `supervised_keys`: `None` (no explicit supervised key pair).</p></td>
    </tr>
  </tbody>
</table>