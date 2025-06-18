# BilingualText2TextDataset

<p>

**BilingualText2TextDataset: A dataset class for bilingual text-to-text translation.**

This dataset class is designed for handling bilingual translation datasets 
where text input in one language is mapped to its corresponding translation.

Go to [MultimodalDataConfig documentation](/docs/data/dataconfigs/MultimodalDataConfig.md) to find out what arguments to put in the config.</p>

<h2>Constructor</h2>
<pre><code>
BilingualText2TextDataset(self, config: multimodalhugs.data.dataset_configs.multimodal_mt_data_config.MultimodalDataConfig, info: Optional[datasets.info.DatasetInfo] = None, *args, **kwargs)
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
- Iterates through each sample and extracts relevant text fields.

**Args:**
- `**kwargs`: Dictionary containing:
    - `metafile_path` (str): Path to the metadata file.
    - `split` (str): The dataset split (`train`, `validation`, or `test`).

**Yields:**
- `Tuple[int, dict]`: Index and dictionary containing processed sample data.</p></td>
    </tr>
    <tr>
      <td><code>_info(self)</code></td>
      <td><p>

**Get dataset information and feature structure.**

Defines the expected structure of the dataset, including input and output text fields.

**Returns:**
- `DatasetInfo`: A dataset metadata object containing:
    - `description`: General dataset information.
    - `features`: The dataset schema with data types.
    - `supervised_keys`: `None` (no explicit supervised key pair).</p></td>
    </tr>
    <tr>
      <td><code>_split_generators(self, dl_manager)</code></td>
      <td><p>

**Define dataset splits based on metadata files.**

**Args:**
- `dl_manager` (DownloadManager): The dataset download manager (not used here since data is local).

**Returns:**
- `List[datasets.SplitGenerator]`: A list of dataset splits (`train`, `validation`, `test`).</p></td>
    </tr>
  </tbody>
</table>