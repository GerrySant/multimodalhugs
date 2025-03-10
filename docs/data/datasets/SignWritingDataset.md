# SignWritingDataset

<p>

**SignWritingDataset: A dataset class for SignWriting-based multimodal translation.**

This dataset class processes SignWriting samples for multimodal machine translation tasks. 
It loads structured datasets from metadata files and prepares examples for training, 
validation, and testing.

Go to [MultimodalMTDataConfig documentation](multimodalhugs/docs/data/dataconfigs/MultimodalMTDataConfig.md) to find out what arguments to put in the config.</p>

<h2>Constructor</h2>
<pre><code>
SignWritingDataset(self, config: multimodalhugs.data.dataset_configs.multimodal_mt_data_config.MultimodalMTDataConfig, *args, **kwargs)
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
- Filters out samples that contain empty values.
- Extracts relevant fields from the dataset.

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

Reads metadata files and creates dataset splits for training, validation, and testing.

**Args:**
- `dl_manager` (DownloadManager): The dataset download manager (not used here).

**Returns:**
- `List[datasets.SplitGenerator]`: A list of dataset splits (`train`, `validation`, `test`).</p></td>
    </tr>
  </tbody>
</table>