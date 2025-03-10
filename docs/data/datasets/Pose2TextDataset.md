# Pose2TextDataset

<p>

**Pose2TextDataset: A dataset class for Pose-to-Text tasks.**

This dataset class is designed for processing sign language pose sequences 
and generating text representations. It leverages metadata files to structure 
the data into train, validation, and test splits.

Go to [Pose2TextDataConfig documentation](/docs/data/dataconfigs/Pose2TextDataConfig.md) to find out what arguments to put in the config.</p>

<h2>Constructor</h2>
<pre><code>
Pose2TextDataset(self, config: multimodalhugs.data.datasets.pose2text.Pose2TextDataConfig, *args, **kwargs)
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
- Filters out missing files.
- Reads pose sequences from binary files.
- Filters samples based on duration constraints.

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
      <td><code>_read_pose(self, file_path)</code></td>
      <td><p>

**Read and cache the pose buffer from a file.**

If the same file is requested sequentially, reuse the cached buffer to optimize performance.

**Args:**
- `file_path` (str): Path to the pose data file.

**Returns:**
- `bytes`: The binary pose data buffer.</p></td>
    </tr>
    <tr>
      <td><code>_split_generators(self, dl_manager)</code></td>
      <td><p>

**Define dataset splits based on metadata files.**

**Args:**
- `dl_manager` (DownloadManager): The dataset download manager (not used here).

**Returns:**
- `List[datasets.SplitGenerator]`: A list of dataset splits (`train`, `validation`, `test`).</p></td>
    </tr>
  </tbody>
</table>