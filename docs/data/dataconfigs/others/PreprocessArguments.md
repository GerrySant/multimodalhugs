# PreprocessArguments

<p>

**PreprocessArguments: Configuration for image and video frame preprocessing.**

This class defines the preprocessing parameters applied to images or video frames 
before passing them into a model. It includes options for resizing, cropping, 
normalization, and rescaling.</p>

<h2>Configuration Fields for PreprocessArguments</h2>
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
      <td><strong>width</strong></td>
      <td><code>int</code></td>
      <td><code>224</code></td>
      <td>Target width (in pixels) for images/frames after preprocessing.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>height</strong></td>
      <td><code>int</code></td>
      <td><code>224</code></td>
      <td>Target height (in pixels) for images/frames after preprocessing.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>channels</strong></td>
      <td><code>int</code></td>
      <td><code>3</code></td>
      <td>Number of color channels in the images/frames (e.g., 3 for RGB).</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>invert_frame</strong></td>
      <td><code>bool</code></td>
      <td><code>True</code></td>
      <td>If True, inverts pixel values for preprocessing.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>dataset_mean</strong></td>
      <td><code>Optional</code></td>
      <td><code>[0.9819, 0.9819, 0.9819]</code></td>
      <td>Mean pixel values for dataset normalization, specified as a list.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>dataset_std</strong></td>
      <td><code>Optional</code></td>
      <td><code>[0.1283, 0.1283, 0.1283]</code></td>
      <td>Standard deviation values for dataset normalization, specified as a list.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>do_resize</strong></td>
      <td><code>bool</code></td>
      <td><code>False</code></td>
      <td>If True, resizes images/frames to the target width and height.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>do_center_crop</strong></td>
      <td><code>bool</code></td>
      <td><code>False</code></td>
      <td>If True, applies center cropping to images/frames.</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>do_rescale</strong></td>
      <td><code>bool</code></td>
      <td><code>True</code></td>
      <td>If True, rescales pixel values to a fixed range (e.g., 0-1).</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>do_normalize</strong></td>
      <td><code>bool</code></td>
      <td><code>True</code></td>
      <td>If True, normalizes pixel values using dataset mean and std.</td>
      <td></td>
    </tr>
  </tbody>
</table>