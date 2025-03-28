import os

def get_markdown_links(directory, base_path, category_name):
    """Generate markdown links for all .md files in a directory."""
    links = []
    for root, _, files in os.walk(directory):
        for file in sorted(files):  # Sort alphabetically for consistency
            if file.endswith(".md"):
                rel_path = os.path.relpath(os.path.join(root, file), base_path)
                display_name = file.replace(".md", "")
                links.append(f"  - [`{display_name}`]({rel_path})")
    return f"- **{category_name}/**\n" + ("\n".join(links) if links else " No documentation available.")

def generate_readme():
    """Automatically generates the README.md file for the MultiModalHugs documentation."""
    base_path = "docs"
    
    data_dataconfigs = get_markdown_links(os.path.join(base_path, "data/dataconfigs"), base_path, "dataconfigs")
    data_datasets = get_markdown_links(os.path.join(base_path, "data/datasets"), base_path, "datasets")
    model_docs = get_markdown_links(os.path.join(base_path, "models"), base_path, "models")
    
    readme_content = f"""# MultiModalHugs Documentation

## 📂 Directory Structure

The MultiModalHugs documentation is organized into the following sections:

### **1. Data Documentation (`docs/data/`):**
Contains configuration files and dataset specifications.

{data_dataconfigs}

{data_datasets}

### **2. General Documentation (`docs/general/`):**
Contains documentation for core configurations and CLI usage.

- [`CLI.md`](general/CLI.md): Detailed guide for using the MultiModalHugs CLI.
- [`configuration.md`](general/configuration.md): Explanation of configuration file parameters.

### **3. Model Documentation (`docs/models/`):**
Includes specifications for model architectures used in MultiModalHugs.

{model_docs}

## 🚀 Getting Started
To set up, train and evaluate a model, follow these steps:

![Steps Overview](media/steps.png)

## 1. Dataset Preparation
For each partition (train, val, test), create a TSV file that captures essential sample details (input paths, timestamps, prompts, target texts) for consistency. 

#### Metadata File Requirements

The `metadata.tsv` files for each partition must include the following fields:

- `signal`: The source text for the translation from which the images will be created / The path of the images to be uploaded (currently with support for `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.npy`)
- `encoder_prompt`: A text string (e.g., `__vhe__`) that helps the model distinguish the source language or modality. Can be empty if not used.
- `decoder_prompt`: A text prompt appended during decoding to guide the model’s generation. Useful for specifying style or language; can be empty if not used.
- `output`: The target text for translation.

> **Note:** If using a pretrained model, ensure you understand how the model expects the `decoder_prompt`. Some models, such as T5, automatically assume this prompt is always the `<pad>` token and prepend it by default. Therefore, explicitly specifying `<pad>`in this field is unnecessary and would result in unexpected behaviour.

## 2. Setup Datasets, Model and Processors:
   ```bash
   multimodalhugs-setup --modality {{pose2text,signwriting2text,image2text}} --config_path CONFIG_PATH
   ```

## 3. Train a model:
   ```bash
   multimodalhugs-train --task translation --config_path CONFIG_PATH
   ```

## 4. Generate outputs with a trained model:
   ```bash
   multimodalhugs-generate --task translation --metric_name METRIC_NAME --config_path CONFIG_PATH --model_name_or_path MODEL_PATH --processor_name_or_path PROCESSOR_PATH --dataset_dir DATASET_PATH --output_dir OUTPUT_DIR
   ```

> **Note:** For more detailed information on each command, refer to the <a href=\"general/CLI.md\">CLI documentation</a>.

## 📁 Examples

To see concrete examples of dataset configurations and usage for different modalities, check out the [`examples/`](examples/) directory.

## 📖 Additional Resources

- Hugging Face `Trainer` API: [TrainingArguments Documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
- MultiModalHugs Repository: (Include link if available)

---

If you have any questions or suggestions, feel free to contribute or raise an issue!
"""

    with open(os.path.join(base_path, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)

if __name__ == "__main__":
    generate_readme()
