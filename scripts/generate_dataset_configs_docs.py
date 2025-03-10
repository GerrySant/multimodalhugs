import os
import re
import inspect
from dataclasses import fields
from multimodalhugs.utils.registry import DATASET_REGISTRY
from multimodalhugs.data.datasets.pose2text import Pose2TextDataset, Pose2TextDataConfig

import re

def convert_markdown_links_to_html(text):
    """
    Converts Markdown-style links [text](url) in a string to HTML <a> tags.
    
    Args:
        text (str): The input string containing Markdown links.
    
    Returns:
        str: The string with Markdown links replaced by HTML <a> tags.
    """
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    replacement = r'<a href="\2">\1</a>'
    return re.sub(pattern, replacement, text)

def generate_config_docs(cls):
    """
    Generates HTML documentation for a configuration dataclass,
    including its docstring and a table of fields with their type,
    default value, description, and any extra info.
    """
    doc_lines = []
    doc_lines.append(f"# {cls.__name__}\n")
    # Include the class docstring if available (cleaned to remove extra indentation)
    if cls.__doc__:
        doc_lines.append(f"<p>\n\n{inspect.cleandoc(cls.__doc__)}</p>")
        doc_lines.append("")  # Blank line

    doc_lines.append(f"<h2>Configuration Fields for {cls.__name__}</h2>")
    doc_lines.append("<table>")
    doc_lines.append("  <thead>")
    doc_lines.append("    <tr>")
    doc_lines.append("      <th>Field</th>")
    doc_lines.append("      <th>Type</th>")
    doc_lines.append("      <th>Default</th>")
    doc_lines.append("      <th>Description</th>")
    doc_lines.append("      <th>Extra Info</th>")
    doc_lines.append("    </tr>")
    doc_lines.append("  </thead>")
    doc_lines.append("  <tbody>")
    
    for field_item in fields(cls):
        name = field_item.name
        type_str = getattr(field_item.type, '__name__', str(field_item.type))
        default = field_item.default if field_item.default != field_item.default_factory else "Factory"
        help_text = field_item.metadata.get("help", "No description provided.")
        extra_info = field_item.metadata.get("extra_info", "")
        
        # Escape any '<' characters so they display correctly in HTML
        help_text = help_text.replace("<", "&lt;")
        extra_info = extra_info.replace("<", "&lt;")
        extra_info = convert_markdown_links_to_html(extra_info)
        
        doc_lines.append("    <tr>")
        doc_lines.append(f"      <td><strong>{name}</strong></td>")
        doc_lines.append(f"      <td><code>{type_str}</code></td>")
        doc_lines.append(f"      <td><code>{default}</code></td>")
        doc_lines.append(f"      <td>{help_text}</td>")
        doc_lines.append(f"      <td>{extra_info}</td>")
        doc_lines.append("    </tr>")
        
    doc_lines.append("  </tbody>")
    doc_lines.append("</table>")
    return "\n".join(doc_lines)

def generate_class_docs(cls):
    """
    Generates Markdown documentation for a class by including:
      - The (cleaned) class docstring,
      - The constructor signature, and
      - Documentation for each method defined in the class (with its signature and docstring)
        arranged in an HTML table.
    """
    lines = []
    
    # Class title and docstring
    lines.append(f"# {cls.__name__}\n")
    if cls.__doc__:
        lines.append(f"<p>\n\n{inspect.cleandoc(cls.__doc__)}</p>")
    else:
        lines.append("<p>No class docstring provided.</p>")
    lines.append("")
    
    # Constructor signature
    lines.append("<h2>Constructor</h2>")
    lines.append("<pre><code>")
    lines.append(f"{cls.__name__}{str(inspect.signature(cls.__init__))}")
    lines.append("</code></pre>")
    lines.append("")
    
    # Methods documentation using an HTML table
    lines.append("<h2>Methods</h2>")
    lines.append("<table>")
    lines.append("  <thead>")
    lines.append("    <tr>")
    lines.append("      <th>Method Signature</th>")
    lines.append("      <th>Description</th>")
    lines.append("    </tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")
    
    for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
        # Skip __init__ since it's documented above
        if name == "__init__":
            continue
        # Filter to include only methods defined in this class
        if not func.__qualname__.startswith(cls.__name__ + "."):
            continue
        
        signature = str(inspect.signature(func))
        method_sig = f"{name}{signature}"
        doc = inspect.cleandoc(func.__doc__) if func.__doc__ else "No docstring provided."
        # Escape < characters in the docstring for proper HTML rendering
        doc = doc.replace("<", "&lt;")
        
        lines.append("    <tr>")
        lines.append(f"      <td><code>{method_sig}</code></td>")
        lines.append(f"      <td><p>\n\n{doc}</p></td>")
        lines.append("    </tr>")
    
    lines.append("  </tbody>")
    lines.append("</table>")
    
    return "\n".join(lines)

def get_config_class_from_dataset(dataset_cls):
    """
    Inspects the __init__ signature of the dataset class to find the type annotation
    of the 'config' parameter. Returns the config class if found, otherwise None.
    """
    sig = inspect.signature(dataset_cls.__init__)
    config_param = sig.parameters.get("config")
    if config_param is None:
        return None
    # Return the annotation if it is a type
    if isinstance(config_param.annotation, type):
        return config_param.annotation
    return None

def document_others_configs():
    """
    Documents configuration classes from the module
    'multimodalhugs.data.dataset_configs.multimodal_mt_data_config'.
    For each class defined in that module, if a documentation file (named <ClassName>.md)
    does not exist in the base config documentation directory, the documentation is generated
    and stored in the 'others' subdirectory.
    """
    import multimodalhugs.data.dataset_configs.multimodal_mt_data_config as mtdc
    base_config_doc_dir = os.path.join("docs", "data", "dataconfigs")
    others_dir = os.path.join(base_config_doc_dir, "others")
    os.makedirs(others_dir, exist_ok=True)
    
    # Iterate over all classes in the module defined in this file
    for name, cls in inspect.getmembers(mtdc, predicate=inspect.isclass):
        # Ensure the class is defined in the module (not imported)
        if cls.__module__ != mtdc.__name__:
            continue
        # Check if it's a dataclass (has __dataclass_fields__)
        if not hasattr(cls, "__dataclass_fields__"):
            continue
        
        # Determine the expected file in the base directory
        expected_file = os.path.join(base_config_doc_dir, f"{cls.__name__}.md")
        if os.path.exists(expected_file):
            continue
        # Otherwise, generate the documentation and store in 'others'
        doc_text = generate_config_docs(cls)
        output_file = os.path.join(others_dir, f"{cls.__name__}.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(doc_text)
        print(f"Generated documentation for {cls.__name__} in others at: {output_file}")

def main():
    # Base output directories
    base_dataset_doc_dir = os.path.join("docs", "data", "datasets")
    base_config_doc_dir = os.path.join("docs", "data", "dataconfigs")
    os.makedirs(base_dataset_doc_dir, exist_ok=True)
    os.makedirs(base_config_doc_dir, exist_ok=True)
    
    # Iterate over each dataset class in the registry
    for dataset_type, dataset_cls in DATASET_REGISTRY.items():
        # Generate dataset class documentation
        dataset_doc = generate_class_docs(dataset_cls)
        dataset_doc_file = os.path.join(base_dataset_doc_dir, f"{dataset_cls.__name__}.md")
        with open(dataset_doc_file, "w", encoding="utf-8") as f:
            f.write(dataset_doc)
        print(f"Generated dataset documentation for {dataset_cls.__name__} at: {dataset_doc_file}")
        
        # Detect the config class from the constructor's 'config' parameter
        config_cls = get_config_class_from_dataset(dataset_cls)
        if config_cls:
            config_doc = generate_config_docs(config_cls)
            # Use the config class's name for the file
            config_doc_file = os.path.join(base_config_doc_dir, f"{config_cls.__name__}.md")
            with open(config_doc_file, "w", encoding="utf-8") as f:
                f.write(config_doc)
            print(f"Generated config documentation for {config_cls.__name__} at: {config_doc_file}")
        else:
            print(f"No config class found for {dataset_cls.__name__}; skipping config documentation.")
    document_others_configs()
if __name__ == "__main__":
    main()
