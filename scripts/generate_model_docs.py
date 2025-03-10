import os
import inspect
import re
from dataclasses import fields
from multimodalhugs.utils.registry import MODEL_REGISTRY

def convert_markdown_links_to_html(text):
    """
    Converts Markdown-style links [text](url) to HTML <a> tags.
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
    if cls.__doc__:
        doc_lines.append(f"<p>{inspect.cleandoc(cls.__doc__)}</p>")
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
        
        # Escape '<' characters for HTML and convert Markdown links if present
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
    lines.append(f"# {cls.__name__}\n")
    if cls.__doc__:
        lines.append(f"<p>\n\n{inspect.cleandoc(cls.__doc__)}\n\n</p>")
    else:
        lines.append("<p>No class docstring provided.</p>")
    lines.append("")
    
    # Document constructor signature
    lines.append("<h2>Constructor</h2>")
    lines.append("<pre><code>")
    lines.append(f"{cls.__name__}{str(inspect.signature(cls.__init__))}")
    lines.append("</code></pre>")
    lines.append("")
    
    # Document methods in an HTML table
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
        if name == "__init__":
            continue
        if not func.__qualname__.startswith(cls.__name__ + "."):
            continue
        
        signature = str(inspect.signature(func))
        method_sig = f"{name}{signature}"
        doc = inspect.cleandoc(func.__doc__) if func.__doc__ else "No docstring provided."
        doc = doc.replace("<", "&lt;")
        
        lines.append("    <tr>")
        lines.append(f"      <td><code>{method_sig}</code></td>")
        lines.append(f"      <td><p>\n\n{doc}\n\n</p></td>")
        lines.append("    </tr>")
    
    lines.append("  </tbody>")
    lines.append("</table>")
    return "\n".join(lines)

def main():
    # Base output directory for model docs
    base_model_doc_dir = os.path.join("docs", "models")
    os.makedirs(base_model_doc_dir, exist_ok=True)
    
    # Iterate over each registered model
    for model_type, model_cls in MODEL_REGISTRY.items():
        # Create a directory for this registered model type
        model_dir = os.path.join(base_model_doc_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate documentation for the model class
        model_doc = generate_class_docs(model_cls)
        model_doc_file = os.path.join(model_dir, f"{model_cls.__name__}.md")
        with open(model_doc_file, "w", encoding="utf-8") as f:
            f.write(model_doc)
        print(f"Generated model documentation for {model_cls.__name__} at: {model_doc_file}")
        
        # Generate documentation for the model config class if available
        config_cls = getattr(model_cls, "config_class", None)
        if config_cls:
            config_doc = generate_config_docs(config_cls)
            config_doc_file = os.path.join(model_dir, f"{config_cls.__name__}.md")
            with open(config_doc_file, "w", encoding="utf-8") as f:
                f.write(config_doc)
            print(f"Generated config documentation for {config_cls.__name__} at: {config_doc_file}")
        else:
            print(f"No config class found for {model_cls.__name__}; skipping config documentation.")

if __name__ == "__main__":
    main()