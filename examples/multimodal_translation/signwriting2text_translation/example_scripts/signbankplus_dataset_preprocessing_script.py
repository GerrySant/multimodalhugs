import pandas as pd
import os
import argparse

from multimodalhugs.custom_datasets import properly_format_signbank_plus

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process and transform a CSV file.")
parser.add_argument("metadata_file", type=str, help="Path to the file containing the split metadata.")
parser.add_argument("output_file", type=str, help="Path to the output TSV file.")
args = parser.parse_args()

# Placeholder functions for constructing new fields
def construct_input(row):
    return ""  # Replace with the actual implementation or keep empty

def construct_source_prompt(row):
    return row['src_lang']  # Replace with the actual implementation or keep empty

def construct_generation_prompt(row):
    return row['tgt_lang']  # Replace with the actual implementation or keep empty

def construct_output_text(row):
    return ""  # Replace with the actual implementation or keep empty

def map_column_to_new_field(original_column, new_column_name, data):
    if original_column in data.columns:
        data[new_column_name] = data[original_column]
    else:
        data[new_column_name] = ""  # Fill with empty if column does not exist

data = properly_format_signbank_plus(args.metadata_file, False)

data['source_prompt'] = data.apply(construct_source_prompt, axis=1)
data['generation_prompt'] = data.apply(construct_generation_prompt, axis=1)

# Example of mapping original columns to new ones
map_column_to_new_field('source', 'input', data)
map_column_to_new_field('target', 'output_text', data)

# Select the desired columns for the new dataset
output_columns = [
    'input',
    'source_prompt',
    'generation_prompt',
    'output_text'
]

# Save the transformed dataset to a new file, determining format by extension
if args.output_file.endswith('.tsv'):
    data[output_columns].to_csv(args.output_file, sep='\t', index=False)
else:
    data[output_columns].to_csv(args.output_file, index=False)

print(f"Transformed dataset saved to {args.output_file}")