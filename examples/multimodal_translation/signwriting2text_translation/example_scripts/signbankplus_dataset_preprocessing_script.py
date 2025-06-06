#!/usr/bin/env python3

import os
import argparse
import pandas as pd

from multimodalhugs.custom_datasets import properly_format_signbank_plus

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process and transform a CSV file.")
    parser.add_argument("metadata_file", type=str, help="Path to the file containing the split metadata.")
    parser.add_argument("output_file", type=str, help="Path to the output TSV file.")
    args = parser.parse_args()
    
    if os.path.exists(args.output_file):
        print(f"Output file '{args.output_file}' already exists. The script will not overwrite it.")
        exit(0)

    # Placeholder functions for constructing new fields
    def construct_input(row):
        return ""  # Replace with the actual implementation or keep empty

    def construct_encoder_prompt(row):
        return row['src_lang']  # Replace with the actual implementation or keep empty

    def construct_decoder_prompt(row):
        return row['tgt_lang']  # Replace with the actual implementation or keep empty

    def construct_output(row):
        return ""  # Replace with the actual implementation or keep empty

    def map_column_to_new_field(original_column, new_column_name, data):
        if original_column in data.columns:
            data[new_column_name] = data[original_column]
        else:
            data[new_column_name] = ""  # Fill with empty if column does not exist

    # Process the dataset
    data = properly_format_signbank_plus(args.metadata_file, False)

    # Construct new fields
    data['encoder_prompt'] = data.apply(construct_encoder_prompt, axis=1)
    data['decoder_prompt'] = data.apply(construct_decoder_prompt, axis=1)

    # Map original columns to new ones
    map_column_to_new_field('source', 'signal', data)
    map_column_to_new_field('target', 'output', data)

    # Select the desired columns for the new dataset
    output_columns = [
        'signal',
        'encoder_prompt',
        'decoder_prompt',
        'output'
    ]

    # Save the transformed dataset to a new file, determining format by extension
    if args.output_file.endswith('.tsv'):
        data[output_columns].to_csv(args.output_file, sep='\t', index=False)
    else:
        data[output_columns].to_csv(args.output_file, index=False)

    print(f"Transformed dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()
