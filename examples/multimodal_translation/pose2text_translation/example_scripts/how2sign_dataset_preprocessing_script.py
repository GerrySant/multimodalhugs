#!/usr/bin/env python3

import os
import argparse
import pandas as pd

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and transform a CSV file.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("pose_directory", type=str, help="Path to the pose files.")
    parser.add_argument("output_file", type=str, help="Path to the output CSV file.")
    return parser.parse_args()

# Placeholder functions for constructing new fields
def leave_blank(row):
    return ""

def set_as_0(row):
    return 0

def construct_source_prompt(row):
    return "__asl__"

def construct_generation_prompt(row):
    return "__en__"

def map_column_to_new_field(original_column, new_column_name, data):
    if original_column in data.columns:
        data[new_column_name] = data[original_column]
    else:
        data[new_column_name] = ""  # Fill with empty if column does not exist

def main():
    # Parse arguments
    args = parse_arguments()
    
    if os.path.exists(args.output_file):
        print(f"Output file '{args.output_file}' already exists. The script will not overwrite it.")
        exit(0)

    # Read the input CSV file
    data = pd.read_csv(args.input_file, delimiter="\t")

    # Create new columns using the placeholder functions
    data['source_signal'] = data.apply(leave_blank, axis=1)
    data['source_start'] = data.apply(set_as_0, axis=1)
    data['source_end'] = data.apply(set_as_0, axis=1)
    data['source_prompt'] = data.apply(construct_source_prompt, axis=1)
    data['generation_prompt'] = data.apply(construct_generation_prompt, axis=1)
    data['output_text'] = data.apply(leave_blank, axis=1)
    
    # Example of mapping original columns to new ones
    map_column_to_new_field('SENTENCE_NAME', 'source_signal', data)
    map_column_to_new_field('SENTENCE', 'output_text', data)
    
    data['source_signal'] = data['source_signal'].apply(lambda x: f"{args.pose_directory}/{x}.pose")

    # Select the desired columns for the new dataset
    output_columns = [
        'source_signal',
        'source_start',
        'source_end',
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

if __name__ == "__main__":
    main()
