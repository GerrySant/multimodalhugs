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
    parser.add_argument("--encoder_prompt", type=str, default="__asl__", help="encoder prompt string.")
    parser.add_argument("--decoder_prompt", type=str, default="__en__", help="decoder prompt string.")
    return parser.parse_args()

# Placeholder functions for constructing new fields
def leave_blank(row):
    return ""

def set_as_0(row):
    return 0

def construct_encoder_prompt(row, encoder_prompt):
    return encoder_prompt

def construct_decoder_prompt(row, decoder_prompt):
    return decoder_prompt

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
    data['signal'] = data.apply(leave_blank, axis=1)
    data['signal_start'] = data.apply(set_as_0, axis=1)
    data['signal_end'] = data.apply(set_as_0, axis=1)
    data['encoder_prompt'] = data.apply(lambda row: construct_encoder_prompt(row, args.encoder_prompt), axis=1)
    data['decoder_prompt'] = data.apply(lambda row: construct_decoder_prompt(row, args.decoder_prompt), axis=1)
    data['output'] = data.apply(leave_blank, axis=1)
    
    # Example of mapping original columns to new ones
    map_column_to_new_field('SENTENCE_NAME', 'signal', data)
    map_column_to_new_field('SENTENCE', 'output', data)
    
    data['signal'] = data['signal'].apply(lambda x: f"{args.pose_directory}/{x}.pose")

    # Select the desired columns for the new dataset
    output_columns = [
        'signal',
        'signal_start',
        'signal_end',
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
