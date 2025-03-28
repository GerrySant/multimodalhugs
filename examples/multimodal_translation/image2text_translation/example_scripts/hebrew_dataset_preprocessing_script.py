#!/usr/bin/env python3

import os
import argparse
import pandas as pd

def read_file_lines(filepath):
    """
    Reads lines from a file efficiently and ensures line count is correct.
    Strips any trailing newlines or whitespace.
    """
    lines = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line.rstrip('\n'))  # Remove newline characters
    return lines

def create_dataframe(sources_file, targets_file, encoder_prompts, decoder_prompts):
    # Read the source and target files
    sources = read_file_lines(sources_file)
    targets = read_file_lines(targets_file)

    # Ensure the two files have the same number of lines
    if len(sources) != len(targets):
        raise ValueError(
            f"Source and target files must have the same number of lines. "
            f"Source lines: {len(sources)}, Target lines: {len(targets)}."
        )

    # Handle encoder prompts
    if os.path.exists(encoder_prompts):
        encoder_prompt = read_file_lines(encoder_prompts)
        if len(encoder_prompt) != len(sources):
            raise ValueError(
                f"encoder prompts file must have the same number of lines as the source file. "
                f"Source lines: {len(sources)}, Prompt lines: {len(encoder_prompt)}."
            )
    else:
        encoder_prompt = [encoder_prompts] * len(sources)

    # Handle decoder prompts
    if os.path.exists(decoder_prompts):
        decoder_prompts = read_file_lines(decoder_prompts)
        if len(decoder_prompts) != len(sources):
            raise ValueError(
                f"decoder prompts file must have the same number of lines as the source file. "
                f"Source lines: {len(sources)}, Prompt lines: {len(decoder_prompts)}."
            )
    else:
        decoder_prompts = [decoder_prompts] * len(sources)

    # Create the DataFrame
    dataframe = pd.DataFrame({
        'signal': sources,
        'encoder_prompt': encoder_prompt,
        'decoder_prompt': decoder_prompts,
        'output': targets
    })

    return dataframe

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process and transform a CSV file.")
    parser.add_argument("sources_file", type=str, help="Path to the file containing source texts.")
    parser.add_argument("targets_file", type=str, help="Path to the file containing target texts.")
    parser.add_argument("encoder_prompts", type=str, help="Path to the file containing encoder prompt texts. If the prompt is fixed, it can be also written here.")
    parser.add_argument("decoder_prompts", type=str, help="Path to the file containing decoder prompt texts. If the prompt is fixed, it can be also written here.")
    parser.add_argument("output_file", type=str, help="Path to the output TSV file.")
    args = parser.parse_args()

    # Check if the output file already exists
    if os.path.exists(args.output_file):
        print(f"Output file '{args.output_file}' already exists. The script will not overwrite it.")
        exit(0)

    try:
        data = create_dataframe(
            args.sources_file, 
            args.targets_file, 
            args.encoder_prompts, 
            args.decoder_prompts
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    # Select the desired columns for the new dataset
    output_columns = [
        'signal',
        'encoder_prompt',
        'decoder_prompt',
        'output'
    ]

    # Save the transformed dataset to a new file
    if args.output_file.endswith('.tsv'):
        data[output_columns].to_csv(args.output_file, sep='\t', index=False)
    else:
        data[output_columns].to_csv(args.output_file, index=False)

    print(f"Transformed dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()
