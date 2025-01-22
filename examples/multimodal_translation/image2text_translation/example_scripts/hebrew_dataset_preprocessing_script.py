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

def create_dataframe(sources_file, targets_file, source_prompts, generation_prompts):
    # Read the source and target files
    sources = read_file_lines(sources_file)
    targets = read_file_lines(targets_file)

    # Ensure the two files have the same number of lines
    if len(sources) != len(targets):
        raise ValueError(
            f"Source and target files must have the same number of lines. "
            f"Source lines: {len(sources)}, Target lines: {len(targets)}."
        )

    # Handle source prompts
    if os.path.exists(source_prompts):
        source_prompt = read_file_lines(source_prompts)
        if len(source_prompt) != len(sources):
            raise ValueError(
                f"Source prompts file must have the same number of lines as the source file. "
                f"Source lines: {len(sources)}, Prompt lines: {len(source_prompt)}."
            )
    else:
        source_prompt = [source_prompts] * len(sources)

    # Handle generation prompts
    if os.path.exists(generation_prompts):
        generation_prompts = read_file_lines(generation_prompts)
        if len(generation_prompts) != len(sources):
            raise ValueError(
                f"Generation prompts file must have the same number of lines as the source file. "
                f"Source lines: {len(sources)}, Prompt lines: {len(generation_prompts)}."
            )
    else:
        generation_prompts = [generation_prompts] * len(sources)

    # Create the DataFrame
    dataframe = pd.DataFrame({
        'source_signal': sources,
        'source_prompt': source_prompt,
        'generation_prompt': generation_prompts,
        'output_text': targets
    })

    return dataframe

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process and transform a CSV file.")
    parser.add_argument("sources_file", type=str, help="Path to the file containing source texts.")
    parser.add_argument("targets_file", type=str, help="Path to the file containing target texts.")
    parser.add_argument("source_prompts", type=str, help="Path to the file containing source prompt texts. If the prompt is fixed, it can be also written here.")
    parser.add_argument("generation_prompts", type=str, help="Path to the file containing generation prompt texts. If the prompt is fixed, it can be also written here.")
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
            args.source_prompts, 
            args.generation_prompts
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    # Select the desired columns for the new dataset
    output_columns = [
        'source_signal',
        'source_prompt',
        'generation_prompt',
        'output_text'
    ]

    # Save the transformed dataset to a new file
    if args.output_file.endswith('.tsv'):
        data[output_columns].to_csv(args.output_file, sep='\t', index=False)
    else:
        data[output_columns].to_csv(args.output_file, index=False)

    print(f"Transformed dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()
