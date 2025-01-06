import pandas as pd
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process and transform a CSV file.")
parser.add_argument("sources_file", type=str, help="Path to the file containing source texts.")
parser.add_argument("targets_file", type=str, help="Path to the file containing target texts.")
parser.add_argument("source_prompts", type=str, help="Path to the file containing source prompt texts. If the prompt is fixed, it can be also written here.")
parser.add_argument("generation_prompts", type=str, help="Path to the file containing generation prompt texts. If the prompt is fixed, it can be also written here.")
parser.add_argument("output_file", type=str, help="Path to the output TSV file.")
args = parser.parse_args()

def create_dataframe(sources_file, targets_file, source_prompts, generation_prompts):

    # Read the source and target files
    with open(sources_file, 'r', encoding='utf-8') as source_file:
        sources = source_file.read().splitlines()

    with open(targets_file, 'r', encoding='utf-8') as target_file:
        targets = target_file.read().splitlines()

    # Ensure the two files have the same number of lines
    if len(sources) != len(targets):
        raise ValueError("Source and target files must have the same number of lines.")

    if os.path.exists(source_prompts):
        with open(source_prompts, 'r', encoding='utf-8') as source_prompts:
            source_prompt = source_prompts.read().splitlines()
    else:
        source_prompt = [source_prompts] * len(sources)

    if os.path.exists(generation_prompts):
        with open(generation_prompts, 'r', encoding='utf-8') as generation_prompts:
            generation_prompts = generation_prompts.read().splitlines()
    else:
        generation_prompts = [generation_prompts] * len(sources)

    # Create the DataFrame
    dataframe = pd.DataFrame({
        'source_text': sources,
        'source_prompt': source_prompt,
        'generation_prompt': generation_prompts,
        'output_text': targets
    })

    return dataframe

try:
    data = create_dataframe(
        args.sources_file, 
        args.targets_file, 
        args.source_prompts, 
        args.generation_prompts
        )

except Exception as e:
    print(f"Error: {e}")


# Select the desired columns for the new dataset
output_columns = [
    'source_text',
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




# He de acabar el script que preprocessa les dades de hebrew, despres acabar de comprobar les classes datasets, i els experiments setup de la carpeta de examples