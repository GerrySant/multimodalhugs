#!/usr/bin/env python
"""
Dispatcher for Evaluation.
Usage example:
    multimodalhugs-generate \
        --task "translation" \
        --metric_name METRIC_NAME \
        --config_path $CONFIG_PATH \
        --output_dir $OUTPUT_PATH \
        [--additional-arg <value> ...] 
"""

import sys
import argparse

def print_global_help():
    help_text = """usage: multimodalhugs-generate [-h] --task {translation}

MultimodalHugs Evaluation CLI. Use --task to select the task beeing evaluated.

options:
  -h, --help            show this help message and exit
  --task {translation}  Specify the Evaluated task (e.g. 'translation').

For details on task-specific arguments, run:
    multimodalhugs-generate --task <selected_task> --help
"""
    print(help_text)

def main():
    # If only global help is requested, print it.
    if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
        print_global_help()
        sys.exit(0)
    
    # Global parser without automatic help handling.
    global_parser = argparse.ArgumentParser(
        description="MultimodalHugs Evaluation CLI. Use --task to select the task beeing evaluated.",
        add_help=False
    )
    global_parser.add_argument(
        "--task",
        required=True,
        choices=["translation"],  # More choices can be added in the future.
        help="Specify the Evaluated task (e.g. 'translation')."
    )
    global_args, remaining_args = global_parser.parse_known_args()
    
    if global_args.task == "translation":
        from multimodalhugs.tasks import translation_generate_main
        # Pass all remaining arguments (which can include --help) to the task-specific parser.
        sys.argv = [sys.argv[0]] + remaining_args
        translation_generate_main()
    else:
        print(f"Task {global_args.task} is not implemented.")
        sys.exit(1)

if __name__ == "__main__":
    main()
