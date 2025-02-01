#!/usr/bin/env python
import sys
import argparse

def main():
    # Create a parser for the global CLI options.
    parser = argparse.ArgumentParser(
        description="MultimodalHugs Training CLI. Use --task to select the training task."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["translation"],  # More choises will be implemented as soon we implement them
        help="Specify the training task (e.g. 'translation')."
    )
    
    # Parse out the --task argument. The remaining arguments will be task-specific.
    args, remaining_args = parser.parse_known_args()
    
    if args.task == "translation":
        # Import the translation runner.
        from multimodalhugs.tasks import translation_main
        # Replace sys.argv with the remaining arguments so that the translation script's own parser sees them.
        sys.argv = [sys.argv[0]] + remaining_args
        translation_main()
    else:
        # If you add more tasks, dispatch them accordingly.
        print(f"Task {args.task} is not implemented.")
        sys.exit(1)

if __name__ == "__main__":
    main()