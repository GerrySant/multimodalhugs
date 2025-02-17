#!/usr/bin/env python
"""
Dispatcher for training setup.
Usage example:
    multimodalhugs-setup --modality "pose2text" --config_path "/path/to/pose2text_config.yaml"
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="MultimodalHugs Setup CLI. Use --modality to select the training setup for a given modality."
    )
    parser.add_argument(
        "--modality",
        required=True,
        choices=["pose2text", "signwriting2text", "image2text"], # More choises will be implemented as soon we implement them
        help="Specify the modality (e.g. 'pose2text', 'signwriting2text', or 'image2text')."
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to the configuration file."
    )
    # Allow additional arguments if needed by the underlying setup scripts.
    args, remaining_args = parser.parse_known_args()

    if args.modality == "pose2text":
        # Import the pose2text setup script from the new package.
        from multimodalhugs.training_setup.pose2sign_training_setup import main as pose2text_setup
        # Reconstruct sys.argv so that the underlying script sees the remaining arguments and the --config-path.
        sys.argv = [sys.argv[0]] + remaining_args + ["--config-path", args.config_path]
        pose2text_setup(args.config_path)
    elif args.modality == "signwriting2text":
        from multimodalhugs.training_setup.signwriting2text_training_setup import main as signwriting_setup
        sys.argv = [sys.argv[0]] + remaining_args + ["--config-path", args.config_path]
        signwriting_setup(args.config_path)
    elif args.modality == "image2text":
        from multimodalhugs.training_setup.image2text_training_setup import main as image2text_setup
        sys.argv = [sys.argv[0]] + remaining_args + ["--config-path", args.config_path]
        image2text_setup(args.config_path)
    else:
        sys.exit(f"Unknown modality: {args.modality}")

if __name__ == "__main__":
    main()
