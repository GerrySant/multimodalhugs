#!/usr/bin/env python
"""
Dispatcher for training setup.
Usage example:
    multimodalhugs-setup --modality "pose2text" --config_path "/path/to/pose2text_config.yaml"
"""

import sys
import logging
import argparse
import textwrap
import inspect

from dataclasses import asdict
from transformers.hf_argparser import HfArgumentParser
from transformers import set_seed

from multimodalhugs.tasks.translation.utils import merge_config_and_command_args
from multimodalhugs.training_setup.setup_configuration_classes import SetupArguments

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

BANNER = textwrap.dedent("""
--------------------------
|                        |
|  multimodalhugs-setup  |
|                        |
--------------------------
""")

MODALITY_MAP = {
    "pose2text": "multimodalhugs.training_setup.pose2text_training_setup",
    "video2text": "multimodalhugs.training_setup.video2text_training_setup",
    "features2text": "multimodalhugs.training_setup.features2text_training_setup",
    "signwriting2text": "multimodalhugs.training_setup.signwriting2text_training_setup",
    "image2text": "multimodalhugs.training_setup.image2text_training_setup",
    "text2text": "multimodalhugs.training_setup.text2text_training_setup"
}

def call_setup(func, sa):
    d = asdict(sa)
    allowed = set(inspect.signature(func).parameters)
    func(**{k: v for k, v in d.items() if k in allowed})

def main():
    print(BANNER)

    parser = HfArgumentParser((SetupArguments,))
    (setup_args,) = parser.parse_args_into_dataclasses()

    if setup_args.config_path:
        setup_args = merge_config_and_command_args(setup_args.config_path, SetupArguments, "setup", setup_args, sys.argv[1:])

    # default behavior: enable all if none were set
    if not (setup_args.do_dataset or setup_args.do_processor or setup_args.do_model):
        setup_args.do_dataset = setup_args.do_processor = setup_args.do_model = True
    
    if setup_args.modality not in MODALITY_MAP:
        sys.exit(f"Unknown modality: {setup_args.modality}")

    # Set seed before initializing model.
    set_seed(setup_args.seed)

    from importlib import import_module
    modality_module = import_module(MODALITY_MAP[setup_args.modality])
    modality_setup_main = modality_module.main

    call_setup(modality_setup_main, setup_args)

if __name__ == "__main__":
    main()
