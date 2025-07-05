import argparse
from omegaconf import OmegaConf

from .setup_utils import (
    load_config, prepare_dataset, load_tokenizers,
    save_processor, build_and_save_model, update_configs
)

from multimodalhugs.data.datasets.bilingual_text2text import BilingualText2TextDataset
from multimodalhugs.data.dataset_configs.multimodal_mt_data_config import MultimodalDataConfig
from multimodalhugs.processors import Text2TextTranslationProcessor


def main(config_path: str, do_dataset: bool, do_processor: bool, do_model: bool):
    """
    Run setup steps for dataset preparation, processor instantiation, and model building.

    Args:
        config_path (str): Path to the OmegaConf YAML configuration file.
        do_dataset (bool): If True, prepare the dataset (download and save).
        do_processor (bool): If True, create and save the processing pipeline.
        do_model (bool): If True, build the model and save the weights.

    Behavior:
        - If none of do_dataset, do_processor, do_model are True, all three steps are performed.
        - Tokenizers are loaded as needed for both processor and model steps.
        - After each chosen step, the corresponding path is recorded and written back into the config.
    """
    cfg = load_config(config_path)

    # If no flags were passed, turn everything on
    if not (do_dataset or do_processor or do_model):
        do_dataset = do_processor = do_model = True

    # 1) Dataset setup
    data_path = None
    if do_dataset:
        print("\nSetting Up Dataset:\n")
        # Instantiate and prepare dataset, then save to disk
        data_cfg = MultimodalDataConfig(cfg)
        data_path = prepare_dataset(
            BilingualText2TextDataset,
            data_cfg,
            cfg.training.output_dir
        )

    # 2) Processor setup
    proc_path = None
    if do_processor:
        print("\nSetting Up Processor:\n")
        # Load tokenizers (needed for both processor and model)
        processor_cfg = getattr(cfg, "processor", None)

        text_tokenizer_path  = getattr(processor_cfg, "text_tokenizer_path", None) if processor_cfg else None
        new_vocabulary       = getattr(processor_cfg, "new_vocabulary", None) if processor_cfg else None
        processor_output_dir = (getattr(processor_cfg, "output_dir", None) or getattr(cfg.training, "output_dir", None))
        processor_run_name   = (getattr(processor_cfg, "run_name", None) or getattr(cfg.training, "run_name", None))
        
        # Load tokenizers (needed for both processor and model)
        tok, pre_tok, new = load_tokenizers(
            text_tokenizer_path,
            new_vocabulary,
            processor_output_dir,
            processor_run_name
        )

        # Instantiate processor with modality-specific args
        proc = Text2TextTranslationProcessor(
            tokenizer=tok
        )
        proc_path = save_processor(proc, processor_output_dir)

    # 3) Model setup
    model_path = None
    if do_model:
        print("\nSetting Up Model:\n")
        # Ensure tokenizers are loaded if only building model
        try:
            tok, pre_tok, new
        except NameError:
            processor_cfg = getattr(cfg, "processor", None)
            tok, pre_tok, new = load_tokenizers(
                getattr(processor_cfg, "text_tokenizer_path", None) if processor_cfg else None,
                getattr(processor_cfg, "new_vocabulary", None) if processor_cfg else None,
                (getattr(processor_cfg, "output_dir", None) or getattr(cfg.training, "output_dir", None)),
                (getattr(processor_cfg, "run_name", None) or getattr(cfg.training, "run_name", None))
            )

        # Convert OmegaConf to primitive dict for model constructor
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        mtype = cfg.model.get("type")
        model_path = build_and_save_model(
            model_type=mtype,
            config_path=config_path,
            tokenizer=tok,
            pretrained_tokenizer=pre_tok,
            new_tokens=new,
            model_cfg=model_cfg,
            output_dir=cfg.training.output_dir,
            run_name=cfg.training.run_name
        )

    # 4) Update config file with paths of created artifacts
    update_configs(
        config_path,
        processor_path=proc_path,
        data_path=data_path,
        model_path=model_path
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Setup dataset, processor, and model for multimodal training."
    )
    p.add_argument(
        "--config_path", required=True,
        help="Path to overarching YAML configuration file."
    )
    p.add_argument(
        "--dataset", action="store_true",
        help="Only prepare the dataset (skip processor and model)."
    )
    p.add_argument(
        "--processor", action="store_true",
        help="Only set up the processor (skip dataset and model)."
    )
    p.add_argument(
        "--model", action="store_true",
        help="Only build the model (skip dataset and processor)."
    )
    args = p.parse_args()

    main(
        config_path=args.config_path,
        do_dataset=args.dataset,
        do_processor=args.processor,
        do_model=args.model
    )
