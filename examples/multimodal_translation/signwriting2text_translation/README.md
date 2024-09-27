# SignWriting2Text Translation

## Before Training

This framework is designed to maximize the use of existing Hugging Face pipelines. Most components required for training (e.g., tokenizers, models, processors) can be initialized using the `.from_pretrained()` method. Therefore, before training, you must instantiate and configure all components involved by either using pretrained models or training custom ones. More information about Auto classes can be found [here](https://huggingface.co/docs/transformers/model_doc/auto).

1. **Create and save the dataset:**

    Implement and run the code responsible for creating and saving the dataset you intend to train on. Make sure the dataset class you choose or implement inherits from `datasets.GeneratorBasedBuilder`. Use the following code as a template:

    ```python
    import torch
    from omegaconf import OmegaConf
    from multimodalhugs.data import SignWritingDataset, MultimodalMTDataConfig

    config_path = "/examples/multimodal_translation/configs/example_config.yaml"

    # Load config and initialize dataset
    config = OmegaConf.load(config_path)
    dataset = SignWritingDataset(config=MultimodalMTDataConfig(config))

    # Download, prepare, and save dataset
    dataset.download_and_prepare(config.training.output_dir + "/datasets")
    dataset.as_dataset().save_to_disk(config.training.output_dir + "/datasets")
    ```

2. **Create and save the Tokenizer:**

    In case you're using a custom tokenizer, implement and run the code responsible for creating and saving it. Ensure that the tokenizer is compatible with the `.save_pretrained()` and `.from_pretrained()` methods from the `AutoTokenizer` class.

2. **Create and save the Processor:**

    Implement and run the code responsible for creating and saving the processor used during training. The processor ensures that the dataset batches are transformed correctly for the model. Make sure it is compatible with the `.save_pretrained()` and `.from_pretrained()` methods from Auto. Use the following code as a template:

    ```python
    import torch
    from omegaconf import OmegaConf
    from multimodalhugs.data import MultimodalMTDataConfig
    from multimodalhugs.processors import SignwritingPreprocessor
    from transformers.models.clip.image_processing_clip import CLIPImageProcessor
    from transformers import M2M100Tokenizer
    from multimodalhugs.data import load_tokenizer_from_vocab_file

    config_path = "/examples/multimodal_translation/configs/example_config.yaml"

    config = OmegaConf.load(config_path)
    dataset_config = MultimodalMTDataConfig(config)

    frame_preprocessor = CLIPImageProcessor(
                do_resize = dataset_config.preprocess.do_resize,
                size = dataset_config.preprocess.width,
                do_center_crop = dataset_config.preprocess.do_center_crop,
                do_rescale = dataset_config.preprocess.do_rescale,
                do_normalize = dataset_config.preprocess.do_normalize,
                image_mean = dataset_config.preprocess.dataset_mean,
                image_std = dataset_config.preprocess.dataset_std,
            )

    tokenizer_m2m = M2M100Tokenizer.from_pretrained(config.data.text_tokenizer_path)
    src_tokenizer = load_tokenizer_from_vocab_file(config.data.src_lang_tokenizer_path)

    input_processor = SignwritingPreprocessor(
            width=dataset_config.preprocess.width,
            height=dataset_config.preprocess.height,
            channels=dataset_config.preprocess.channels,
            invert_frame=dataset_config.preprocess.invert_frame,
            dataset_mean=dataset_config.preprocess.dataset_mean,
            dataset_std=dataset_config.preprocess.dataset_std,
            frame_preprocessor = frame_preprocessor,
            tokenizer=tokenizer_m2m,
            lang_tokenizer=src_tokenizer,
    )

    input_processor.save_pretrained(save_directory= config.training.output_dir + "/signwriting_processor", push_to_hub=False)
    ```

4. **Create and save a model:**

    Implement and run the code responsible for creating and saving the model you want to train. Make sure the model is compatible with the `.save_pretrained()` and `.from_pretrained()` methods from `AutoModel` classes. Use the following code as a template:

    ```python
    from omegaconf import OmegaConf
    from transformers import M2M100Tokenizer
    from multimodalhugs.data import load_tokenizer_from_vocab_file
    from multimodalhugs.models import MultiModalEmbedderModel

    config_path = "/examples/multimodal_translation/configs/example_config.yaml"

    cfg = OmegaConf.load(config_path)
    tokenizer_m2m = M2M100Tokenizer.from_pretrained(cfg.data.text_tokenizer_path)
    src_tokenizer = load_tokenizer_from_vocab_file(cfg.data.src_lang_tokenizer_path)

    model = MultiModalEmbedderModel.build_model(cfg.model, src_tokenizer, tokenizer_m2m)
    model.save_pretrained(f"{cfg.training.output_dir}/trained_model")
    ```


5. **Register all the custom Auto Class Component involved in the Training:**

    In the example code below, `AutoConfig`, `AutoModelForSeq2SeqLM`, and `AutoProcessor` are used to register the Model Config, Model, and Processor, respectively. If you are using a custom tokenizer, you can use any `AutoTokenizer` class to register it.

    ```python
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor
    from multimodalhugs.processors import SignwritingPreprocessor
    from multimodalhugs.models import MultiModalEmbedderModel, MultiModalEmbedderConfig

    AutoConfig.register("multimodal_embedder", MultiModalEmbedderConfig)
    AutoModelForSeq2SeqLM.register(MultiModalEmbedderConfig, MultiModalEmbedderModel)
    SignwritingPreprocessor.register_for_auto_class()
    AutoProcessor.register("signwritting_processor", SignwritingPreprocessor)
    ```


## Model Training

This section provides instructions on how to train the multimodal embedder model using the provided bash script. The script uses a pretrained signwriting embedder and runs training on a multimodal dataset. 

### Steps to Train the Model

1. Activate the virtual environment containing all necessary dependencies.
2. Set up the environment variables for the model, processor, data paths, and output directory.
3. Run the Python script with the specified parameters for training and evaluation.

#### Sample Bash Script

```bash
#!/usr/bin/env bash

# Activate the virtual environment
source /path/to/your/environment/bin/activate

# Set model and processor paths
export MODEL_NAME="signwritting_embedder_polynommial"
export MODEL_PATH="/path/to/your/model_directory"
export PROCESSOR_PATH="/path/to/your/processor_directory"
export DATA_PATH="/path/to/your/dataset_directory"
export OUTPUT_PATH="/path/to/your/output_directory"
export CUDA_VISIBLE_DEVICES=0

# Run the training script
python ./examples/multimodal_translation/run_translation.py \
    --model_name_or_path $MODEL_PATH \
    --processor_name_or_path $PROCESSOR_PATH \
    --run_name $MODEL_NAME \
    --dataset_dir $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --do_train True \
    --do_eval True \
    --fp16 \
    --label_smoothing_factor 0.1 \
    --remove_unused_columns False \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy "steps" \
    --eval_steps 10000 \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 3 \
    --overwrite_output_dir \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0.001 \
    --warmup_steps 10000 \
    --max_steps 100000 \
    --lr_scheduler_type "polynomial"

```

## Implemented Example:

[Here](/example_scripts/) you will find the templates to run the training. With them, you only have to enter the following commandline in order to train the model:

```bash
cd /path/to/multimodalhugs/
. /examples/multimodal_translation/signwriting2text_translation/example_scripts/signbankplus_training.sh
```

