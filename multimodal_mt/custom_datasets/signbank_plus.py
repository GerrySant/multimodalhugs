import csv
import gzip
import json
from pathlib import Path

from tqdm import tqdm

from sign_gpt.custom_datasets.dataset_utils import format_task
from sign_gpt.language_utils.i18n import i18n

csv.field_size_limit(2 ** 20)  # Increase limit to 1MB (2^20 characters)

DATASET_NAME = "signbank_plus"
DATA_PATH = Path(__file__).parent.parent.parent / "processed" / DATASET_NAME
DATA_PATH.mkdir(parents=True, exist_ok=True)

data_dir = Path("/Users/amitmoryossef/dev/sign-language-processing/signbank-annotation/signbank-plus/data/parallel")
if not data_dir.exists():
    raise FileNotFoundError(f"Data directory not found: {data_dir}")

paths = {
    "train": {
        "cleaned": data_dir / "cleaned" / "train.csv",
        "more": data_dir / "more" / "train.csv",
    },
    "validation": {
        "cleaned": data_dir / "cleaned" / "dev.csv",
    },
    "test": {
        "cleaned": data_dir / "test" / "all.csv",
    }
}

TASKS = {
    "signwriting_to_text": {
        "system": "Given a sequence of SignWriting in {signed_language}, translate it into {spoken_language}.",
        "messages": [{
            "input": "{signwriting}",
            "output": "{text}",
        }]
    },
    "text_to_signwriting": {
        "system": "Given {spoken_language} text, translate it into {signed_language} using SignWriting.",
        "messages": [{
            "input": "{text}",
            "output": "{signwriting}",
        }]
    }
}

for split, split_data in paths.items():
    split_files = {
        task: gzip.open(DATA_PATH / f"{task}.{split}.jsonl.gz", "wt", encoding="utf-8")
        for task in TASKS
    }

    for data_type, file_path in split_data.items():
        with open(file_path, "r", encoding="utf-8") as f:
            csv.field_size_limit(2 ** 20)  # Increase limit to 1MB (2^20 characters)
            data = list(csv.DictReader(f))

        for datum in tqdm(data):
            spoken_language, signed_language, *signs = datum['source'].split(' ')
            params = {
                "text": datum['target'],
                "spoken_language": i18n("languages", spoken_language[1:]),
                "signed_language": i18n("signed_languages", signed_language[1:]),
                "data_type": data_type,
                "signwriting": ' '.join(signs)
            }
            for task, file in split_files.items():
                file.write(json.dumps(format_task(TASKS[task], params)) + "\n")

    for file in split_files.values():
        file.close()