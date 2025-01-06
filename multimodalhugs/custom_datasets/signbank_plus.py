import os
import csv
import logging

import pandas as pd

from pathlib import Path
from typing import Union

# If you have not defined `logger` yet, you should initialize it like this:
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def properly_format_signbank_plus(path: Union[str, Path], save_corrected: bool = True) -> pd.DataFrame:

    '''
    Inputs:
        - path: Path to the .csv file storing the data.n.
    '''
    def load_sign_writting_file(name: str, array_fields: list[str] = ["texts", "annotated_texts"]):
        with open(name, "r", encoding="utf-8") as f:
            csv.field_size_limit(2 ** 20)  # Increase limit to 1MB (2^20 characters)
            all_instances = list(csv.DictReader(f))

        for instance in all_instances:
            if "puddle_id" in instance:
                instance["puddle_id"] = int(instance["puddle_id"])
            for field in array_fields:
                if field in instance:
                    instance[field] = [t.strip() for t in instance[field].split("᛫")]
                    instance[field] = [t for t in instance[field] if t != ""]

        return all_instances

    def load_sign_writting_data(main_file: str, *modifiers: str):
        main_data = load_sign_writting_file(main_file)
        for modifier in modifiers:
            # Load modifier data
            modifier_data = load_sign_writting_file(modifier)
            dict_data = {(instance["puddle_id"], instance["example_id"]): instance for instance in modifier_data}

            # Update main data
            for instance in main_data:
                key = (instance["puddle_id"], instance["example_id"])
                if key in dict_data:
                    instance.update(dict_data[key])

        return main_data

    if str(path)[-4:] == ".csv":
        dict_list = load_sign_writting_data(path)
        df = pd.DataFrame(dict_list)
        df['tgt_lang'] = df['source'].apply(lambda x: x.split(' ')[0][1:])
        df['tgt_lang'] = df['tgt_lang'].apply(lambda x: x.split('-')[0])
        df['src_lang'] = df['source'].apply(lambda x: x.split(' ')[1][1:])
        df['source'] = df['source'].apply(lambda x: ' '.join(x.split(' ')[2:]))
    elif str(path)[-4:] == ".tsv":
        df = pd.read_csv(path, sep='\t')
        df['tgt_lang'] = df['tgt_lang'].apply(lambda x: x[1:])
        df['src_lang'] = df['src_lang'].apply(lambda x: x[1:])
    else:
        logger.error("manifest_file has not an accepted format")
    dir_name, file_name = os.path.split(path)
    new_file_name = "corrected_" + file_name
    out_path = os.path.join(dir_name, new_file_name)
    df = df.fillna('')
    if save_corrected:
        df.to_csv(out_path, index=False)
    logger.info(f"Saving correctly formated data on: {out_path}")
    return df