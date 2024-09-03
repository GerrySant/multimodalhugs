import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorMultimodalSeq2Seq:
    processor: Any

    def __call__(
        self, samples: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        batch = self.processor(samples, return_tensors="pt")

        return batch