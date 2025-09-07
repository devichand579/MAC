from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Any

from chatas.code.utils.dataset import Dialog

# QB -> NLL
# Minicpm -> NLL
# General -> prefix len, # utterance no # distance from img
# Paligemma
# QwenVl
# MPC -> NLL from frequency *
import numpy as np
np.random.seed(42)


class BaseModel(ABC):
    signal_keys: list[str] | None

    def __init__(self):
        """
        Initializes the BaseModel instance. check if signal_keys is set.
        If not set, raises a ValueError.
        """
        if not hasattr(self, 'signal_keys') or self.signal_keys is None:
            raise ValueError("signal_keys must be set in the derived class.")
        
        if 'idx' not in self.signal_keys:
            raise ValueError("signal_keys must contain 'idx'.")
        
    
    @abstractmethod
    def get_signals(self, input_dialog: Dialog) -> dict[str, int | float | str]:
        """
        Abstract method to get confidence signals based on the input dialog.
        
        :param input_dialog: An instance of Dialog containing the input data.
        :return: A list of signals derived from the input dialog.
        """
        raise NotImplementedError("This method should be implemented in the derived class.")



class CombinedPred:
    idx: str
    pred_by_model: dict[str, str]

    def __init__(self, idx: str, pred_by_model: dict[str, str]):
        """
        Initializes the CombinedPred instance.
        
        :param idx: The identifier for the prediction.
        :param pred_by_model: A dictionary mapping model names to their predictions.
        """
        self.idx = idx
        self.pred_by_model = pred_by_model
    
    def add_prediction(self, model_name: str, prediction: str):
        """
        Adds a prediction for a specific model.
        
        :param model_name: The name of the model.
        :param prediction: The prediction made by the model.
        """
        self.pred_by_model[model_name] = prediction




def save_combined_pred_by_idx(
    combined_pred_by_idx: dict[str, CombinedPred], path: str = 'combined_pred_by_idx.json'
):
    """
    Saves the combined predictions by index to a file.
    
    :param combined_pred_by_idx: A dictionary mapping indices to CombinedPred instances.
    """
    import json
    with open(path, 'w') as f:
        json.dump(
            {idx: pred.pred_by_model for idx, pred in combined_pred_by_idx.items()},
            f,
            indent=4
        )


def load_combined_pred_by_idx(path: str = 'combined_pred_by_idx.json') -> dict[str, CombinedPred]:
    """
    Loads the combined predictions by index from a file.
    
    :return: A dictionary mapping indices to CombinedPred instances.
    """
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return {idx: CombinedPred(idx, pred) for idx, pred in data.items()}