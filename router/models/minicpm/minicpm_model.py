import os
import pandas as pd
from typing import Dict, Optional, List, Any

from router.utils.modelling_utils import BaseModel
from chatas.code.utils.dataset import Dialog

class MiniCPMModel(BaseModel):
    
    def __init__(self, nll_data_path: Optional[str] = "./output/logprobs.minicpm"):
        """
        Initializes the MiniCPMModel instance.
        Sets the signal_keys to include 'idx', 'nll', 'first_token_lp'.
        param nll_data_path: Path to the CSV file containing NLL data. If provided,
                             the model will load NLL values from this file.
        """
        self.signal_keys = ['idx', 'nll', 'first_token_lp']
        self.nll_values: Dict[str, float] = {}
        self.first_token_lp_values: Dict[str, float] = {}
        self.nll_data_path = nll_data_path
        
        # Load NLL values from file if provided
        if nll_data_path and os.path.exists(nll_data_path):
            self._load_nll_data(nll_data_path)
            self._load_first_token_lp_data(nll_data_path)
            
        super().__init__()

    def _load_nll_data(self, file_path: str) -> None:
        try:
            df = pd.read_csv(file_path, sep=";") 
            # Assuming the CSV has 'idx' and 'nll' columns
            for _, row in df.iterrows():
                if pd.notna(row['nll']):  # Only store non-NaN NLL values
                    self.nll_values[str(row['idx'])] = float(row['nll'])
                if pd.notna(row['first_token_lp']):  # Only store non-NaN first_token_lp values
                    self.first_token_lp_values[str(row['idx'])] = float(row['first_token_lp'])
            print(f"Loaded {len(self.nll_values)} NLL values from {file_path}")
        except Exception as e:
            print(f"Error loading NLL data: {e}")
        
    
    def _load_first_token_lp_data(self, file_path: str) -> None:
        try:
            df = pd.read_csv(file_path, sep=";") 
            # Assuming the CSV has 'idx' and 'pred' columns for first token log probabilities
            for _, row in df.iterrows():
                if pd.notna(row['pred']):  # Only store non-NaN first token log probabilities
                    self.first_token_lp_values[str(row['idx'])] = float(row['pred'])
            print(f"Loaded {len(self.first_token_lp_values)} first token log probabilities from {file_path}")
        except Exception as e:
            print(f"Error loading first token log probabilities: {e}")
        

    def get_signals(self, input_dialog: Dialog) -> dict[str, int | float | str]:
        """
        Extracts NLL (negative log-likelihood) features from the input dialog.
        """
        idx = input_dialog.idx
        nll = self.nll_values.get(idx)
        first_token_lp = self.first_token_lp_values.get(idx, None)
        
        return {
            'idx': idx,
            'nll': nll,
            'first_token_lp': self.first_token_lp_values.get(idx)  
        }

    def update_nll(self, idx: str, nll_value: float) -> None:
        self.nll_values[idx] = nll_value

    def update_nll_batch(self, idx_nll_pairs: List[Dict[str, Any]]) -> None:
        for item in idx_nll_pairs:
            if 'idx' in item and 'nll' in item and item['nll'] is not None:
                self.nll_values[str(item['idx'])] = float(item['nll'])
            if 'idx' in item and 'first_token_lp' in item and item['first_token_lp'] is not None:
                self.first_token_lp_values[str(item['idx'])] = float(item['first_token_lp'])

    def save_nll_data(self, file_path: Optional[str] = None) -> None:
        save_path = file_path or self.nll_data_path
        if not save_path:
            print("No file path provided for saving NLL data.")
            return
            
        try:
            df = pd.DataFrame([
                {'idx': idx, 'nll': nll, 'first_token_lp': self.first_token_lp_values.get(idx)} 
                for idx, nll in self.nll_values.items()
            ])
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False, sep=";")
            print(f"Saved {len(self.nll_values)} NLL values to {save_path}")
        except Exception as e:
            print(f"Error saving NLL data: {e}")


if __name__ == "__main__":
    # Example usage
    from chatas.code.utils.dataset import Dialog, Utterance
    import tempfile
    
    # Create a temporary CSV file with some NLL data
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(b"idx;pred;nll;first_token_lp\nexample_idx;Some prediction;10.5;1.2\n")
    
    # Initialize the model with the data file
    model = MiniCPMModel(nll_data_path=temp_path)
    
    # Create a test dialog
    dialog = Dialog(idx="example_idx", utterances=[Utterance(text="Hello", images=[]),
                                                Utterance(text="How are you?", images=[]),
                                                Utterance(text="I'm good", images=[])])
    
    # Get signals for the dialog
    features = model.get_signals(dialog)
    print(features)  # Should output: {'idx': 'example_idx', 'nll': 10.5}
    
    # Update an NLL value
    model.update_nll("new_idx", 15.7)
    
    # Update batch of NLL values
    model.update_nll_batch([
        {'idx': 'batch_idx_1', 'nll': 20.1, 'first_token_lp': 1.2},
        {'idx': 'batch_idx_2', 'nll': 25.3, 'first_token_lp': 1.3}
    ])
    
    # Save the updated data
    new_path = temp_path + ".new"
    model.save_nll_data(new_path)
    
    # Clean up temporary files
    os.unlink(temp_path)
    os.unlink(new_path)
