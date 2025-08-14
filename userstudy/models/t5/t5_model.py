import sys
sys.path.append(".")

from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from typing import Optional
from models.t5.utils import prefix_encoder, suffix_decoder, merge_prefix_suffix
# Assuming these utility functions exist in the specified path


class T5Model:
    """
    A wrapper class for T5ForConditionalGeneration model for text prediction tasks.

    This class handles model and tokenizer initialization, checkpoint loading,
    and running predictions on a specified device.
    """
    def __init__(self, model_name: str, context: bool = False, ckpt: Optional[str] = None, device: str = 'cpu'):
        """
        Initializes the T5Model.

        Args:
            model_name (str): The name of the pretrained T5 model to use (e.g., 't5-base').
            context (bool): Whether to add a special token for context separation.
            ckpt (Optional[str]): Path to a model checkpoint to load.
            device (str): The device to run the model on ('cpu' or 'cuda').
        """
        self.device = device
        print(f"Initializing model on device: {self.device}")

        # Load model and tokenizer from Hugging Face
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left', max_length=512)

        # Add special tokens if required
        self.tokenizer.add_tokens('<tspace>')
        if context:
            self.tokenizer.add_tokens('<|EOU|>')
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Load checkpoint if provided
        if ckpt is not None:
            try:
                # Load checkpoint data, mapping it directly to the specified device
                ckpt_data = torch.load(ckpt, map_location=torch.device(self.device))
                if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
                    self.model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
                    print(f"Checkpoint loaded from {ckpt}")
                else:
                    self.model.load_state_dict(ckpt_data, strict=False)
                    print(f"Checkpoint loaded (raw state dict) from {ckpt}")
            except FileNotFoundError:
                print(f"Checkpoint file not found at {ckpt}. Using vanilla pretrained model.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Using vanilla pretrained model.")
        else:
            print("No checkpoint provided, using vanilla pretrained model.")

        # Move model to the specified device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, max_length: int = 512):
        """
        Generates a prediction for the given input text.

        Args:
            text (str): The input text string.
            max_length (int): The maximum length for the generated sequence.

        Returns:
            str: The generated text, merged with the input prefix.
        """
        inputs = [text]
        # Encode the input text
        encoding = prefix_encoder(self.tokenizer, inputs, max_length=max_length, batch=True)

        # Move tensors to the designated device
        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)

        # Generate output sequences
        with torch.no_grad():
            # Handle models wrapped in DataParallel
            model_to_generate = self.model.module if hasattr(self.model, "module") else self.model
            generated_outputs = model_to_generate.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=3,
                max_new_tokens=max_length,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True
            )

        # Process generated sequences and probabilities
        gen_sequences = generated_outputs.sequences[:, 1:]
        probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)

        try:
            gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
        except RuntimeError as e:
            # Handle potential shape mismatches during probability calculation
            print(f"Exception occurred while calculating probabilities: {e}")
            print("Generated sequences:", self.tokenizer.batch_decode(gen_sequences, skip_special_tokens=True))
            return text

        # Calculate negative log-likelihood
        mask = (gen_sequences != self.tokenizer.pad_token_id).float()
        nll = -torch.log(gen_probs) * mask
        nll = nll.sum(1)
        subword_lens = mask.sum(1)

        # Move sequences to CPU for decoding
        gen_sequences = gen_sequences.cpu()

        # Decode and merge the output
        for i in range(len(inputs)):
            prefix = self.tokenizer.decode(encoding.input_ids[i], skip_special_tokens=True)
            prefix = prefix.replace("<tspace>", " ")
            pred = suffix_decoder(self.tokenizer, gen_sequences[i])
            print(f"pred: {pred}, prefix: {prefix}, nll: {nll[i].item():.4f}, subword_len: {subword_lens[i].item()}")
            total_sentence = merge_prefix_suffix(prefix, pred)
            return total_sentence

        return text


if __name__ == "__main__":
    # Automatically select 'cuda' if available, otherwise default to 'cpu'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device} ---")

    # Path to the checkpoint file (replace with your actual path or set to None)
    # NOTE: The provided path is a placeholder.
    checkpoint_path = "ckpt/t5/t5-base-boozy-cerulean-shark-epoch_39.pth" #"ckpt/t5/t5-base-boozy-cerulean-shark-epoch_39.pth"

    # Initialize the model
    model = T5Model(
        model_name="t5-base",
        context=True,
        ckpt=checkpoint_path,
        device=device
    )

    # Example usage
    text = "i have been to paris <|EOU|> it is also the capital "
    prediction = model.predict(text)
    print("-" * 20)
    print(f"Input: {text}")
    print(f"Prediction: {prediction}")
    print("-" * 20)

    text_2 = "what is the best way to learn python? <|EOU|> i want to be a "
    prediction_2 = model.predict(text_2)
    print(f"Input: {text_2}")
    print(f"Prediction: {prediction_2}")
    print("-" * 20)
