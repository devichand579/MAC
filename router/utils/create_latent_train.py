import sys
sys.path.append(".")
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer
import os


def create_embeddings(
    master_dataset_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: str = "cuda"
):
    """
    Create latent embeddings for the 'pred' column in master_dataset.csv using BERT or similar models.
    Saves embeddings as numpy array and creates a mapping file linking embeddings to idx and model labels.
    
    Args:
        master_dataset_path: Path to the master_dataset.csv file
        output_path: Base path for output files (default: derived from master_dataset_path)
        model_name: Name of the sentence transformer model to use (default: all-MiniLM-L6-v2)
        batch_size: Batch size for processing embeddings
        device: Device to use ('cuda' or 'cpu')
    """
    print(f"Loading dataset from {master_dataset_path}")
    df = pd.read_csv(master_dataset_path)
    
    print(f"Dataset loaded: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if required columns exist
    required_cols = ['idx', 'pred', 'model']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Load the sentence transformer model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    pred_texts = df['pred'].astype(str).tolist()
    
    print(f"Creating embeddings for {len(pred_texts)} texts...")
    # Create embeddings in batches
    embeddings = []
    for i in tqdm(range(0, len(pred_texts), batch_size), desc="Processing batches"):
        batch_texts = pred_texts[i:i+batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # Normalize embeddings for better similarity calculations
        )
        embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings)
    print(f"Embeddings created. Shape: {all_embeddings.shape}")
    
    base_name = master_dataset_path.replace('.csv', '')
    embeddings_npy_path = f"{base_name}_embeddings.npy"
    mapping_path = f"{base_name}_embedding_mapping.csv"
    
    # Save embeddings as numpy array
    print(f"Saving embeddings to {embeddings_npy_path}")
    np.save(embeddings_npy_path, all_embeddings)
    print(f"Embeddings numpy array saved. Shape: {all_embeddings.shape}")
    
    # Create mapping file linking embeddings to idx and model
    # The row index in embeddings corresponds to the row index in the mapping
    mapping_df = pd.DataFrame({
        'idx': df['idx'].values,
        'model': df['model'].values,
        'embedding_index': np.arange(len(df))  # Index in the embeddings array
    })
    
    # Include score if available
    if 'score' in df.columns:
        mapping_df['score'] = df['score'].values
    
    print(f"Saving embedding mapping to {mapping_path}")
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Mapping file saved with {len(mapping_df)} entries")
    
    print(f"\nSummary:")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Embedding dimension: {all_embeddings.shape[1]}")
    print(f"  - Model used: {model_name}")
    print(f"  - Embeddings numpy file: {embeddings_npy_path}")
    print(f"  - Mapping CSV file: {mapping_path}")    
    return embeddings_npy_path, mapping_path
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create latent embeddings for pred column using BERT or similar models"
    )
    parser.add_argument(
        '--master_dataset_path',
        type=str,
        default='master_dataset.csv',
        help='Path to the master_dataset.csv file'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model name (options: all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6-v2, etc.)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing embeddings'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for model inference'
    )
    
    args = parser.parse_args()
    
    create_embeddings(
        master_dataset_path=args.master_dataset_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device
    )

