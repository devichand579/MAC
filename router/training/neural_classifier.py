import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import logging
import joblib
import argparse

from router.utils.dataset_utils import DatasetType, get_combined_score, get_dataset, partial_f1
from router.utils.modelling_utils import load_combined_pred_by_idx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)





class EmbeddingDataset(Dataset):
    """Dataset class for embeddings and labels"""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    """2-3 layer MLP for model classification"""
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.2):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_mlp_model(embeddings_path: str, mapping_path: str, dataset_type: DatasetType, 
                   hidden_dims=[256, 128, 64], epochs=100, batch_size=256, learning_rate=0.001, 
                   device='cuda' if torch.cuda.is_available() else 'cpu', 
                   use_cost_loss=False, lambda_weight=0.5):
    """
    Train a 2-3 layer MLP on embeddings for model classification.
    
    Args:
        embeddings_path: Path to the embeddings .npy file
        mapping_path: Path to the embedding mapping CSV file
        dataset_type: The type of dataset being used
        hidden_dims: List of hidden layer dimensions (default: [128, 64] for 2 hidden layers)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to use ('cuda' or 'cpu')
        use_cost_loss: Whether to use cost-weighted loss (default: False)
        lambda_weight: Weight for cross-entropy loss in combined loss: lambda * CE + (1-lambda) * Cost (default: 0.5)
    
    Returns:
        model: Trained PyTorch model
        label_encoder: LabelEncoder for model labels
    """
    # Load embeddings and mapping
    logger.info(f"Loading embeddings from {embeddings_path}")
    if not os.path.exists(embeddings_path):
        logger.error(f"Embeddings file {embeddings_path} does not exist.")
        return None, None
    
    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
    
    logger.info(f"Loading mapping from {mapping_path}")
    if not os.path.exists(mapping_path):
        logger.error(f"Mapping file {mapping_path} does not exist.")
        return None, None
    
    mapping_df = pd.read_csv(mapping_path)
    logger.info(f"Loaded mapping with shape: {mapping_df.shape}")
    
    # Ensure embeddings and mapping are aligned by embedding_index
    mapping_df = mapping_df.sort_values('embedding_index').reset_index(drop=True)
    
    # Extract labels
    model_labels = mapping_df['model'].values
    idx_values = mapping_df['idx'].values
    
    # Encode model labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(model_labels)
    num_classes = len(label_encoder.classes_)
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Model label distribution:\n{pd.Series(model_labels).value_counts().to_string()}")
    
    # Define cost mapping (avg time per sample in seconds)
    cost_mapping = {
        'paligemma': 1.4802,
        'qwen2.vl.2b.instruct': 0.7338,
        'minicpm_image': 2.0891,
        'qb': 0.00094
    }
    
    # Create cost tensor for each class
    cost_tensor = None
    if use_cost_loss:
        costs = []
        for class_name in label_encoder.classes_:
            # Normalize class name to match cost_mapping keys
            class_name_normalized = class_name.lower().strip()
            if class_name_normalized in cost_mapping:
                costs.append(cost_mapping[class_name_normalized])
            else:
                # Default cost if model not in mapping (use average)
                default_cost = sum(cost_mapping.values()) / len(cost_mapping)
                logger.warning(f"Model '{class_name}' not found in cost_mapping. Using default cost: {default_cost}")
                costs.append(default_cost)
        
        # Normalize costs to [0, 1] range for better training stability
        max_cost = max(costs)
        min_cost = min(costs)
        if max_cost > min_cost:
            costs_normalized = [(c - min_cost) / (max_cost - min_cost) for c in costs]
        else:
            costs_normalized = [0.5] * len(costs)  # All same cost
        
        cost_tensor = torch.FloatTensor(costs_normalized).to(device)
        logger.info(f"Cost tensor (normalized): {dict(zip(label_encoder.classes_, costs_normalized))}")
        logger.info(f"Using cost-weighted loss with lambda={lambda_weight}")
    
    # Verify embeddings and labels are aligned
    if len(embeddings) != len(encoded_labels):
        logger.error(f"Mismatch: embeddings length ({len(embeddings)}) != labels length ({len(encoded_labels)})")
        # Align them using embedding_index
        max_idx = min(len(embeddings), len(mapping_df))
        embeddings = embeddings[:max_idx]
        encoded_labels = encoded_labels[:max_idx]
        idx_values = idx_values[:max_idx]
        logger.info(f"Aligned to {max_idx} samples")
    
    # Split data
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        embeddings, encoded_labels, idx_values, 
        test_size=0.2, random_state=42, 
        stratify=encoded_labels if len(np.unique(encoded_labels)) > 1 else None
    )
    
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Create datasets and dataloaders
    train_dataset = EmbeddingDataset(X_train, y_train)
    test_dataset = EmbeddingDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_dim = embeddings.shape[1]
    model = MLPClassifier(input_dim, hidden_dims, num_classes).to(device)
    logger.info(f"Model architecture:\n{model}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    logger.info(f"Starting training on {device}...")
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            
            # Compute cross-entropy loss
            ce_loss = criterion(outputs, batch_labels)
            
            # Compute combined loss if cost loss is enabled
            if use_cost_loss and cost_tensor is not None:
                # Get softmax probabilities
                probs = torch.softmax(outputs, dim=1)
                # Compute expected cost: sum over classes of (probability * cost)
                expected_cost = torch.sum(probs * cost_tensor.unsqueeze(0), dim=1)
                # Average over batch
                cost_loss = torch.mean(expected_cost)
                # Combined loss: lambda * CE + (1 - lambda) * Cost
                loss = lambda_weight * ce_loss + (1 - lambda_weight) * cost_loss
            else:
                loss = ce_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_embeddings, batch_labels in test_loader:
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_embeddings)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, "
                       f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
    
    logger.info(f"Training completed. Best test accuracy: {best_test_acc:.2f}%")
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_embeddings, batch_labels in test_loader:
            batch_embeddings = batch_embeddings.to(device)
            outputs = model(batch_embeddings)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)
    
    logger.info(f"Final Test Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{report}")
    
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    
    # Create reverse mapping for model labels
    model_label_map = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    rev_model_label_map = {v: k for k, v in model_label_map.items()}
    print(f"Label mapping: {model_label_map}")
    
    # Evaluate with combined score
    dataset = get_dataset(dataset_type)
    combined_pred_by_idx_filename = f"combined_pred_by_idx_{dataset_type.value}.json"
    combined_pred_by_idx = load_combined_pred_by_idx(combined_pred_by_idx_filename)
    
    y_pred_df = pd.DataFrame({
        'idx': idx_test,
        'model': [rev_model_label_map[pred] for pred in all_preds]
    })
    y_test_df = pd.DataFrame({
        'idx': idx_test,
        'model': [rev_model_label_map[label] for label in all_labels]
    })
    
    print(f"\nPredicted model distribution:")
    pred_counts = y_pred_df['model'].value_counts()
    print(pred_counts)
    
    # Calculate final time cost based on prediction fractions
    total_test_samples = len(y_pred_df)
    final_time_per_sample = 0.0
    total_time = 0.0
    
    print(f"\nCost analysis for test set:")
    print(f"{'Model':<30} {'Count':<10} {'Fraction':<12} {'Cost (s/sample)':<18} {'Fraction×Cost':<15} {'Total Time (s)':<15}")
    print("-" * 100)
    
    for model_name in pred_counts.index:
        count = pred_counts[model_name]
        fraction = count / total_test_samples
        # Get cost for this model (normalize name to match cost_mapping keys)
        model_name_normalized = model_name.lower().strip()
        if model_name_normalized in cost_mapping:
            cost = cost_mapping[model_name_normalized]
        else:
            # Default cost if not found
            default_cost = sum(cost_mapping.values()) / len(cost_mapping)
            cost = default_cost
        
        fraction_times_cost = fraction * cost
        model_total_time = count * cost
        final_time_per_sample += fraction_times_cost
        total_time += model_total_time
        
        print(f"{model_name:<30} {count:<10} {fraction:<12.4f} {cost:<18.4f} {fraction_times_cost:<15.4f} {model_total_time:<15.4f}")
    
    print("-" * 100)
    print(f"{'SUM (Average Time per Sample)':<30} {'-':<10} {'-':<12} {'-':<18} {final_time_per_sample:<15.4f} {'-':<15}")
    print(f"{'Total Time for Test Set':<30} {'-':<10} {'-':<12} {'-':<18} {'-':<15} {total_time:<15.4f}")
    print(f"\nFinal average time per sample (sum of fraction × cost): {final_time_per_sample:.4f} seconds")
    print(f"Total time for test set ({total_test_samples} samples): {total_time:.4f} seconds")
    
    logger.info(f"Final average time per sample: {final_time_per_sample:.4f} seconds")
    logger.info(f"Total time for test set: {total_time:.4f} seconds")
    
    for model_name, label in model_label_map.items():
        print(f"If only {model_name} is used")
        temp_y_pred_df = y_pred_df.copy()
        temp_y_pred_df['model'] = model_name
        score = get_combined_score(dataset, combined_pred_by_idx, temp_y_pred_df)
        print(f"Score for {model_name}: {score}")
    
    score = get_combined_score(dataset, combined_pred_by_idx, y_pred_df)
    print(f"Score using router: {score}")
    score = get_combined_score(dataset, combined_pred_by_idx, y_test_df)
    print(f"Max Score on test set: {score}")
    
    return model, label_encoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MLP classifier on embeddings for model selection")
    parser.add_argument('--embeddings_path', type=str, default=None, 
                       help='Path to embeddings .npy file (default: auto-detect from dataset)')
    parser.add_argument('--mapping_path', type=str, default=None,
                       help='Path to embedding mapping CSV (default: auto-detect from dataset)')
    parser.add_argument('--dataset', type=str, default='mmdd', choices=['mmdd', 'image_chat'],
                       help='Dataset type')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[ 256, 128, 64],
                       help='Hidden layer dimensions (default: [128, 64] for 2 hidden layers)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training (default: 256)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--use_cost_loss', action='store_true',
                       help='Enable cost-weighted loss: lambda * CE + (1-lambda) * Cost')
    parser.add_argument('--lambda', type=float, default=0.5, dest='lambda_weight',
                       help='Weight for cross-entropy loss in combined loss (default: 0.5)')
    
    args = parser.parse_args()
    
    # Determine dataset type
    if args.dataset == "mmdd":
        dataset_type = DatasetType.MMDD
        default_embeddings = 'master_dataset_mmdd_embeddings.npy'
        default_mapping = 'master_dataset_mmdd_embedding_mapping.csv'
    elif args.dataset == "image_chat":
        dataset_type = DatasetType.IMAGECHAT
        default_embeddings = 'master_dataset_imagechat_embeddings.npy'
        default_mapping = 'master_dataset_imagechat_embedding_mapping.csv'
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    # Set default paths if not provided
    embeddings_path = args.embeddings_path or default_embeddings
    mapping_path = args.mapping_path or default_mapping
    
    # Auto-detect device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using embeddings: {embeddings_path}")
    logger.info(f"Using mapping: {mapping_path}")
    logger.info(f"Using device: {device}")
    
    trained_model, label_encoder = train_mlp_model(
        embeddings_path=embeddings_path,
        mapping_path=mapping_path,
        dataset_type=dataset_type,
        hidden_dims=args.hidden_dims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        use_cost_loss=args.use_cost_loss,
        lambda_weight=args.lambda_weight
    )
    
    if trained_model and label_encoder:
        # Save the model and label encoder
        model_save_path = f'mlp_classifier_{args.dataset}.pth'
        encoder_save_path = f'mlp_label_encoder_{args.dataset}.pkl'
        
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'hidden_dims': args.hidden_dims,
            'input_dim': trained_model.network[0].in_features,
            'num_classes': trained_model.network[-1].out_features,
        }, model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        joblib.dump(label_encoder, encoder_save_path)
        logger.info(f"Label encoder saved to {encoder_save_path}")
        print("\nModel and label encoder saved successfully.")
    else:
        logger.error("Model training failed or returned None. Model and encoder not saved.")
        print("\nModel training failed. Check logs for details. Model and encoder not saved.")
