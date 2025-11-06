"""
Training script for GNN model with FEVER dataset

This script trains the GNN model using the FEVER dataset.
It converts FEVER examples to knowledge graphs using LLM extraction,
then trains the GNN to detect contradictions.

Usage:
    python train_gnn.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import json
import pickle

from src.gnn.gnn_model import FactVerificationGNN
from src.gnn.graph_embeddings import GraphEmbedding
from src.llm.llm_client import LLMClient
from src.extraction.entity_extractor import EntityExtractor
from src.graph.graph_builder import GraphBuilder
from src.data.fever_loader import FEVERDatasetLoader
from src.utils.config import Config


class DatasetLoader:
    """
    Loads FEVER dataset and converts to graph format for GNN training
    """

    def __init__(self):
        self.llm_client = LLMClient()
        self.extractor = EntityExtractor(self.llm_client)
        self.fever_loader = FEVERDatasetLoader()
        self.graph_builder = GraphBuilder()
        self.embedding = GraphEmbedding(feature_dim=32)

    def load_training_data(self, n_samples=1000, split='train'):
        """
        Load and process FEVER dataset

        Args:
            n_samples: Number of examples to use for training
            split: Dataset split ('train', 'validation', 'test')

        Returns:
            List of PyTorch Geometric Data objects with labels
        """
        # Load FEVER dataset
        balanced_data = self.fever_loader.get_balanced_dataset(
            n_samples=n_samples,
            split=split
        )

        print(f"\nConverting {len(balanced_data)} examples to graphs...")
        print("This will use LLM extraction - it may take some time...")
        print()

        graph_data_list = []
        failed_count = 0

        for i, example in enumerate(balanced_data):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(balanced_data)} examples... "
                      f"(Success: {len(graph_data_list)}, Failed: {failed_count})")

            try:
                # Convert to graph using LLM extraction
                pyg_data = self.process_article_to_graph(
                    example['text'],
                    example['label_numeric']
                )
                
                # Skip if graph is too small (likely extraction failed)
                if pyg_data.num_nodes < 2:
                    failed_count += 1
                    continue
                    
                graph_data_list.append(pyg_data)

            except Exception as e:
                failed_count += 1
                if (i + 1) % 100 == 0:
                    print(f"  Warning: Failed to process example {i+1}: {e}")
                continue

        print(f"\n✓ Successfully converted {len(graph_data_list)} examples to graphs")
        print(f"  Failed: {failed_count} examples")
        print()

        return graph_data_list

    def process_article_to_graph(self, article_text, has_contradiction):
        """
        Convert article to graph format for GNN

        Args:
            article_text: Raw article text (claim + evidence)
            has_contradiction: Label (0 or 1)

        Returns:
            PyTorch Geometric Data object with label
        """
        # Extract facts using LLM
        extraction = self.extractor.extract_all(article_text)

        # Build graph
        graph = self.graph_builder.build_from_extraction(extraction)

        # Convert to PyTorch Geometric format
        pyg_data = self.embedding.networkx_to_pyg(graph.graph)

        # Add label (graph-level label for binary classification)
        # For graph-level classification, we need to set y as a single value
        pyg_data.y = torch.tensor([has_contradiction], dtype=torch.long)

        return pyg_data


class LLMLogDatasetLoader:
    """
    Loads LLM log graph data with FEVER labels for supervised training
    """
    def __init__(self, max_datapoints=100):
        from src.data.mock_fever_dataset import MockFEVERDataset
        self.mock_dataset = MockFEVERDataset(max_datapoints=max_datapoints)
        self.max_datapoints = max_datapoints

    def load_training_data(self):
        """
        Loads up to max_datapoints graph samples from LLM log with FEVER labels
        Returns:
            List of PyTorch Geometric Data objects with labels
        """
        graph_data_list = self.mock_dataset.get_graph_data_list()
        return graph_data_list


def train_gnn_model(
    train_samples=1000,
    val_samples=200,
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001,
    early_stopping_patience=10
):
    """
    Training loop for GNN with FEVER dataset

    Args:
        train_samples: Number of training examples
        val_samples: Number of validation examples
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        early_stopping_patience: Early stopping patience
    """

    print("=" * 80)
    print("🧠 GNN Training Script with FEVER Dataset")
    print("=" * 80)
    print()

    # Configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nPlease set up your .env file with OpenAI API key")
        return

    # Create models directory
    Path('models').mkdir(exist_ok=True)

    # Load training dataset
    print("📂 Step 1: Loading FEVER Training Dataset...")
    dataset_loader = DatasetLoader()
    
    try:
        training_data = dataset_loader.load_training_data(
            n_samples=train_samples,
            split='train'
        )
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        print("\nMake sure you have installed: pip install datasets")
        return

    if not training_data or len(training_data) < 10:
        print("❌ Not enough training data loaded!")
        print(f"   Loaded: {len(training_data) if training_data else 0} examples")
        print("   Need at least 10 examples to train")
        return

    # Load validation dataset
    print("📂 Step 2: Loading FEVER Validation Dataset...")
    try:
        val_data = dataset_loader.load_training_data(
            n_samples=val_samples,
            split='validation'
        )
    except Exception as e:
        print(f"⚠️  Warning: Could not load validation set: {e}")
        print("   Using train/val split from training data...")
        # Fallback: split training data
        split_idx = int(0.8 * len(training_data))
        val_data = training_data[split_idx:]
        training_data = training_data[:split_idx]

    if not val_data or len(val_data) < 5:
        # Fallback: split training data
        split_idx = int(0.8 * len(training_data))
        val_data = training_data[split_idx:]
        training_data = training_data[:split_idx]

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    print(f"✓ Loaded {len(training_data)} training samples")
    print(f"✓ Loaded {len(val_data)} validation samples")
    print()

    # Initialize model
    print("🏗️  Step 3: Initializing GNN Model...")
    model = FactVerificationGNN(
        input_dim=32,
        hidden_dim=64,
        output_dim=2,  # Binary: contradiction or not
        num_layers=2,
        dropout=0.1,
        use_attention=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("✓ Model initialized")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Training samples: {len(training_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print()

    # Training loop
    print("🚀 Step 4: Training...")
    print()
    
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        for batch in train_loader:
            optimizer.zero_grad()

            out = model(batch)
            # Filter out any samples with y < 0 in the batch
            valid_mask = (batch.y.squeeze() >= 0)
            if valid_mask.sum() == 0:
                continue  # skip batch if no valid targets
            out_valid = out[valid_mask]
            y_valid = batch.y.squeeze()[valid_mask]
            loss = F.nll_loss(out_valid, y_valid)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = out_valid.argmax(dim=1)
            train_correct += (pred == y_valid).sum().item()
            train_total += y_valid.size(0)
            all_train_preds.extend(pred.cpu().numpy())
            all_train_labels.extend(y_valid.cpu().numpy())

        train_acc = train_correct / train_total if train_total > 0 else 0.0
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_losses.append(avg_train_loss)

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                out = model(batch)
                loss = F.nll_loss(out, batch.y.squeeze())

                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y.squeeze()).sum().item()
                val_total += batch.y.size(0)
                
                all_val_preds.extend(pred.cpu().numpy())
                all_val_labels.extend(batch.y.squeeze().cpu().numpy())

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_val_labels, all_val_preds, average='binary', zero_division=0
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save full checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'config': {
                    'input_dim': 32,
                    'hidden_dim': 64,
                    'output_dim': 2,
                    'num_layers': 2,
                    'dropout': 0.1,
                    'use_attention': True
                }
            }
            torch.save(checkpoint, 'models/gnn_fever.pth')
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}")
            print(f"  Best Val Acc: {best_val_acc:.4f}")
            print()

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1} (patience: {early_stopping_patience})")
            break

    elapsed_time = time.time() - start_time

    print("=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"   Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"   Total Training Time: {elapsed_time/60:.2f} minutes")
    print(f"   Model saved to: models/gnn_fever.pth")
    print()
    print("📊 Training Summary:")
    print(f"   - Training samples: {len(training_data)}")
    print(f"   - Validation samples: {len(val_data)}")
    print(f"   - Epochs trained: {epoch+1}")
    print(f"   - Final train accuracy: {train_acc:.4f}")
    print(f"   - Final validation accuracy: {val_acc:.4f}")
    print()


def integrate_gnn_with_demo():
    """
    Example of how to use trained GNN in demo.py
    """
    print("=" * 80)
    print("📖 How to Integrate Trained GNN with Demo")
    print("=" * 80)
    print()

    print("Once you have a trained model, update demo.py:")
    print()
    print("```python")
    print("# After Step 3 (building graph)...")
    print()
    print("# Step 3.5: GNN Analysis (if model exists)")
    print("import os")
    print("if os.path.exists('models/gnn_best.pth'):")
    print("    print('🤖 Step 3.5: Running GNN Analysis...')")
    print("    ")
    print("    # Load trained model")
    print("    from src.gnn.gnn_model import FactVerificationGNN")
    print("    from src.gnn.graph_embeddings import GraphEmbedding")
    print("    ")
    print("    model = FactVerificationGNN(input_dim=32)")
    print("    model.load_state_dict(torch.load('models/gnn_best.pth'))")
    print("    model.eval()")
    print("    ")
    print("    # Convert graph to PyG format")
    print("    embedding = GraphEmbedding(feature_dim=32)")
    print("    pyg_data = embedding.networkx_to_pyg(knowledge_graph.graph)")
    print("    ")
    print("    # Get GNN predictions")
    print("    with torch.no_grad():")
    print("        out = model(pyg_data)")
    print("        gnn_scores = torch.exp(out[:, 1])  # Credibility scores")
    print("    ")
    print("    print(f'✓ GNN analysis complete')")
    print("    print(f'  Average credibility: {gnn_scores.mean():.2f}')")
    print("    print()")
    print("```")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train GNN model with FEVER or LLM log dataset')
    parser.add_argument('--use_llm_log', action='store_true', help='Train using LLM log data instead of FEVER')
    parser.add_argument('--llm_log_limit', type=int, default=100, help='Number of LLM log datapoints to use (default: 100)')
    parser.add_argument('--train_samples', type=int, default=1000,
                        help='Number of training examples (default: 1000)')
    parser.add_argument('--val_samples', type=int, default=200,
                        help='Number of validation examples (default: 200)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')

    args = parser.parse_args()

    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║       GNN Training with FEVER Dataset                     ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    if args.use_llm_log:
        print(f"📂 Step 1: Loading LLM Log Graph Dataset (limit: {args.llm_log_limit})...")
        dataset_loader = LLMLogDatasetLoader(max_datapoints=args.llm_log_limit)
        training_data = dataset_loader.load_training_data()
        val_data = []  # No validation split for LLM log (unsupervised)
        # Train model directly on LLM log data
        train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
        print(f"✓ Loaded {len(training_data)} training samples from LLM log")
        print()
        # Initialize model
        print("🏗️  Step 2: Initializing GNN Model...")
        model = FactVerificationGNN(
            input_dim=32,
            hidden_dim=64,
            output_dim=2,  # Binary: contradiction or not
            num_layers=2,
            dropout=0.1,
            use_attention=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print("✓ Model initialized")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Training samples: {len(training_data)}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.lr}")
        print()
        # Training loop (skip validation)
        print("🚀 Step 3: Training...")
        best_train_acc = 0.0
        train_losses = []
        start_time = time.time()
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            all_train_preds = []
            all_train_labels = []
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch)
                # Filter out any samples with y < 0 in the batch
                valid_mask = (batch.y.squeeze() >= 0)
                if valid_mask.sum() == 0:
                    continue  # skip batch if no valid targets
                out_valid = out[valid_mask]
                y_valid = batch.y.squeeze()[valid_mask]
                loss = F.nll_loss(out_valid, y_valid)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pred = out_valid.argmax(dim=1)
                train_correct += (pred == y_valid).sum().item()
                train_total += y_valid.size(0)
                all_train_preds.extend(pred.cpu().numpy())
                all_train_labels.extend(y_valid.cpu().numpy())
            train_acc = train_correct / train_total if train_total > 0 else 0.0
            avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            train_losses.append(avg_train_loss)
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                torch.save(model.state_dict(), 'models/gnn_llmlog.pth')
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}")
                print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"  Best Train Acc: {best_train_acc:.4f}")
                print()
        elapsed_time = time.time() - start_time
        print("=" * 80)
        print("✅ Training Complete!")
        print("=" * 80)
        print(f"   Best Training Accuracy: {best_train_acc:.4f}")
        print(f"   Total Training Time: {elapsed_time/60:.2f} minutes")
        print(f"   Model saved to: models/gnn_llmlog.pth")
        print()
        print("📊 Training Summary:")
        print(f"   - Training samples: {len(training_data)}")
        print(f"   - Epochs trained: {epoch+1}")
        print(f"   - Final train accuracy: {train_acc:.4f}")
        print()
        exit(0)

    if args.use_llm_log:
        print(f"📂 Step 1: Loading LLM Log Graph Dataset (limit: {args.llm_log_limit})...")
        dataset_loader = LLMLogDatasetLoader(max_datapoints=args.llm_log_limit)
        training_data = dataset_loader.load_training_data()
        val_data = []  # No validation split for LLM log (unsupervised)
        # Proceed directly to model training, skip FEVER loading and LLM extraction
    else:
        print("📂 Step 1: Loading FEVER Training Dataset...")
        dataset_loader = DatasetLoader()
        training_data = dataset_loader.load_training_data(
            n_samples=args.train_samples,
            split='train'
        )
        print("📂 Step 2: Loading FEVER Validation Dataset...")
        try:
            val_data = dataset_loader.load_training_data(
                n_samples=args.val_samples,
                split='validation'
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not load validation set: {e}")
            print("   Using train/val split from training data...")
            # Fallback: split training data
            split_idx = int(0.8 * len(training_data))
            val_data = training_data[split_idx:]
            training_data = training_data[:split_idx]

    # Train model
    train_gnn_model(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        early_stopping_patience=args.patience
    )

    print()
    print("💡 Next Steps:")
    print("   1. Model saved to: models/gnn_fever.pth")
    print("   2. Use the trained model in demo.py for enhanced detection")
    print("   3. Check training metrics above for performance")
    print()
