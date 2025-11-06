# GNN Training Dataset Integration Plan

## Selected Dataset: FEVER (Fact Extraction and VERification)

### Why FEVER?
1. **Size**: 185,445 claims with evidence sentences
2. **Language**: English (matches your current system)
3. **Labels**: SUPPORTED, REFUTED, NOT ENOUGH INFO
4. **Accessibility**: Available on HuggingFace
5. **Quality**: Human-annotated by Stanford NLP researchers
6. **Relevance**: Direct contradiction detection (REFUTED claims)

---

## Implementation Plan

### Phase 1: Dataset Setup (2-3 hours)

#### Step 1.1: Install Dataset Dependencies
```bash
pip install datasets
pip install scikit-learn
```

#### Step 1.2: Create Dataset Loader Module
**File**: `src/data/fever_loader.py`

```python
from datasets import load_dataset
from typing import List, Dict, Tuple
import random

class FEVERDatasetLoader:
    """
    Loads and preprocesses FEVER dataset for GNN training
    """

    def __init__(self):
        self.dataset = None

    def load(self, split='train', limit=None):
        """
        Load FEVER dataset from HuggingFace

        Args:
            split: 'train', 'validation', or 'test'
            limit: Maximum number of examples to load (None for all)

        Returns:
            List of (text, label) tuples
        """
        print(f"Loading FEVER dataset ({split} split)...")
        self.dataset = load_dataset('fever', 'v1.0', split=split)

        if limit:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))

        print(f"✓ Loaded {len(self.dataset)} examples")
        return self.dataset

    def prepare_for_gnn(self, example):
        """
        Convert FEVER example to format suitable for knowledge graph

        FEVER format:
        {
            'claim': "The Rodney King riots took place in the most populous county in the USA.",
            'label': 'SUPPORTS',  # or 'REFUTES' or 'NOT ENOUGH INFO'
            'evidence': [
                ['Rodney_King', 0, 'The 1992 Los Angeles riots...'],
                ...
            ]
        }

        Returns:
            {
                'text': combined claim + evidence,
                'has_contradiction': bool (True if REFUTES),
                'label_numeric': 0 or 1
            }
        """
        claim = example['claim']
        label = example['label']

        # Extract evidence sentences
        evidence_texts = []
        if 'evidence' in example and example['evidence']:
            for evidence_set in example['evidence']:
                if evidence_set and len(evidence_set) > 0:
                    # evidence_set format: [doc_id, sentence_id, text]
                    for ev in evidence_set:
                        if len(ev) >= 3:
                            evidence_texts.append(ev[2])

        # Combine claim with evidence
        combined_text = claim
        if evidence_texts:
            combined_text += " " + " ".join(evidence_texts[:3])  # Use top 3 evidence

        # Convert label to binary (contradiction detection)
        has_contradiction = (label == 'REFUTES')
        label_numeric = 1 if has_contradiction else 0

        return {
            'text': combined_text,
            'has_contradiction': has_contradiction,
            'label_numeric': label_numeric,
            'original_label': label
        }

    def get_balanced_dataset(self, n_samples=1000):
        """
        Get balanced dataset with equal contradictions and non-contradictions

        Args:
            n_samples: Total number of samples (will be split 50/50)

        Returns:
            List of prepared examples
        """
        if not self.dataset:
            self.load(split='train')

        contradictions = []
        non_contradictions = []

        print("Preparing balanced dataset...")
        for example in self.dataset:
            prepared = self.prepare_for_gnn(example)

            if prepared['has_contradiction']:
                contradictions.append(prepared)
            else:
                non_contradictions.append(prepared)

            # Stop when we have enough of each
            if len(contradictions) >= n_samples // 2 and len(non_contradictions) >= n_samples // 2:
                break

        # Balance the dataset
        balanced = contradictions[:n_samples // 2] + non_contradictions[:n_samples // 2]
        random.shuffle(balanced)

        print(f"✓ Created balanced dataset:")
        print(f"  - Contradictions: {len(contradictions[:n_samples // 2])}")
        print(f"  - Non-contradictions: {len(non_contradictions[:n_samples // 2])}")
        print(f"  - Total: {len(balanced)}")

        return balanced
```

#### Step 1.3: Update Training Script
**File**: `train_gnn.py` (update DatasetLoader class)

Replace the placeholder `DatasetLoader` with:

```python
from src.data.fever_loader import FEVERDatasetLoader

class DatasetLoader:
    """
    Loads FEVER dataset and converts to graph format
    """

    def __init__(self):
        self.llm_client = LLMClient()
        self.extractor = EntityExtractor(self.llm_client)
        self.fever_loader = FEVERDatasetLoader()

    def load_training_data(self, n_samples=1000):
        """
        Load and process FEVER dataset

        Args:
            n_samples: Number of examples to use for training

        Returns:
            List of (graph_data, label) tuples
        """
        # Load FEVER dataset
        balanced_data = self.fever_loader.get_balanced_dataset(n_samples=n_samples)

        print(f"Converting {len(balanced_data)} examples to graphs...")
        graph_data_list = []

        for i, example in enumerate(balanced_data):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(balanced_data)} examples...")

            try:
                # Convert to graph
                pyg_data = self.process_article_to_graph(
                    example['text'],
                    example['label_numeric']
                )
                graph_data_list.append(pyg_data)

            except Exception as e:
                print(f"  Warning: Failed to process example {i}: {e}")
                continue

        print(f"✓ Successfully converted {len(graph_data_list)} examples to graphs")
        return graph_data_list
```

---

### Phase 2: GNN Training Pipeline (3-4 hours)

#### Step 2.1: Create Training Configuration
**File**: `configs/gnn_training_config.yaml`

```yaml
# GNN Training Configuration

dataset:
  name: "FEVER"
  train_samples: 5000
  val_samples: 1000
  test_samples: 1000
  balance: true

model:
  input_dim: 32
  hidden_dim: 64
  output_dim: 2  # Binary classification
  num_layers: 2
  dropout: 0.1
  use_attention: true

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10

output:
  model_path: "models/gnn_fever.pth"
  log_path: "outputs/training_logs/"
  checkpoint_interval: 10
```

#### Step 2.2: Enhanced Training Script with Metrics
**File**: `train_gnn_fever.py`

```python
"""
GNN Training Script with FEVER Dataset
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path

from src.gnn.gnn_model import FactVerificationGNN
from src.data.fever_loader import FEVERDatasetLoader
from train_gnn import DatasetLoader

class GNNTrainer:
    """
    Trainer for GNN model with FEVER dataset
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def initialize_model(self):
        """Initialize GNN model and optimizer"""
        self.model = FactVerificationGNN(
            input_dim=self.config['model']['input_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            output_dim=self.config['model']['output_dim'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout'],
            use_attention=self.config['model']['use_attention']
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )

        print(f"✓ Model initialized")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            self.optimizer.zero_grad()

            out = self.model(batch)
            loss = F.nll_loss(out, batch.y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                out = self.model(batch)
                loss = F.nll_loss(out, batch.y)

                total_loss += loss.item()
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )

        return avg_loss, accuracy, precision, recall, f1

    def train(self, train_loader, val_loader):
        """Full training loop"""
        best_val_acc = 0.0
        patience_counter = 0
        patience = self.config['training']['early_stopping_patience']

        print(f"\n🚀 Starting Training...")
        print(f"   Epochs: {self.config['training']['num_epochs']}")
        print(f"   Batch Size: {self.config['training']['batch_size']}")
        print(f"   Learning Rate: {self.config['training']['learning_rate']}")
        print()

        for epoch in range(self.config['training']['num_epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(self.config['output']['model_path'])
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                print(f"  Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}")
                print(f"  Best Val Acc: {best_val_acc:.4f}")
                print()

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"✅ Training Complete!")
        print(f"   Best Validation Accuracy: {best_val_acc:.4f}")
        print()

        return best_val_acc

    def save_model(self, path):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

    def plot_training_curves(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curve
        ax2.plot(self.val_accuracies, label='Val Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        output_dir = Path(self.config['output']['log_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'training_curves.png', dpi=150)
        print(f"✓ Training curves saved to {output_dir / 'training_curves.png'}")


def main():
    print("=" * 80)
    print("🧠 GNN Training with FEVER Dataset")
    print("=" * 80)
    print()

    # Configuration
    config = {
        'dataset': {
            'name': 'FEVER',
            'train_samples': 5000,
            'val_samples': 1000,
        },
        'model': {
            'input_dim': 32,
            'hidden_dim': 64,
            'output_dim': 2,
            'num_layers': 2,
            'dropout': 0.1,
            'use_attention': True
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 50,
            'learning_rate': 0.001,
            'early_stopping_patience': 10
        },
        'output': {
            'model_path': 'models/gnn_fever.pth',
            'log_path': 'outputs/training_logs/'
        }
    }

    # Load dataset
    print("📂 Step 1: Loading FEVER Dataset...")
    dataset_loader = DatasetLoader()

    print(f"   Loading {config['dataset']['train_samples']} training examples...")
    train_data = dataset_loader.load_training_data(n_samples=config['dataset']['train_samples'])

    print(f"   Loading {config['dataset']['val_samples']} validation examples...")
    val_data = dataset_loader.load_training_data(n_samples=config['dataset']['val_samples'])

    if not train_data or not val_data:
        print("❌ Failed to load dataset")
        return

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['training']['batch_size'])

    print(f"✓ Dataset loaded")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print()

    # Initialize trainer
    print("🏗️  Step 2: Initializing GNN Model...")
    trainer = GNNTrainer(config)
    trainer.initialize_model()
    print()

    # Train
    best_acc = trainer.train(train_loader, val_loader)

    # Plot results
    trainer.plot_training_curves()

    print()
    print("=" * 80)
    print("✅ Training Complete!")
    print(f"   Best Validation Accuracy: {best_acc:.4f}")
    print(f"   Model saved to: {config['output']['model_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
```

---

### Phase 3: Integration with Demo (1-2 hours)

#### Step 3.1: Update Demo to Use Trained GNN
**File**: `demo_with_gnn.py`

```python
"""
Demo with trained GNN model integration
"""

import os
import torch
from demo import main as demo_main

def load_trained_gnn():
    """Load trained GNN model if available"""
    model_path = 'models/gnn_fever.pth'

    if not os.path.exists(model_path):
        print("⚠️  No trained GNN model found")
        print(f"   Expected path: {model_path}")
        print(f"   Run: python train_gnn_fever.py")
        return None

    print("🤖 Loading trained GNN model...")

    from src.gnn.gnn_model import FactVerificationGNN

    # Load checkpoint
    checkpoint = torch.load(model_path)
    config = checkpoint['config']

    # Initialize model
    model = FactVerificationGNN(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        use_attention=config['model']['use_attention']
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ GNN model loaded")
    print(f"  Validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    print()

    return model

def main():
    # Load trained GNN
    gnn_model = load_trained_gnn()

    # Run demo with GNN integration
    demo_main(gnn_model=gnn_model)

if __name__ == "__main__":
    main()
```

---

### Phase 4: Evaluation and Metrics (1 hour)

#### Step 4.1: Evaluation Script
**File**: `evaluate_gnn.py`

```python
"""
Evaluate trained GNN model on test set
"""

import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from train_gnn_fever import DatasetLoader
from src.gnn.gnn_model import FactVerificationGNN

def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            probs = torch.exp(out)  # Convert log probabilities to probabilities
            pred = out.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of contradiction

    # Classification report
    print("\n" + "=" * 80)
    print("📊 Classification Report")
    print("=" * 80)
    print(classification_report(
        all_labels,
        all_preds,
        target_names=['No Contradiction', 'Contradiction']
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('outputs/training_logs/confusion_matrix.png', dpi=150)
    print("✓ Confusion matrix saved to outputs/training_logs/confusion_matrix.png")

    return all_preds, all_labels, all_probs


def main():
    print("=" * 80)
    print("🧪 GNN Model Evaluation")
    print("=" * 80)
    print()

    # Load model
    model_path = 'models/gnn_fever.pth'
    checkpoint = torch.load(model_path)
    config = checkpoint['config']

    model = FactVerificationGNN(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        use_attention=config['model']['use_attention']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("✓ Model loaded")
    print()

    # Load test data
    print("📂 Loading test dataset...")
    dataset_loader = DatasetLoader()
    test_data = dataset_loader.load_training_data(n_samples=1000)
    test_loader = DataLoader(test_data, batch_size=32)

    print(f"✓ Loaded {len(test_data)} test examples")
    print()

    # Evaluate
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
```

---

## Summary: Implementation Timeline

| Phase | Task | Duration | Output |
|-------|------|----------|--------|
| 1 | Dataset Setup | 2-3 hours | `src/data/fever_loader.py` |
| 2 | Training Pipeline | 3-4 hours | `train_gnn_fever.py`, trained model |
| 3 | Demo Integration | 1-2 hours | `demo_with_gnn.py` |
| 4 | Evaluation | 1 hour | `evaluate_gnn.py`, metrics |

**Total Estimated Time**: 7-10 hours

---

## Expected Results

After training on FEVER dataset:
- **Validation Accuracy**: 75-85% (expected)
- **Precision/Recall**: ~0.70-0.80
- **Model File**: `models/gnn_fever.pth` (~5MB)
- **Training Curves**: Loss and accuracy plots
- **Confusion Matrix**: Visual performance analysis

---

## Next Steps After Training

1. **Integrate with Demo**: Use `demo_with_gnn.py` for enhanced detection
2. **Compare Performance**: Symbolic rules vs GNN vs Hybrid (ensemble)
3. **Fine-tune**: Adjust hyperparameters based on validation results
4. **Documentation**: Update README with GNN training instructions
5. **GitHub**: Commit trained model and training scripts

---

## Questions to Consider

1. **Do you want to train on a smaller sample first** (1000 examples) to test the pipeline?
2. **Should I proceed with creating these files** now?
3. **Do you prefer FEVER over DeFaktS** given the language compatibility?

Let me know if you'd like me to proceed with implementation!
