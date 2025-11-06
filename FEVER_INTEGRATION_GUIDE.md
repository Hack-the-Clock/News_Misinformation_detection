# 🚀 FEVER Dataset Integration Guide

## Overview

The FEVER dataset integration is now complete! You can train your GNN model using the FEVER dataset, which will be processed through your LLM extraction pipeline to create rich knowledge graphs.

---

## 📋 Prerequisites

1. **Install Dependencies**:
```bash
pip install datasets scikit-learn
```

2. **Set Up OpenAI API Key**:
   - Make sure your `.env` file has `OPENAI_API_KEY` set
   - The LLM extraction requires API access

---

## 🚀 Quick Start

### Basic Training

```bash
# Train with default settings (1000 training, 200 validation samples)
python train_gnn.py
```

### Custom Training

```bash
# Train with custom parameters
python train_gnn.py \
    --train_samples 2000 \
    --val_samples 400 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --patience 10
```

### Arguments

- `--train_samples`: Number of training examples (default: 1000)
- `--val_samples`: Number of validation examples (default: 200)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 10)

---

## 📊 What Happens During Training

1. **Load FEVER Dataset**: Downloads from HuggingFace
2. **Balance Dataset**: Creates 50/50 split (contradictions vs non-contradictions)
3. **LLM Extraction**: Each example goes through your LLM extraction pipeline
   - Extracts entities, events, relationships
   - Creates temporal knowledge graphs
4. **Graph Conversion**: Converts NetworkX graphs to PyTorch Geometric format
5. **GNN Training**: Trains the GNN model to detect contradictions
6. **Model Saving**: Saves best model to `models/gnn_fever.pth`

---

## ⏱️ Expected Training Time

- **Small dataset** (1000 samples): ~2-4 hours
  - LLM extraction: ~1-2 hours
  - GNN training: ~30-60 minutes

- **Medium dataset** (5000 samples): ~8-12 hours
  - LLM extraction: ~6-8 hours
  - GNN training: ~1-2 hours

- **Large dataset** (10000+ samples): ~16-24 hours
  - LLM extraction: ~12-18 hours
  - GNN training: ~2-4 hours

**Note**: Training time depends on:
- OpenAI API rate limits
- Number of examples
- LLM extraction complexity
- Your hardware (CPU/GPU)

---

## 📁 Output Files

After training, you'll have:

- `models/gnn_fever.pth`: Trained model checkpoint
  - Contains model weights
  - Training configuration
  - Best validation accuracy

---

## 🎯 Expected Performance

Based on similar GNN fact-checking systems:

- **Validation Accuracy**: 75-85% (expected)
- **Precision/Recall**: ~0.70-0.80
- **F1 Score**: ~0.75-0.80

**Note**: Performance depends on:
- Dataset size
- LLM extraction quality
- Model hyperparameters
- Training duration

---

## 🔧 Troubleshooting

### Issue: "No module named 'datasets'"

**Solution**:
```bash
pip install datasets scikit-learn
```

### Issue: "OpenAI API key not found"

**Solution**:
1. Create `.env` file in project root
2. Add: `OPENAI_API_KEY=sk-your-key-here`

### Issue: "Error loading FEVER dataset"

**Solution**:
- Check internet connection
- Try again (HuggingFace may be temporarily unavailable)
- Verify `datasets` library is installed

### Issue: "Too many API calls" / Rate limiting

**Solution**:
- Reduce `--train_samples` (start with 500)
- Add delays between API calls (modify `fever_loader.py`)
- Use OpenAI API with higher rate limits

### Issue: "Out of memory"

**Solution**:
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--train_samples`
- Use smaller model (reduce `hidden_dim` in `train_gnn.py`)

---

## 📈 Monitoring Training

The training script prints progress every 5 epochs:

```
Epoch 5/50
  Train Loss: 0.6234 | Train Acc: 0.7123
  Val Loss: 0.6543 | Val Acc: 0.6890
  Val Precision: 0.7012 | Val Recall: 0.6789 | Val F1: 0.6898
  Best Val Acc: 0.6890
```

**Key Metrics**:
- **Train/Val Loss**: Should decrease over time
- **Train/Val Acc**: Should increase over time
- **Precision/Recall/F1**: Balance between precision and recall
- **Best Val Acc**: Best validation accuracy so far

---

## 🎓 Next Steps

After training:

1. **Use in Demo**: Integrate trained model with `demo.py`
2. **Evaluate**: Test on test set (if available)
3. **Fine-tune**: Adjust hyperparameters based on results
4. **Combine**: Use with symbolic rules for neurosymbolic approach

---

## 💡 Tips

1. **Start Small**: Begin with 500-1000 samples to test the pipeline
2. **Monitor Costs**: LLM extraction uses OpenAI API (costs apply)
3. **Check Progress**: Watch for early stopping (model may converge early)
4. **Save Checkpoints**: Model is saved automatically (best validation accuracy)
5. **Experiment**: Try different hyperparameters for better performance

---

## 📚 Files Created

- `src/data/fever_loader.py`: FEVER dataset loader
- `src/data/__init__.py`: Data module init
- Updated `train_gnn.py`: Training script with FEVER integration
- Updated `requirements.txt`: Added datasets and scikit-learn
- Updated `src/gnn/gnn_model.py`: Graph-level classification support

---

## ✅ Integration Complete!

The FEVER dataset integration is ready to use. Run `python train_gnn.py` to start training!

For questions or issues, check:
- `DATASET_INTEGRATION_PLAN.md`: Detailed integration plan
- `FEVER_VS_LIAR_COMPARISON.md`: Why FEVER was chosen
- `LLM_GRAPH_ENHANCEMENT_ANALYSIS.md`: How LLM improves graphs

