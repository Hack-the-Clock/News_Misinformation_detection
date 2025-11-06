# Quick Command Reference

## 🚀 Most Important Commands

### Test Complete Neurosymbolic System (When API Available)
```bash
python test_full_pipeline.py --num_samples 10
```
**What it does**: Tests GNN + Symbolic Logic together on FEVER dataset
**Time**: ~5 minutes (30s delays between samples)
**Output**: Comprehensive metrics, agreement analysis, detailed examples

---

## 📊 Testing Commands

### Test GNN Model (No API Calls)
```bash
python test_gnn_demo.py
```
**What it does**: Verifies GNN model works with mock graphs
**Time**: <1 minute
**Use when**: Want to check if GNN is working

### Test GNN on FEVER Dataset
```bash
python test_demo_fever.py --num_samples 10
```
**What it does**: Evaluates GNN predictions vs ground truth
**Time**: ~5 minutes (30s delays)
**Use when**: Want to evaluate GNN accuracy

### Test Full Demo
```bash
python demo.py
```
**What it does**: Complete pipeline with visualizations
**Time**: ~1 minute
**Use when**: Want to see end-to-end system in action

---

## 🎓 Training Commands

### Train GNN with Mock Data (Current)
```bash
python train_gnn.py --use_llm_log --llm_log_limit 100 --epochs 30
```
**What it does**: Trains on existing mock extractions
**Time**: ~5 minutes
**Output**: models/gnn_llmlog.pth

### Generate More Mock Data
```bash
python generate_mock_llm_data.py --num_samples 200
python train_gnn.py --use_llm_log --llm_log_limit 200 --epochs 50
```
**What it does**: Creates more training data without API calls
**Time**: <1 minute + 10 minutes training

### Train with Real LLM Data (When API Available)
```bash
python train_gnn.py --train_samples 500 --val_samples 100 --epochs 50
```
**What it does**: Trains on real LLM extractions
**Time**: ~3 hours (includes LLM calls + training)
**Expected**: 70-80% accuracy

---

## 🔍 Analysis Commands

### Inspect Training Data Quality
```bash
python inspect_llm_extractions.py
```
**What it does**: Analyzes mock extraction statistics
**Output**: Entity/event/relation counts, patterns

### Check GNN Model Info
```bash
source venv/bin/activate
python -c "
import torch
from src.gnn.gnn_model import FactVerificationGNN
model = FactVerificationGNN(32, 64, 2, 2, 0.1, True)
model.load_state_dict(torch.load('models/gnn_llmlog.pth'))
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```
**Output**: Model statistics

---

## 📂 Data Commands

### Download FEVER Dataset (Already Done)
```bash
# Train set
wget https://fever.ai/download/fever/train.jsonl -P data/fever/

# Dev set
wget https://fever.ai/download/fever/shared_task_dev.jsonl -P data/fever/dev.jsonl
```

### Check Dataset Stats
```bash
python -c "
from src.data.fever_loader import FEVERDatasetLoader
loader = FEVERDatasetLoader()
samples = loader.get_balanced_dataset(n_samples=100, split='train')
print(f'Total samples: {len(samples)}')
print(f'SUPPORTS: {sum(1 for s in samples if s[\"label_numeric\"] == 0)}')
print(f'REFUTES: {sum(1 for s in samples if s[\"label_numeric\"] == 1)}')
"
```

---

## 🐛 Debugging Commands

### Check Environment
```bash
source venv/bin/activate
python -c "
import torch
import torch_geometric
print(f'PyTorch: {torch.__version__}')
print(f'PyTorch Geometric: {torch_geometric.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Test LLM Connection (When API Available)
```bash
python -c "
from src.llm.llm_client import LLMClient
client = LLMClient()
response = client.extract_structured_data(
    'Test article',
    'Extract entities',
    'json'
)
print('API working!' if response else 'API failed')
"
```

### Check API Rate Limits
```bash
python -c "
from src.llm.llm_client import LLMClient
import openai
try:
    client = LLMClient()
    client.client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': 'hi'}],
        max_tokens=5
    )
    print('✓ API available')
except Exception as e:
    print(f'✗ API error: {e}')
"
```

---

## 📈 Performance Comparison

### Run All Tests (When API Available)
```bash
# 1. Test GNN only
python test_demo_fever.py --num_samples 10 > results_gnn.txt

# 2. Test full pipeline
python test_full_pipeline.py --num_samples 10 > results_full.txt

# 3. Compare
echo "=== GNN Only ==="
grep "Accuracy:" results_gnn.txt
echo "=== Full Pipeline ==="
grep "Accuracy:" results_full.txt
```

---

## 🔧 Maintenance Commands

### Clean Up Old Models
```bash
# Backup current model
cp models/gnn_llmlog.pth models/gnn_llmlog_backup_$(date +%Y%m%d).pth

# Clean old backups (keep last 5)
ls -t models/gnn_llmlog_backup_*.pth | tail -n +6 | xargs rm -f
```

### Reset Mock Data
```bash
# Backup current log
cp logs/llm_calls.log logs/llm_calls_backup.log

# Generate fresh mock data
python generate_mock_llm_data.py --num_samples 100

# Retrain
python train_gnn.py --use_llm_log --llm_log_limit 100 --epochs 30
```

---

## 🎯 Recommended Workflow

### Right Now (API Limited)
```bash
# 1. Verify GNN is working
python test_gnn_demo.py

# 2. Check data quality
python inspect_llm_extractions.py

# 3. Generate more mock data if needed
python generate_mock_llm_data.py --num_samples 200
python train_gnn.py --use_llm_log --llm_log_limit 200 --epochs 50
```

### When API Resets
```bash
# 1. Test complete system
python test_full_pipeline.py --num_samples 10

# 2. Run full demo
python demo.py

# 3. Generate real training data
python train_gnn.py --train_samples 500 --val_samples 100 --epochs 50

# 4. Re-evaluate
python test_full_pipeline.py --num_samples 20
```

---

## 📚 Documentation Files

- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Complete project overview
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide
- **[TRAINING_SUMMARY.md](TRAINING_SUMMARY.md)** - GNN training details
- **[TEST_RESULTS.md](TEST_RESULTS.md)** - GNN test results
- **[FULL_PIPELINE_TEST_RESULTS.md](FULL_PIPELINE_TEST_RESULTS.md)** - Complete system test
- **[COMMANDS.md](COMMANDS.md)** - This file

---

## ⚡ Quick Tips

### Activate Virtual Environment (Always First!)
```bash
source venv/bin/activate
```

### Check If Model Exists
```bash
ls -lh models/gnn_llmlog.pth
# Should show: 374K file
```

### Check If Mock Data Exists
```bash
wc -l logs/llm_calls.log
# Should show: 100 lines
```

### Check API Key
```bash
echo $OPENAI_API_KEY
# Should show: sk-proj-...
```

### Monitor API Usage
```bash
# Check rate limit status in error messages
# "Used 200/200" = daily limit hit
# Wait until tomorrow or add payment method
```

---

## 🎉 Success Indicators

### System is Working When:
- ✅ `test_gnn_demo.py` shows predictions
- ✅ `models/gnn_llmlog.pth` exists (374 KB)
- ✅ `logs/llm_calls.log` has 100+ lines
- ✅ `demo.py` shows "Step 3.5: GNN Analysis"
- ✅ No "model not found" errors

### Ready for Production When:
- ✅ GNN accuracy > 70% on FEVER
- ✅ Combined accuracy > 75%
- ✅ Agreement rate > 70%
- ✅ Both SUPPORTS and REFUTES predicted correctly
- ✅ Confidence scores correlate with accuracy

---

## 🆘 Common Issues

### "Model not found"
```bash
python train_gnn.py --use_llm_log --llm_log_limit 100 --epochs 30
```

### "No LLM log data"
```bash
python generate_mock_llm_data.py --num_samples 100
```

### "Rate limit exceeded"
```bash
# Wait for reset (daily at midnight PST)
# Or add payment method at platform.openai.com
```

### "Module not found"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Static graphs not supported"
```bash
# This happens when graph has no edges
# LLM needs to extract relations
# Check if API is working or use better mock data
```

---

## 📞 Next Steps

**Immediate**: Run when API resets
```bash
python test_full_pipeline.py --num_samples 10
```

**This Week**: Generate real training data
```bash
python train_gnn.py --train_samples 500 --epochs 50
```

**This Month**: Achieve production performance
- 70-80% GNN accuracy
- 75-85% combined accuracy
- Deploy system
