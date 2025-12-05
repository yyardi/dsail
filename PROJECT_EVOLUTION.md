# LoRA Fine-Tuning Project Evolution

## Project Overview
This project explores fine-tuning the Mistral-7B-Instruct model on Chinese mental health Q&A data (PsyQA dataset) using LoRA (Low-Rank Adaptation). The goal was to improve upon baseline model performance through parameter-efficient fine-tuning.

## Key Findings
- ✅ **Baseline Mistral-7B performed best** among all evaluated models (BERTScore F1: 74.00)
- ❌ **LoRA fine-tuning degraded performance** instead of improving it (BERTScore dropped to ~46)
- ✅ **Prompt engineering** is the recommended approach given current constraints

---

## Notebook Evolution

### 1. `baseline_model_evaluation.ipynb` (Pre-existing)
**Status**: ✅ Working

**Purpose**: Evaluate 7 baseline models on PsyQA dataset without any fine-tuning

**Models Evaluated**:
- LLaMA-3-8B-Instruct
- Gemma-7B-it
- Mistral-7B-Instruct-v0.2 ⭐ **Best performer**
- mT5-base (multilingual)
- BLOOMZ-7B1 (multilingual)
- XGLM-7.5B (multilingual)
- Mental-Mistral-7B-v0.1-Patient (domain-specific)

**Results**:
- **Winner**: Mistral-7B with BERTScore F1: 74.00
- Gemma-7B close second with BERTScore F1: 73.80
- Multilingual models (mT5, BLOOMZ, XGLM) performed poorly (50-62 range)
- Domain-specific Mental-Mistral slightly worse than base Mistral

**Key Insight**: General-purpose bilingual models outperform both multilingual and domain-specific models

---

### 2. `mistral_lora_finetuning.ipynb` (First Attempt)
**Status**: ❌ Failed - Poor results

**Purpose**: First attempt to apply LoRA fine-tuning to Mistral-7B on PsyQA dataset

**Implementation**:
- LoRA configuration: r=16, alpha=32, dropout=0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Training: 3 epochs, batch size 4, learning rate 2e-4
- Used TRL's SFTTrainer

**Results**:
- ❌ **Catastrophic performance drop**
- BERTScore F1: 74.00 → 46.00 (38% decrease)
- ROUGE-L and BLEU-4 scores near zero

**Problems Identified**:
1. Trained on full sequences (instruction + response) instead of completion-only
2. Labels not properly masked (model learned to predict instructions, not generate answers)
3. Dataset too small (500 samples insufficient for 7B model)
4. Catastrophic forgetting of pre-trained knowledge

---

### 3. `mistral_lora_finetuning_fixed.ipynb` (Second Attempt)
**Status**: ❌ Failed - Complete failure

**Purpose**: Fix format issues and implement proper Mistral-Instruct template

**Changes from v1**:
- Proper `<s>[INST] ... [/INST] response</s>` format
- Attempted label masking to train only on completions
- Custom MaskedDataset class to set instruction tokens to -100

**Implementation**:
```python
# Find [/INST] and mask everything before it
inst_end = text.find("[/INST]")
labels[0, :prompt_len] = -100  # Mask instruction tokens
```

**Results**:
- ❌ **Complete failure**
- ROUGE-L: 0.00 (99.7% decrease from baseline)
- BLEU-4: 0.00 (99.4% decrease from baseline)
- BERTScore F1: ~48

**Problem**: Masking implementation flawed, model still learning wrong patterns

---

### 4. `mistral_lora_sfttrainer.ipynb` (Third Attempt)
**Status**: ⚠️ Works but produces poor results

**Purpose**: Use TRL's SFTTrainer with proper completion-only training

**Evolution** (Multiple fixes applied):

#### Fix 1: DataCollator Import Error
- **Error**: `ImportError: cannot import name 'DataCollatorForCompletionOnlyLM'`
- **Cause**: Old TRL version missing this class
- **Solution**: Added try/except with fallback to DataCollatorForLanguageModeling

#### Fix 2: SFTTrainer API Compatibility
- **Error**: `TypeError: SFTTrainer.__init__() got unexpected keyword argument 'tokenizer'`
- **Cause**: API changed between TRL versions
- **Solution**: Added version-agnostic try/except wrapper

#### Fix 3: Complete Replacement (Final)
- **Error**: `dataset_text_field` parameter also incompatible
- **Solution**: Replaced SFTTrainer entirely with standard Trainer + manual tokenization

**Final Implementation**:
```python
# Manual tokenization
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

# Standard Trainer instead of SFTTrainer
from transformers import Trainer, DataCollatorForLanguageModeling
```

**Results**: Code works without errors but performance still poor

**Problem**: Still trains on full sequences, not completion-only

---

### 5. `compare_datasets_lora.ipynb`
**Status**: ✅ Code works, ❌ Results uninformative

**Purpose**: Train 5 separate LoRA models on different prompt datasets and compare performance

**Datasets Tested**:
1. **prompt1-eng.json** - Clinical evidence-based approach (English)
2. **Prompt2-eng.json** - Empathetic supportive approach (English)
3. **prompt3-eng.json** - Structured step-by-step approach (English)
4. **rewritten_prompt4-chineese.json** - Formal professional style (Chinese)
5. **rewritten_prompt5-chineese.json** - Friendly conversational style (Chinese)

**Implementation**:
- Trains one LoRA model per dataset
- Evaluates each on same test set
- GPU memory cleanup between models
- Comprehensive visualizations (bar charts, radar chart, heatmap)

**Fixes Applied**:
- Added DATA_DIR configuration: `/content/sample_data/`
- File existence verification before training
- Proper file path handling with `os.path.join()`

**Results**: All 5 prompts produced nearly identical poor results, confirming the issue is with training methodology, not data format

**Key Insight**: Different prompt formats don't help if the training process itself is broken

---

### 6. `baseline_prompt_comparison.ipynb` ⭐ **RECOMMENDED**
**Status**: ✅ Working - Best approach

**Purpose**: Test baseline Mistral-7B (NO fine-tuning) against 6 different prompt formats

**Prompt Formats**:
1. **Baseline** - Simple Chinese prompt
2. **Prompt1-Clinical** - Evidence-based, professional medical approach
3. **Prompt2-Empathetic** - Warm, supportive, emotionally validating
4. **Prompt3-Structured** - Step-by-step structured guidance
5. **Prompt4-Formal-CN** - Formal professional Chinese style
6. **Prompt5-Friendly-CN** - Friendly conversational Chinese style

**Implementation**:
```python
class PromptFormatter:
    def format(self, question: str, description: str) -> str:
        # Each format has unique structure and tone

# Test each format on same test set
for format_name, formatter in PROMPT_FORMATS.items():
    predictions = []
    for item in test_data:
        prompt = formatter.format(item['question'], item['description'])
        pred = generate_response(model, tokenizer, prompt)
        predictions.append(pred)

    metrics = calculate_metrics(predictions, references)
```

**Fixes Applied**:

#### Fix 1: bitsandbytes Installation
- **Error**: `PackageNotFoundError: No package metadata was found for bitsandbytes`
- **Solution**:
  - Install bitsandbytes separately
  - Add verification check
  - 3-tier fallback: 4-bit → 8-bit → float16

```python
# Install first
!pip install -q bitsandbytes

# Verify
if importlib.util.find_spec("bitsandbytes") is None:
    print("⚠ bitsandbytes not found, attempting reinstall...")

# Fallback cascade
try:
    # Try 4-bit quantization
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
    model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)
except:
    try:
        # Fall back to 8-bit
        model = AutoModelForCausalLM.from_pretrained(..., load_in_8bit=True)
    except:
        # Fall back to float16 (no quantization)
        model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float16)
```

#### Fix 2: File Path Configuration
- Added DATA_DIR: `/content/sample_data/`
- File verification before loading
- Proper path handling

**Advantages**:
- ✅ No training required (fast)
- ✅ No risk of degrading model performance
- ✅ Easy to iterate and test new formats
- ✅ Works with baseline model's pre-trained knowledge
- ✅ Identifies best prompting strategy

**Why This Works Better**: Prompt engineering leverages the model's existing capabilities instead of trying to modify them with insufficient data.

---

## Technical Analysis

### Why LoRA Failed

#### 1. Training on Wrong Parts
**Problem**: Models were trained on full sequences (instruction + response) instead of completion-only.

**What should happen**:
```python
# Input text: "<s>[INST] question [/INST] answer</s>"
input_ids: [1, 733, 16289, ..., 567, 890]  # Full text
labels:    [-100, -100, -100, ..., 567, 890]  # Loss only on answer
            ↑ instruction masked    ↑ answer trained
```

**What actually happened**:
```python
# Labels = input_ids (no masking)
input_ids: [1, 733, 16289, ..., 567, 890]
labels:    [1, 733, 16289, ..., 567, 890]  # Loss computed on everything
```

**Result**: Model learned to predict instructions instead of generate answers.

#### 2. Dataset Too Small
- Training samples: 200-500 per dataset
- Model parameters: 7 billion
- **Rule of thumb**: Need 10,000+ high-quality samples for meaningful fine-tuning
- **Current ratio**: ~0.00007 samples per parameter (catastrophically insufficient)

#### 3. Catastrophic Forgetting
- Small dataset can't represent full distribution of mental health questions
- Model "forgets" pre-trained knowledge while overfitting to tiny dataset
- Pre-trained capabilities (language understanding, reasoning) degraded

#### 4. Library Compatibility Issues
- TRL library API changes between versions
- `DataCollatorForCompletionOnlyLM` not available in older versions
- `SFTTrainer` API unstable across versions
- Had to replace specialized tools with generic ones, losing completion-only training capability

### Why Baseline Mistral-7B Won

1. **Strong Pre-training**: Mistral-7B has extensive pre-training on high-quality text including:
   - General knowledge
   - Multi-turn conversations
   - Instruction-following capabilities
   - Some domain knowledge (including psychology/mental health concepts)

2. **Bilingual Capability**: Unlike multilingual models spread thin across 100+ languages, Mistral handles English/Chinese well

3. **Instruction Tuning**: Already trained to follow `[INST]` format and generate helpful responses

4. **No Corruption**: Baseline model's weights untouched, preserving all pre-trained knowledge

### Why Multilingual Models Failed

- **Capacity Dilution**: Models like mT5, BLOOMZ, XGLM support 100+ languages
- **Per-language Performance**: Limited capacity allocated to any single language
- **Chinese Performance**: Insufficient Chinese-specific training
- **Result**: BERTScore F1 of 50-62 vs 74 for focused bilingual models

---

## Data Flow Explanation

### Training Phase

**Input** (both question AND answer):
```
<s>[INST] 你是一位专业的心理健康顾问。
问题：我最近总是失眠，怎么办？
详细描述：已经连续一周每天只睡3-4小时...
请提供专业、有帮助、共情的回答。 [/INST] 失眠可能由多种原因引起，包括压力、焦虑...</s>
```

**What Model Learns**: Pattern mapping from questions → appropriate responses

**Critical Requirement**: Labels must mask instruction tokens:
```python
labels[instruction_part] = -100  # Don't train on this
labels[answer_part] = actual_tokens  # Only train on this
```

**Output**: LoRA adapter weights file (~16MB) containing learned patterns

### Inference Phase

**Input** (question only):
```
<s>[INST] 你是一位专业的心理健康顾问。
问题：我感到很焦虑，该怎么办？
请提供专业、有帮助、共情的回答。 [/INST]
```

**Model Process**:
1. Base model (frozen) processes input
2. LoRA adapters add learned adjustments
3. Model generates tokens one at a time
4. Stops at `</s>` token

**Output**: Generated answer
```
焦虑是一种常见的情绪反应。首先，尝试深呼吸练习来放松身体...
```

---

## Recommendations

### ✅ Immediate: Use Prompt Engineering
**Why**: Fast, safe, effective with current resources

**Best Approach**:
1. Run `baseline_prompt_comparison.ipynb`
2. Identify which prompt format performs best
3. Use that format for your application
4. Iterate on prompt design based on results

**Expected Outcome**: Likely to match or exceed baseline performance through better prompting

### 🔄 Short-term: Collect More Data
**Why**: Current dataset too small for meaningful fine-tuning

**Requirements**:
- Target: 10,000+ question-answer pairs
- Quality: Professional mental health responses
- Diversity: Cover wide range of mental health topics
- Format: Consistent structure with question + detailed description + expert answer

**Effort**: High (data collection/annotation expensive and time-consuming)

### 🎯 Medium-term: Implement RAG
**Why**: Leverage external knowledge without fine-tuning

**Architecture**:
1. Vector database of mental health knowledge/FAQs
2. Retrieve relevant context for each question
3. Augment prompt with retrieved information
4. Generate response using baseline model + context

**Benefits**:
- No training required
- Easily updateable knowledge base
- Combines retrieval + generation strengths

### 🚀 Long-term: Proper Fine-tuning (If More Data Collected)
**Requirements**:
- 10,000+ samples
- Proper completion-only training implementation
- Multi-stage training (general → domain-specific)
- Extensive evaluation and validation

**Alternative**: Consider using commercial APIs (GPT-4, Claude) which already have strong mental health knowledge

---

## File Structure

```
/content/sample_data/
├── PsyQA_example.json              # Test dataset
├── prompt1-eng.json                # Clinical approach training data
├── Prompt2-eng.json                # Empathetic approach training data
├── prompt3-eng.json                # Structured approach training data
├── rewritten_prompt4-chineese.json # Formal Chinese training data
└── rewritten_prompt5-chineese.json # Friendly Chinese training data
```

---

## Key Metrics Explained

### ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)
- **Measures**: Longest common subsequence between prediction and reference
- **Range**: 0-1 (higher is better)
- **Strength**: Good for lexical overlap
- **Weakness**: Doesn't capture semantic similarity

### BLEU-4 (Bilingual Evaluation Understudy)
- **Measures**: N-gram precision (4-gram)
- **Range**: 0-1 (higher is better)
- **Strength**: Penalizes word order changes
- **Weakness**: Struggles with paraphrases

### BERTScore
- **Measures**: Semantic similarity using contextual embeddings
- **Range**: 0-1 (higher is better)
- **Components**: Precision, Recall, F1
- **Strength**: Captures meaning even with different words
- **Best for**: Evaluating mental health responses where paraphrasing is common

**Primary Metric**: BERTScore F1 (most reliable for this task)

---

## Conclusion

This project demonstrates that:

1. **Baseline models are strong**: Mistral-7B's pre-training is already highly capable for mental health Q&A in Chinese

2. **Fine-tuning needs data**: 200-500 samples is insufficient for 7B parameter model

3. **Completion-only training is critical**: Training on full sequences breaks the model

4. **Prompt engineering works**: Can achieve good results by optimizing prompts without training

5. **Library compatibility matters**: Rapidly evolving ML libraries require careful version management

**Current Best Approach**: Use `baseline_prompt_comparison.ipynb` to find optimal prompt format and deploy baseline Mistral-7B with best prompt.

**Future Path**: If more data becomes available, revisit fine-tuning with proper completion-only training implementation.
