# Graph Report - Linux_SLM_V1  (2026-09-12)

## Corpus Check
- 25 files · ~278,207 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 588 nodes · 827 edges · 17 communities (11 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 18 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- BPE Vocabulary
- Training Pipeline
- LinuxSLM Model Core
- LinuxSLM Model (Archived)
- Model v1 Architecture
- Model from Scratch
- Evaluation Pipeline
- Chat Interface
- Generation Testing
- Dataset Preparation
- Dependencies & Tokenizer
- Dataset Tokenization
- Tokenizer Training

## God Nodes (most connected - your core abstractions)
1. `LinuxSLM` - 26 edges
2. `main()` - 20 edges
3. `LinuxSLM` - 17 edges
4. `GPTConfig` - 14 edges
5. `GPTConfig` - 13 edges
6. `main()` - 13 edges
7. `main()` - 9 edges
8. `generate_reply()` - 8 edges
9. `generate_one()` - 8 edges
10. `load_model()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `build_model()` --uses--> `GPTConfig`  [INFERRED]
  eval.py → archive/model.py
- `build_model()` --uses--> `GPTConfig`  [INFERRED]
  test_generate.py → archive/model.py
- `build_model()` --uses--> `GPTConfig`  [INFERRED]
  train.py → archive/model.py
- `generate_reply()` --uses--> `LinuxSLM`  [INFERRED]
  chat.py → archive/model.py
- `load_model()` --uses--> `LinuxSLM`  [INFERRED]
  chat.py → archive/model.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Byte-Pair Encoding Tokenizer Subsystem** — requirements_tokenizers, tokenizer_merges, tokenizer_test [INFERRED 0.75]
- **Model Training Dependency Stack** — requirements_einops, requirements_numpy, requirements_tokenizers, requirements_torch [INFERRED 0.75]

## Communities (17 total, 2 thin omitted)

### Community 0 - "BPE Vocabulary"
Cohesion: 0.01
Nodes (320): 0, 00, 1, ½, ¼, 2, 3, ¾ (+312 more)

### Community 1 - "Training Pipeline"
Cohesion: 0.11
Nodes (35): GradScaler, Logger, memmap, append_metrics(), _as_cpu_byte_tensor(), _as_cuda_rng_state_list(), build_model(), cleanup_old_checkpoints() (+27 more)

### Community 2 - "LinuxSLM Model Core"
Cohesion: 0.10
Nodes (13): apply_rotary_pos_emb(), CausalSelfAttention, GPTConfig, LinuxSLM, MLP, Module, no_grad, Optimizer (+5 more)

### Community 3 - "LinuxSLM Model (Archived)"
Cohesion: 0.10
Nodes (13): apply_rotary_pos_emb(), CausalSelfAttention, GPTConfig, LinuxSLM, MLP, Module, no_grad, Optimizer (+5 more)

### Community 4 - "Model v1 Architecture"
Cohesion: 0.10
Nodes (15): apply_rotary_pos_emb(), CausalSelfAttention, GPTConfig, LinuxSLM, MLP, Tensor, LinuxSLM --------- Decoder-only Transformer for Linux command generation.…, Rotate half the hidden dimension. (..., d) -> (-x2, x1) (+7 more)

### Community 5 - "Model from Scratch"
Cohesion: 0.11
Nodes (14): apply_rotary_pos_emb(), CausalSelfAttention, GPTConfig, init_weights(), LinuxSLM, no_grad, LinuxSLM Transformer Part 1 - Imports - RMSNorm - Rotary Embeddings (RoPE) -…, SwiGLU Feed Forward Network hidden_dim = 8/3 * n_embd (LLaMA style) (+6 more)

### Community 6 - "Evaluation Pipeline"
Cohesion: 0.17
Nodes (25): build_model(), clean_candidate(), decode(), detect_device(), exact_match(), generate_one(), load_benchmark(), load_checkpoint() (+17 more)

### Community 7 - "Chat Interface"
Cohesion: 0.15
Nodes (21): build_chat_prompt(), build_model(), clean_reply(), decode(), detect_device(), generate_reply(), load_checkpoint(), load_model() (+13 more)

### Community 8 - "Generation Testing"
Cohesion: 0.25
Nodes (18): build_model(), decode(), detect_device(), generate(), get_special_token(), load_checkpoint(), load_model(), load_tokenizer() (+10 more)

### Community 9 - "Dataset Preparation"
Cohesion: 0.46
Nodes (7): main(), process_file(), process_json_array(), process_jsonl(), prepare_dataset.py Reads all JSON files from data/raw/, extracts useful text…, record_to_text(), write_record()

### Community 10 - "Dependencies & Tokenizer"
Cohesion: 0.29
Nodes (7): Project Dependency Manifest (requirements.txt), einops, NumPy, tokenizers (Hugging Face), PyTorch, Tokenizer BPE Merges (Byte-Pair Merge List, v0.2), Tokenizer Test File

## Ambiguous Edges - Review These
- `Tokenizer BPE Merges (Byte-Pair Merge List, v0.2)` → `Tokenizer Test File`  [AMBIGUOUS]
  tokenizer_test.txt · relation: conceptually_related_to

## Knowledge Gaps
- **324 isolated node(s):** `<pad>`, `<bos>`, `<eos>`, `<unk>`, `<eod>` (+319 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 385 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Tokenizer BPE Merges (Byte-Pair Merge List, v0.2)` and `Tokenizer Test File`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `LinuxSLM` connect `LinuxSLM Model Core` to `Generation Testing`, `Training Pipeline`, `Evaluation Pipeline`, `Chat Interface`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Why does `LinuxSLM` connect `LinuxSLM Model (Archived)` to `Generation Testing`, `Training Pipeline`, `Evaluation Pipeline`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **Why does `GPTConfig` connect `LinuxSLM Model (Archived)` to `Generation Testing`, `Training Pipeline`, `Evaluation Pipeline`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `LinuxSLM` (e.g. with `generate_reply()` and `load_model()`) actually correct?**
  _`LinuxSLM` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `GPTConfig` (e.g. with `build_model()` and `build_model()`) actually correct?**
  _`GPTConfig` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `<pad>`, `<bos>`, `<eos>` to the rest of the system?**
  _324 weakly-connected nodes found - possible documentation gaps or missing edges._