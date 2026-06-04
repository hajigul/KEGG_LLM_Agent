# KEGG50k Drug-Relation Discovery LLM Agent

A local LLM agent that discovers the relationship between two biomedical entities in the **KEGG50k** knowledge graph (drugs, genes, diseases, pathways, networks). You ask a question in plain English; a small LLM running on your own machine calls a trained relation-prediction model and explains the result.

No OpenAI key required. Everything runs locally via [Ollama](https://ollama.com).

## What it does

KEGG50k is a biomedical knowledge graph made of triplets in the form `head  relation  tail`, for example:

```
D01920    DRUG_TARGET_GENE    HSA:495
H00787    DISEASE_PATHWAY     hsa04080
N00060    NETWORK_GENE        HSA:23401
```

This project tackles **relation prediction** (also called drug-relation discovery): given a head entity and a tail entity, predict the most likely relation(s) between them, with confidence scores.

There are two components:

1. A trained relation-prediction model. It learns entity embeddings and a neural classifier that maps a `(head, tail)` pair to a relation.
2. A local LLM agent. A small model (e.g. Llama 3.2 3B) reads your natural-language question, calls the prediction model as a tool, and explains the predicted relations in plain language.

## How it works

The system has two phases. You train the model once; afterwards the agent loads the saved model and answers questions.

```
Training:   triplet files  ->  build id mappings  ->  train model  ->  saved artifacts
Inference:  your question   ->  local LLM  ->  predict_relation tool  ->  trained model
                                    ^                                          |
                                    +------- plain-language answer <-----------+
```

The LLM handles language understanding and explanation; the trained model handles the actual prediction.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- Python packages: `torch`, `numpy`, `ollama`

## Installation

1. Install the Python dependencies:

   ```bash
   pip install torch numpy ollama
   ```

2. Install Ollama from https://ollama.com/download, then pull a light model:

   ```bash
   ollama pull llama3.2:3b
   ```

   If your machine is low on RAM, use `llama3.2:1b` instead. `qwen2.5:3b` is another good option. Set your choice in `LLM_MODEL` at the top of `kegg_llm_agent.py`.

## Dataset setup

Place your three triplet files in a `data/` folder:

```
data/
  train.txt
  valid.txt
  test.txt
```

Each line is a tab-separated triplet: `head <TAB> relation <TAB> tail`. The loader also falls back to whitespace splitting if tabs are not present.

Set the dataset path in `kegg_agent.py`:

```python
DATA_DIR = r"C:\path\to\your\data"
```

If your files are named `train`, `test`, `valid` without the `.txt` extension, edit the `TRAIN_FILE`, `VALID_FILE`, and `TEST_FILE` lines to match.

## Usage

### 1. Train the model

```bash
python kegg_agent.py --train
```

This reads the triplets, builds entity/relation mappings, trains the model, and saves `relation_agent.pt` and `mappings.pkl` into your data folder. It prints training loss, validation accuracy, and final test accuracy.

### 2. Use the predictor directly (no LLM)

```bash
python kegg_agent.py --predict H00787 hsa04080
```

Or interactively:

```bash
python kegg_agent.py --interactive
```

### 3. Run the LLM agent

Make sure Ollama is running, then:

```bash
python kegg_llm_agent.py
```

Ask questions like:

- `What is the relation between H00787 and hsa04080?`
- `How are D01920 and HSA:495 connected?`

The agent extracts the two entity IDs, runs the prediction model, and explains the top relations with confidence scores.

## Entity ID formats

| Type    | Example   |
|---------|-----------|
| Drug    | `D01920`  |
| Gene    | `HSA:495` |
| Disease | `H00787`  |
| Pathway | `hsa04080`|
| Network | `N00060`  |

## Configuration

Training settings live at the top of `kegg_agent.py`:

| Setting      | Default | Meaning                            |
|--------------|---------|------------------------------------|
| `EMB_DIM`    | 200     | Entity embedding dimension         |
| `EPOCHS`     | 50      | Training epochs                    |
| `BATCH_SIZE` | 1024    | Mini-batch size                    |
| `LR`         | 1e-3    | Learning rate                      |

The LLM model name is set by `LLM_MODEL` in `kegg_llm_agent.py`.

## Project structure

```
kegg_llm_agent/
  kegg_agent.py        # trains the model and provides the prediction tool
  kegg_llm_agent.py    # local LLM agent that calls the tool
  data/                # your triplet files (and saved model after training)
  README.md
```

## Notes and limitations

- Tool-calling can be unreliable on very small models (1B-3B). `llama3.2:3b` and `qwen2.5:3b` are the most dependable in that size range. The agent includes a regex fallback that extracts two IDs from your message and calls the tool directly if the LLM fails to.
- If `import ollama` works but calls fail with a connection error, Ollama is not running. Launch the Ollama app, or run `ollama serve`.
- This is a link/relation prediction model, not a general-purpose biomedical reasoner. Predictions reflect patterns in the training triplets.
- GPU is used automatically if a CUDA-capable device is available; otherwise it runs on CPU.

## Possible extensions

- **Tail prediction** (drug-target discovery): given a head and a relation, score all candidate tails to answer questions like "what genes does D01920 target?"
- Add entity name/description lookup so the agent can refer to drugs and genes by name, not just ID.

## License

Add your chosen license here (e.g. MIT).
