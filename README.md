# KGP-QA: Boosting LLM on Multi-Document Question Answering with Knowledge Graph Prompting

## Environment Setup
### 1. Clone the project
```
git clone https://github.com/zukangy/KGP-MDQA.git
cd KGP-MDQA
```

### 2. Virtual Environment (conda or pyenv)

Anaconda
```
conda install -c anaconda python=3.8
```

[Pyenv](https://github.com/pyenv/pyenv)

Recommend installing with [here](https://github.com/pyenv/pyenv?tab=readme-ov-file#basic-github-checkout)

* In a terminal

```
# Install python3.8, only need to run once
pyenv install 3.8.16
```

```
cd KG-LLM-MDQA/
pyenv local 3.8.16
python3.8 -m venv .env --copies
```
* Activate the environment
```
. .env/bin/activate
```
Or
```
source .env/bin/activate
```

## Scripts Breakdown
1) KGP-CuriousLLM/create_dirs.py: create necessary empty folders for file management.
```
cd KGP-CuriousLLM/
python create_dirs.py
```
2) KGP-CuriousLLM/MDR_main.py: Train a MDR encoder for passage embedding; the embedding will be used in the knowledge graph construction. Model configuration is in configs/MDR.yml.
```
cd KGP-CuriousLLM/
python MDR_main.py
```
3) KGP-CuriousLLM/MDR_embedding_main.py: Generate embeddings with MDR model from test_docs.json.
```
cd KGP-CuriousLLM/
python MDR_embedding_main.py
```
4) KGP-CuriousLLM/kg_construct_main: Construct a KG for either HotpotQA or 2WikiMQA.
```
cd KGP-CuriousLLM/
python kg_construct_main.py
```
5) Fine-tune a curious Mistral-7B model in [MLX framework](https://github.com/ml-explore/mlx) using [QLora](https://github.com/ml-explore/mlx-examples/tree/main/lora).
```
cd KGP-CuriousLLM/
# If quantize the model into 8 bit.
python quantize_mistral_main.py

# Fine-tune Mistral; modified config.yml if resume training from checkpoint.
python ft_mistral_main.py

# To perform grid search on the test set for evaluation
python grid_search_mistral_main.py

# To generate metrics
python Evaluations/eval_followup_llm.py
``` 

6) Finally, to start graph traversal to collect evidence.
```
cd cd KGP-CuriousLLM/
```

7) KGP-CuriousLLM/T5_main.py: Train a T5 model for graph traversal. 