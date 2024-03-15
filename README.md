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
1) KGP-MDQA/create_dirs.py: create necessary empty folders for file management.
2) KGP-MDQA/MDR_main.py: Train a MDR model for document embedding; the embedding will be used in the knowledge graph construction.
3) KGP-MDQA/MDR_embedding_main.py: Generate embeddings with MDR model from test_docs.json.
4) KGP-MDQA/kg_construct_main: Construct a KG.
5) KGP-MDQA/quantize_mistral_main.py: quantize a Mistral-7B model to fit the Mac's memory using the [MLX](https://github.com/ml-explore/mlx) framework. 
6) KGP-MDQA/ft_mistral_main.py: Fine-tune a Mistral-7B model using the quantized version and LORA technique for graph traversal. 
7) KGP-MDQA/question_generation_main.py: Use GPT3.5-turbo to generate a new dataset consists of questions, supporting questions (both of which are the input), and follow-up questions (output). 
8) KGP-MDQA/T5_main.py: Train a T5 model for graph traversal. 