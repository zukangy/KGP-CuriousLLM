# KGP-QA: Boosting LLM on Multi-Document Question Answering with Knowledge Graph Prompting

## Environment Setup
### 1. Clone the project
```
git clone https://github.com/zukangy/KG-LLM-MDQA.git
cd KG-LLM-MDQA
```

### 2. Clone the submodules
```
git submodule update --init --recursive --remote
```

### 3. Virtual Environment (conda or pyenv)

Anaconda
```
conda install -c anaconda python=3.8
```

[Pyenv](https://github.com/pyenv/pyenv)

    * Recommend installing with [here](https://github.com/pyenv/pyenv?tab=readme-ov-file#basic-github-checkout)

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