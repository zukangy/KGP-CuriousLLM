# CuriousLLM: Elevating Multi-Document QA with Reasoning-Infused Knowledge Graph Prompting

In our work, we proposed a novel reasoning-infused LLM traversal agent which generates question-based query to guide the search through a knowledge graph. The overall logic is illustrated below. For detailed explanation, please refer to our [paper]().

![Flowchart](/images/workflow.png)

## Environment Setup
### 1. Clone the project
```
git clone https://github.com/zukangy/KGP-CuriousLLM.git
cd KGP-CuriousLLM
```

### 2. Virtual Environment (pyenv and MacOS)

[Pyenv](https://github.com/pyenv/pyenv)

Recommend installing with [here](https://github.com/pyenv/pyenv?tab=readme-ov-file#basic-github-checkout)

* In a terminal

```
# Install python3.8, only need to run once
pyenv install 3.8.16
```

```
cd KGP-CuriousLLM/
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

    1) Below scripts only work on Apple Silicon Mac. 
    2) Recommend a 4-bit quantization for low-RAM Mac
    3) The equivalent HF implementation should be starightforward using the [trainer](https://huggingface.co/docs/transformers/main_classes/trainer) class. 
    4) For reference, the training time was roughly 8 hours based on the specs in the yaml file. 
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

6) Finally, to start the experimentation of graph traversal to collect evidence.

    1. ./configs/kgp/ has config files for all methods in the paper. Please adjust line 15 of kgp_main.py to run different experiment.  
    2. This isn't a complete pipeline. This script only collects evidence. The reason for so is for ease of downstream analysis. But you are wellcome to complete the entire pipeline by piecing the scripts together. 
    3. Output will be saved to ./DATA/KG/evidence/[hotpot_evidence_100 or wiki_evidence_100].
```
cd KGP-CuriousLLM/
python kgp_main.py
```

7) Train a T5 model for graph traversal. 

    1. We provide a training script using the [trainer](https://huggingface.co/docs/transformers/main_classes/trainer) class with Lora. But we instead used the optimal model provided by [KGP-T5](https://github.com/YuWVandy/KG-LLM-MDQA). 

8) Generate answers based on output evidence from the experiments. 
```
# For GPT
# Please modify the data_path and save_path arguments in the script if needed. 
python KGP/LLMs/GPT/generate_answer_gpt.py

# For Mistral-7B
python KGP/LLMs/Mistral/generate_answer.py
```

### Data
1. Data, KGs, and passage embedding are provided [here](https://drive.google.com/drive/folders/1sdgi9g5uuXARsLzveDwoeoi4xKf0eg4V?usp=sharing).
2. Additionally, please refer to [KGP-T5](https://github.com/YuWVandy/KG-LLM-MDQA) for the raw data as well as other datasets we haven't tested in our experimentation.



### Citatioin