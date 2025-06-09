

## Directory Structure

```plaintext
.
|____scripts
|____src
| |____latent
| | (SFT code)
| |____grpo_attention_tuning (RL code)
| | |____latent_grpo_processor.py
| | |____latent_grpo_dataset.py
| | |____model.py
| | |____grpo_trainer.py
| | |____train_noise_grpo.py
| | |____noise_eval.py
| |____utils (others)
|____data (example data and our processing code)
| |____process.py
|____environment.yaml
|____README.md
```

## Environment

```bash
conda env create -n your_env -f environment.yaml
```

## Train and evaluation

```bash
conda activate your_env
# SFT
bash scripts/latent_train.sh
# modified GRPO (change the path to your sft model)
bash scripts/attention_grpo.sh
# for evaluation
bash scripts/grpo_eval.sh
```

> **Note:**
> - We strongly recommend that you use two GPUs to run GRPO. We did not test our code on more GPUs due to computational resources. This is related to the calculation of the advantage function.

## Hyperparameter Search

To achieve expected results, tune the learning rate in GRPO:

- `lr`: {1e-5, 1e-4, 5e-4}

You can tune more hyperparameters to get more impressive results.

> **Note:**
> - Results may vary across different devices even with the same hyperparameters, due to differences in computation precision. We conduct our experiments using 2 NVIDIA A100 GPUs.

