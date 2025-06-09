import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
# os.environ["WANDB_DISABLED"] = "true"
import fire
import torch
import torch.nn as nn

import trl
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import LatentModel

from latent_grpo_dataset import D3Dataset, get_hash, get_prefix_data
from grpo_trainer import NoiseGRPORecTrainer


def train(
    # model/data params, first is your latent model path (train from latent reasoning)
    base_model: str = "output_dir/Toys_and_Games/latent",  # the only required argument
    train_file: str = "data/train/Toys_and_Games_5_2017-1-2018-11.csv",
    eval_file: str = "data/valid/Toys_and_Games_5_2017-1-2018-11.csv",
    info_file: str = "data/info/Toys_and_Games_5_2017-1-2018-11.txt",
    lr: float = float(6.0e-6),
    end_k: int = -1,
    num_epochs: int = 1,
    batch_size: int = 256,
    micro_batch_size: int = 8,
    sample: int = 2000,
    seed: int = 42,
    cutoff_len: int = 512,
    output_dir: str = "output_dir/Video_Games/grpo_test",
    category: str = "Toys_and_Games", # category name

    num_generations: int = 8, # grpo args
    beta: float = 0.01, # KL coefficient
    num_iterations: int = 2, # number of iterations
    epsilon: float = 0.2, # clipping value
    epsilon_high: float = 0.28, # 其实这两个一个是high，一个是low. 当没有high的时候，low就是high
    max_completion_length: int = 256,

    use_vllm: bool = False,
    vllm_gpu_memory_utilization: float = 0.7,

    resume_from_checkpoint: str = None,
    local_rank: int = 0,

):
    # print(train_file)
    category_dict = {"Office_Products": "office products", "Books": "books", "steam": "games",
                     "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games",
                     "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors",
                     "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie",
                     "yelp": "resturant"}
    print(category)
    category = category_dict[category]
    print(lr, type(lr))
    # alloc_first(all_s=70)
    device_map = "auto"
    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    assert batch_size==micro_batch_size*gradient_accumulation_steps*world_size

    model = LatentModel.from_pretrained(base_model,
        torch_dtype=torch.bfloat16,
    )
    model.attention.end_k = end_k
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False
    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = count_trainable_params(model)
    print(f"Trainable parameters: {trainable_params}")
    # replace_with_noisy_linear(model.attention)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    assert tokenizer("<|Thought|>")['input_ids'][0] == len(tokenizer) - 1

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    hash_dict = get_prefix_data(info_file, tokenizer)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []

    train_data = D3Dataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed,
                           category=category, test=False)
    val_data = D3Dataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed,
                         category=category, test=False)

    print("LOAD DATA FINISHED")

    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})

    #############################
    # Initialize the GRPO trainer
    #############################
    def reward(completions, targets, **kwargs):
        def strip_(a):
            return a.strip().strip('\n').strip('\"')
        # aa=[1 if strip_(com) == strip_(tar) else 0 for com, tar in zip(completions, targets)]
        # print(f'共有{len(set(completions))}种答案, {sum(aa)}分')
        return [1 if strip_(com) == strip_(tar) else 0 for com, tar in zip(completions, targets)]

    reward_funcs = [reward]
    trainer = NoiseGRPORecTrainer(
    # from grpo_trainer import HalfGRPORecTrainer
    # trainer = HalfGRPORecTrainer(
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        model=model,
        reward_funcs=reward_funcs,
        args= trl.GRPOConfig(
            # deepspeed='scripts/zero2.json',
            warmup_steps=200,
            # warmup_ratio=0.05,
            num_generations=num_generations,
            max_prompt_length=512,
            temperature=1.0,
            max_completion_length=max_completion_length,
            use_vllm=use_vllm,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            learning_rate=lr,
            beta=beta,
            num_iterations=num_iterations,
            epsilon=epsilon,
            epsilon_high=epsilon_high,
            num_train_epochs=num_epochs,
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            seed=seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler_type="cosine",
            eval_strategy="no",
            # eval_steps=1000,
            # save_strategy="epoch",
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=True,
            report_to="none",
            bf16=True,
            logging_steps=1,
        ),
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        processing_class=tokenizer,
    )
    model.config.use_cache = False
    ###############
    # Training loop
    ###############
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(hf_train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)



