from v_dataset import VerifierDataset
from build_verifier import VerifierModel, load_generator_and_tokenizer
from peft import LoraConfig, PeftConfig, get_peft_model, PeftModel
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torch.distributed import  destroy_process_group, init_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import SequentialSampler
import os
from argparse import ArgumentParser

if __name__ == "__main__":
    backend = "nccl"
    init_process_group(backend = backend)
    local_rank = int(os.environ["LOCAL_RANK"])

    generator_path = "checkpoint/generator"

    generator, tokenizer = load_generator_and_tokenizer(
        generator_path = model_path,
        load_k_bit = True,
        local_rank = local_rank,
    )
#     generator = generator.merge_and_unload()
    
#     r = 64
#     lora_alpha = 16
#     lora_dropout = 0.1

#     lora_config = LoraConfig(
#         r = r,
#         lora_alpha = lora_alpha,
#         lora_dropout = lora_dropout,
#         bias = "none",
#         task_type = "CAUSAL_LM",
#         target_modules = [
#              "q_proj" , "k_proj" , "v_proj", "o_proj", "gate_proj" , "up_proj" ,"down_proj", "lm_head",
#         ]
#     )
#     generator = get_peft_model(generator, lora_config)
    
    verifier = VerifierModel(backbone = generator, checkpoint_dir = None)
    verifier = verifier.to(f"cuda:{local_rank}")
    verifier = DDP(verifier, device_ids = [local_rank])

    VDataset_cls = VerifierDataset(
        tokenizer = tokenizer,
        data_path = "longhoang06/vi-ovm-dataset",
        max_length = 512,
        load_data_method = "hf_hub",
        mapping = True,
    )
    dataset = dataset.select(range(10))
    dataset = VDatasset_cls.dataset.set_format("torch")
    train_dataloader = DataLoader(
        dataset,
        batch_size = 1,
        sampler = DistributedSampler(dataset),
        pin_memory = True,
    )
    
    def train():
        
        def is_master_process():
            ddp_rank = int(os.environ['RANK'])
            return ddp_rank == 0
        
        epochs = 2
        lr = 2e-4
        max_norm_value = 0.3
        warmup_ratio = 0.03
        logging_steps = 300   
        num_update_steps_per_epoch = len(train_dataloader)
        num_steps = num_update_steps_per_epoch * epochs
        num_warmup_steps = int(warmup_ratio * num_steps)
        optimizer = AdamW(model.parameters(), lr = lr, weight_decay = 0.001)
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer = optimizer,
            num_warmup_steps = num_warmup_steps,
            num_training_steps = num_steps,
        )
        
        for epoch in range(epochs):
            train_dataloader.sampler.set_epoch(epoch)
            total_loss = 0
            cur_steps = 0
            verifier.train()
            for batch in train_dataloader:
                batch = {k:v.to(local_rank) for k, v in batch.items()}
                outputs = verifier(
                    input_ids = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    labels = batch["labels"],
                    v_labels = batch["v_labels"],
                    output_all_losses = True,
                )
                    
                loss = outputs.loss
                all_losses = outputs.all_losses
                total_loss += loss.item()
                loss.backward()
                    
                clip_grad_norm_(verifier.parameters(), max_norm_value)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                    
                cur_steps += 1
                    
                if cur_steps % logging_steps == 0 and is_master_process():
                    print(f'Epoch: {epoch + 1} -- cur_steps: {cur_steps} -- avg_loss: {total_loss/cur_steps} -- llm_loss: {all_losses["llm_loss"]} -- v_loss: {all_losses["v_loss"]}')
            
            if is_master_process():
                print("SAVING......................................................................")
                verifier.save_model("checkpoint/verifier")
                print("*********** SAVE SUCCESSFULLY ***********")
                print(f"------------------- End of epoch {epoch + 1} -------------------")
                
    # TRAINING
    train()
    destroy_process_group()
