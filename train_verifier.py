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
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", default = None, type = str)
    parser.add_argument("--state_dir", default = None, type = str)
    args = parser.parse_args()
       
    backend = "nccl"
    init_process_group(backend = backend)
    local_rank = int(os.environ["LOCAL_RANK"])

    model_path = "mistralai/Mistral-7B-v0.1"
    peft_path = "checkpoint/generator"

    generator, tokenizer = load_generator_and_tokenizer(
        model_path = model_path,
        peft_path = peft_path,
        local_rank = local_rank,
    )
    
    r = 64
    lora_alpha = 16
    lora_dropout = 0.1

    lora_config = LoraConfig(
        r = r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        bias = "none",
        task_type = "CAUSAL_LM",
        target_modules = [
             "q_proj" , "k_proj" , "v_proj", "o_proj", "gate_proj" , "up_proj" ,"down_proj", "lm_head",
        ]
    )
    generator = get_peft_model(generator, lora_config)
    
    verifier = VerifierModel(backbone = generator, checkpoint_dir = args.checkpoint_dir)
    verifier = verifier.to(f"cuda:{local_rank}")
    verifier = DDP(verifier, device_ids = [local_rank])

    VDataset_cls = VerifierDataset(
        tokenizer = tokenizer,
        data_path = "longhoang06/vi-ovm-dataset",
        max_length = 512,
        load_data_method = "hf_hub",
        mapping = True,
    )

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
        saving_steps = 10000
        optimizer = AdamW(verifier.parameters(), lr = lr, weight_decay = 0.001) 
        
        num_update_steps_per_epoch = len(train_dataloader)
        if args.state_dir:
            state_cp = torch.load(args.state_dir)
            global_steps = state_cp["global_steps"]
            optimizer.load_state_dict(state_cp["optim_state"])
            num_steps = num_update_steps_per_epoch * epochs - global_steps
            lr_scheduler = get_scheduler(
                "cosine",
                optimizer = optimizer,
                num_warmup_steps = 0,
                num_training_steps = num_steps,
            )
            lr_scheduler.load_state_dict(state_cp["scheduler_state"])
            total_loss = state_cp["total_loss"]
        else:
            global_steps = 0
            num_steps = num_update_steps_per_epoch * epochs
            lr_scheduler = get_scheduler(
                "cosine",
                optimizer = optimizer,
                num_warmup_steps = int(warump_ratio * num_steps),
                num_training_steps = num_steps,
            )
            total_loss = 0
        
        idx = 0
        for epoch in range(epochs):
            train_dataloader.sampler.set_epoch(epoch)
            verifier.train()
            
            for batch in train_dataloader:
                idx += 1
                if idx > global_steps:
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
                    
                    global_steps += 1
                    
                    if global_steps % logging_steps == 0 and is_master_process():
                        print(f'Epoch: {epoch + 1} -- global_steps: {global_steps} -- avg_loss: {total_loss/global_steps} -- llm_loss: {all_losses["llm_loss"]} -- v_loss: {all_losses["v_loss"]}')
                    
                    if global_steps % saving_steps == 0 and is_master_process():
                        print("SAVING..............................")
                        torch.save(verifier.module.state_dict(), "verifier_model.pt")
                        torch.save(
                            {
                                "optim_state": optimizer.state_dict(),
                                 "scheduler_state": lr_scheduler.state_dict(),
                                 "global_steps": global_steps,
                                 "total_loss": total_loss,
                            },
                            "verifier_state"
                        )
                        print("*********** SAVE SUCCESSFULLY ***********")
                        
            if idx == global_steps and is_master_process():
                print(f'Epoch: {epoch + 1} -- global_steps: {global_steps} -- avg_loss: {total_loss/global_steps} -- llm_loss: {all_losses["llm_loss"]} -- v_loss: {all_losses["v_loss"]}')
                print("SAVING..............................")
                torch.save(verifier.module.state_dict(), "verifier_model.pt")
                torch.save(
                    {
                        
                        "optim_state": optimizer.state_dict(),
                        "scheduler_state": lr_scheduler.state_dict(),
                        "global_steps": global_steps,
                        "total_loss": total_loss,
                    },
                    "verifier_state"
                )
                print("*********** SAVE SUCCESSFULLY ***********")
                print(f"------------------- End of epoch {epoch + 1} -------------------")
                
    # TRAINING
    train()
    destroy_process_group()

                
            
            
                             
                           
                                   
                    
        
        
        
        
    
    








