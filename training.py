import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator, PartialState
from adversary import Adversary 
from peft import PeftModel, LoraConfig
from collections import defaultdict
import copy
from peft_injection import apply_adversarial_wrappers, restore_original_modules
from utils import preprocess_for_dpo, preprocess_for_it
from loss import compute_dpo_loss, compute_lm_loss, compute_kl_loss
from activation_cache import ActivationCache
from adversary import Adversary
from itertools import islice


def train_tamper_resistant_model(model: PeftModel, adversary: nn.Module, tokenizer, harmful_dataloader, benign_dataloader):
    """
    The complete, state-of-the-art training loop for creating a tamper-resistant model
    using a k:k interleaved adversarial schedule, with multi-GPU support via accelerate.
    """
    
    # --- Accelerator Setup ---
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        log_with=None,
    )
    device = accelerator.device

    # --- Gradient Checkpointing ---
    model.base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # --- Reference Model Setup ---
    with torch.no_grad():
        ref_model = copy.deepcopy(model.base_model)
        ref_model.eval()

    # --- Optimizer Setup ---
    defender_params = [p for n, p in model.named_parameters() if "defender" in n]
    optimizer_d = AdamW(defender_params, lr=3e-5)
    optimizer_a = AdamW(adversary.parameters(), lr=2e-7)

    # --- Prepare with Accelerator ---
    # Store original models for PeftModel-specific operations
    original_model = model
    original_ref_model = ref_model
    
    model, adversary, ref_model, optimizer_d, optimizer_a, harmful_dataloader, benign_dataloader = accelerator.prepare(
        model, adversary, ref_model, optimizer_d, optimizer_a, harmful_dataloader, benign_dataloader
    )

    # --- Dataloader and Schedule Setup ---
    k_steps = 2000
    harmful_iterator = iter(harmful_dataloader)
    benign_iterator = iter(benign_dataloader)
    num_total_steps = len(harmful_dataloader)
    num_blocks = num_total_steps // k_steps
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    act_cache = ActivationCache(model, target_modules)

    for epoch in range(3):
        if accelerator.is_main_process:
            print(f"\n===== Epoch {epoch+1}/{3} =====")
        
        for block_idx in range(num_blocks):
            # ==========================================================
            #           PHASE 1: ADVERSARY TRAINING BLOCK (k steps)
            # ==========================================================
            if accelerator.is_main_process:
                print(f"\n--- Block {block_idx+1}/{num_blocks}: Training Adversary for {k_steps} steps ---")
            adversary.train()
            model.eval()

            for step in range(k_steps):
                try:
                    harmful_batch = next(harmful_iterator)
                except StopIteration:
                    harmful_iterator = iter(harmful_dataloader)
                    harmful_batch = next(harmful_iterator)

                adversary_dpo_batch = preprocess_for_dpo(harmful_batch, tokenizer)
                adversary_dpo_batch = {k: v.to(device) for k, v in adversary_dpo_batch.items()}

                act_cache.clear_cache()
                with torch.no_grad():
                    # Use original model for PeftModel-specific operations
                    accelerator.unwrap_model(original_model).set_adapter("defender")
                    with act_cache:
                        _ = model(input_ids=adversary_dpo_batch['chosen_input_ids'])
                
                    lora_weights_adv = {}
                    for layer_idx, layer_name in enumerate(act_cache.activations.keys()):
                        activations = act_cache.activations[layer_name]
                        device = next(adversary.parameters()).device
                        activations = activations.to(device, dtype=torch.bfloat16)
                        config_name = layer_name.split('.')[-1]
                        U_gen, V_gen = adversary(activations, config_name)
                        lora_weights_adv[layer_name] = (U_gen, V_gen)
                original_modules = apply_adversarial_wrappers(model, lora_weights_adv)
                
                loss_a = compute_dpo_loss(model, ref_model, adversary_dpo_batch, device)
                
                accelerator.backward(loss_a)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_a.step()
                optimizer_a.zero_grad()

                restore_original_modules(model, original_modules)

                if accelerator.is_main_process and (step + 1) % 5 == 0:
                    print(f"  Adv Step {step+1}/{k_steps}, Loss: {loss_a.item():.4f}")

            # ==========================================================
            #           PHASE 2: DEFENDER TRAINING BLOCK (k steps)
            # ==========================================================
            if accelerator.is_main_process:
                print(f"\n--- Block {block_idx+1}/{num_blocks}: Training Defender for {k_steps} steps ---")
            adversary.eval()
            model.train()
            for n, p in accelerator.unwrap_model(original_model).named_parameters():
                p.requires_grad = "defender" in n

            for step in range(k_steps):
                try:
                    harmful_batch = next(harmful_iterator)
                    benign_batch = next(benign_iterator)
                except StopIteration:
                    harmful_iterator = iter(harmful_dataloader)
                    benign_iterator = iter(benign_dataloader)
                    harmful_batch = next(harmful_iterator)
                    benign_batch = next(benign_iterator)

                harmful_batch = {k: v.to(device) for k, v in harmful_batch.items()}
                benign_batch = {k: v.to(device) for k, v in benign_batch.items()}

                act_cache.clear_cache()
                with torch.no_grad():
                    # Use original model for PeftModel-specific operations
                    accelerator.unwrap_model(original_model).set_adapter("defender")
                    with act_cache:
                        _ = model(input_ids=harmful_batch['chosen_input_ids'])
                
                    lora_weights_adv = {}
                    for layer_idx, layer_name in enumerate(act_cache.activations.keys()):
                        activations = act_cache.activations[layer_name]
                        device = next(adversary.parameters()).device
                        activations = activations.to(device, dtype=torch.bfloat16)
                        config_name = layer_name.split('.')[-1]
                        U_gen, V_gen = adversary(activations, config_name)
                        lora_weights_adv[layer_name] = (U_gen, V_gen)
                original_modules_s = apply_adversarial_wrappers(model, lora_weights_adv)
                
                defender_dpo_batch = {
                    "chosen_input_ids": harmful_batch["safe_input_ids"],
                    "chosen_attention_mask": harmful_batch["safe_attention_mask"],
                    "chosen_labels": harmful_batch["safe_labels"],
                    "rejected_input_ids": harmful_batch["harmful_input_ids"],
                    "rejected_attention_mask": harmful_batch["harmful_attention_mask"],
                    "rejected_labels": harmful_batch["harmful_labels"],
                }

                # Swap chosen/rejected for defender objective
                defender_dpo_batch['chosen_input_ids'], defender_dpo_batch['rejected_input_ids'] = \
                    defender_dpo_batch['rejected_input_ids'], defender_dpo_batch['chosen_input_ids']
                defender_dpo_batch['chosen_attention_mask'], defender_dpo_batch['rejected_attention_mask'] = \
                    defender_dpo_batch['rejected_attention_mask'], defender_dpo_batch['chosen_attention_mask']
                defender_dpo_batch['chosen_labels'], defender_dpo_batch['rejected_labels'] = \
                    defender_dpo_batch['rejected_labels'], defender_dpo_batch['chosen_labels']

                loss_s = compute_dpo_loss(model, ref_model, defender_dpo_batch, device)
                restore_original_modules(model, original_modules_s)

                # Retain loss
                loss_lm = compute_lm_loss(model, benign_batch)
                loss_kl = compute_kl_loss(model, ref_model, benign_batch)

                total_loss_d = (1.0 * loss_s + 0.8 * loss_lm + 0.3 * loss_kl)
                
                accelerator.backward(total_loss_d)
                optimizer_d.step()
                optimizer_d.zero_grad()

                if accelerator.is_main_process and (step + 1) % 5 == 0:
                    print(f"  Def Step {step+1}/{k_steps}, Loss_S: {loss_s.item():.4f}, Loss_LM: {loss_lm.item():.4f}, Loss_KL: {loss_kl.item():.4f}")

    model.eval()
    # Use original model for final operations
    accelerator.unwrap_model(original_model).set_adapter("defender")

    # Unwrap and merge
    unwrapped_model = accelerator.unwrap_model(original_model)
    merged_model = unwrapped_model.merge_and_unload()

    if accelerator.is_main_process:
        print("Adapter merged successfully. Returning the hardened base model and the trained adversary.")

    return merged_model

import argparse
import json
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel

# -----------------------------
# Argument Parser
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train Tamper-Resistant Model")

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="HF model ID to load"
    )

    parser.add_argument(
        "--harmful_path",
        type=str,
        default="dpo_data.json",
        help="Path to harmful dataset JSON"
    )

    parser.add_argument(
        "--it_path",
        type=str,
        default="instruction_tuning_data.json",
        help="Path to instruction-tuning dataset JSON"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="result/hardened_model",
        help="Where to save the merged/hardened model"
    )

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':

    args = get_args()

    model_id = args.model_name

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        cache_dir='cache_dir',
        trust_remote_code=True
    )

    modules_to_save = [
        "embed_tokens",
        "input_layernorm",
        "post_attention_layernorm",
        "norm"
    ]

    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        modules_to_save=modules_to_save,
    )

    model = PeftModel(base_model, peft_config, adapter_name='defender')

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Layer configs
    layer_configs_7b = {
        'q_proj': (3584, 3584),
        'o_proj': (3584, 3584),
        'v_proj': (3584, 512),
        'k_proj': (3584, 512)
    }

    layer_configs_3b = {
        'q_proj': (2048, 2048),
        'o_proj': (2048, 2048),
        'v_proj': (2048, 256),
        'k_proj': (2048, 256)
    }

    # Using 3B config (your current assumption)
    adversary = Adversary(8, layer_configs_3b)

    # -----------------------------
    # Load datasets
    # -----------------------------
    with open(args.harmful_path) as f:
        harmful_ds = json.load(f)

    with open(args.it_path) as f:
        it_ds = json.load(f)

    harmful_loader = DataLoader(harmful_ds, batch_size=2, shuffle=True)
    it_loader = DataLoader(it_ds, batch_size=2, shuffle=True)

    # -----------------------------
    # Train model
    # -----------------------------
    final_model = train_tamper_resistant_model(
        model,
        adversary,
        tokenizer,
        harmful_loader,
        it_loader
    )

    save_target = final_model
    try:
        save_target.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        if accelerator.is_main_process:
            print(f"Saved hardened model + tokenizer to: {args.save_path}")
    except Exception as exc:
        # fallback: try saving unwrapped model
        try:
            accelerator.unwrap_model(model).base_model.save_pretrained(args.save_path)
            tokenizer.save_pretrained(args.save_path)
            if accelerator.is_main_process:
                print(f"Saved base_model (fallback) + tokenizer to: {args.save_path}")
        except Exception as exc2:
            if accelerator.is_main_process:
                print("WARNING: Failed to save model automatically. Exception:", exc, exc2)
    
