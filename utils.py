import torch
from transformers import AutoTokenizer
from datasets import Dataset
from typing import Dict, List

def preprocess_for_dpo(
    examples: Dict[str, List[str]], 
    tokenizer: AutoTokenizer, 
    max_length: int = 2048,
    label_pad_token_id: int = -100
) -> Dict[str, torch.Tensor]:
    """
    Preprocesses a batch of preference data for DPO training.

    This function takes a dictionary of lists (as provided by `dataset.map(batched=True)`),
    where each list corresponds to 'prompt', 'chosen', and 'rejected' strings.
    It formats and tokenizes the data into the six required tensors for the custom DPO loss function.

    Args:
        examples (Dict[str, List[str]]): A batch of examples from a Hugging Face dataset.
                                         Expected keys: 'prompt', 'chosen', 'rejected'.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        max_length (int): The maximum sequence length for truncation.
        label_pad_token_id (int): The ID to use for masking out tokens in the labels.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the six required tokenized tensors.
    """
    # Ensure the tokenizer has a pad token; for decoder-only models, this is often the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = len(examples['prompt'])
    
    # --- 1. Format chosen and rejected completions using the chat template ---
    chosen_full_texts = []
    rejected_full_texts = []
    prompt_only_texts = []

    for i in range(batch_size):
        prompt_text = examples['prompt'][i]
        chosen_text = examples['chosen'][i]
        rejected_text = examples['rejected'][i]

        # Format for chosen response
        messages_chosen = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": chosen_text}
        ]
        chosen_full_texts.append(tokenizer.apply_chat_template(messages_chosen, tokenize=False))
        
        # Format for rejected response
        messages_rejected = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": rejected_text}
        ]
        rejected_full_texts.append(tokenizer.apply_chat_template(messages_rejected, tokenize=False))
        
        # Format prompt only to calculate its length for masking
        messages_prompt = [{"role": "user", "content": prompt_text}]
        prompt_only_texts.append(tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True))

    # --- 2. Tokenize all formatted texts ---
    tokenized_chosen = tokenizer(
        chosen_full_texts,
        truncation=True,
        max_length=max_length,
        padding="longest" # Pad to the longest sequence in this batch
    )
    tokenized_rejected = tokenizer(
        rejected_full_texts,
        truncation=True,
        max_length=max_length,
        padding="longest"
    )
    # Tokenize prompts without padding to get their true lengths
    tokenized_prompts = tokenizer(prompt_only_texts, truncation=True, max_length=max_length)

    # --- 3. Create the final batch dictionary ---
    batch_dict = {}

    # --- 4. Create Labels by masking out the prompt portion ---
    # The labels are a copy of the input_ids, with prompt tokens set to -100
    chosen_labels = torch.tensor(tokenized_chosen['input_ids']).clone()
    rejected_labels = torch.tensor(tokenized_rejected['input_ids']).clone()
    
    for i in range(batch_size):
        prompt_len = len(tokenized_prompts['input_ids'][i])
        
        # Mask chosen labels
        chosen_labels[i, :prompt_len] = label_pad_token_id
        # Also mask padding tokens if any exist in the response part
        chosen_labels[i][tokenized_chosen['attention_mask'][i] == 0] = label_pad_token_id
        
        # Mask rejected labels
        rejected_labels[i, :prompt_len] = label_pad_token_id
        # Also mask padding tokens
        rejected_labels[i][tokenized_rejected['attention_mask'][i] == 0] = label_pad_token_id

    # Populate the final dictionary with all required tensors
    batch_dict["chosen_input_ids"] = torch.tensor(tokenized_chosen['input_ids'])
    batch_dict["chosen_attention_mask"] = torch.tensor(tokenized_chosen['attention_mask'])
    batch_dict["chosen_labels"] = chosen_labels

    batch_dict["rejected_input_ids"] = torch.tensor(tokenized_rejected['input_ids'])
    batch_dict["rejected_attention_mask"] = torch.tensor(tokenized_rejected['attention_mask'])
    batch_dict["rejected_labels"] = rejected_labels
    
    return batch_dict



def preprocess_for_it(
    examples: Dict[str, List[str]], 
    tokenizer: AutoTokenizer, 
    max_length: int = 1024,
    label_pad_token_id: int = -100
) -> Dict[str, torch.Tensor]:
    """
    Preprocesses a batch of instruction-tuning data for the retain loss.

    This function takes a dictionary of lists (as provided by `dataset.map(batched=True)`),
    where each list corresponds to 'prompt' and 'response' strings.
    It formats and tokenizes the data into the three required tensors for the 
    `obj_standard_max_next_token` loss function.

    Args:
        examples (Dict[str, List[str]]): A batch of examples from a Hugging Face dataset.
                                         Expected keys: 'prompt', 'response'.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        max_length (int): The maximum sequence length for truncation.
        label_pad_token_id (int): The ID to use for masking out tokens in the labels.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'attention_mask', and 'labels'.
    """
    # Ensure the tokenizer has a pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    batch_size = len(examples['prompt'])
    full_texts = []
    prompt_only_texts = []

    # Correctly iterate over the batched examples
    for prompt, response in zip(examples['prompt'], examples['response']):
        # Format the full conversation
        messages_complete = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        full_texts.append(tokenizer.apply_chat_template(messages_complete, tokenize=False))
        
        # Format the prompt only to get its length for masking
        messages_prompt = [{"role": "user", "content": prompt}]
        prompt_only_texts.append(tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True))

    # Tokenize the full conversation texts, returning PyTorch tensors
    tokenized_complete = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        padding="longest",
        return_tensors="pt"
    )

    # Tokenize the prompts without padding to get their true lengths
    tokenized_prompts = tokenizer(
        prompt_only_texts,
        truncation=True,
        max_length=max_length,
        padding=False # More efficient
    )

    # Create labels by cloning input_ids
    labels = tokenized_complete.input_ids.clone()

    for i in range(batch_size):
        prompt_len = len(tokenized_prompts.input_ids[i])
        
        # Mask the prompt portion of the labels
        labels[i, :prompt_len] = label_pad_token_id
    
    # Also mask any padding tokens that might be in the response part
    # This is implicitly handled by CrossEntropyLoss's ignore_index, but explicit is good.
    # We can rely on the attention mask for this.
    labels[tokenized_complete.attention_mask == 0] = label_pad_token_id
    
    return {
        "input_ids": tokenized_complete.input_ids,
        "attention_mask": tokenized_complete.attention_mask,
        "labels": labels,
    }    