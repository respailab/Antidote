from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss

def log_p_loss(
    logits: torch.Tensor, labels: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """
    Compute the log probability loss for a language model.

    This function calculates the cross-entropy loss between the predicted logits
    and the true labels, typically used in language modeling tasks.

    Args:
        logits (torch.Tensor): The predicted logits from the model, typically of shape
                               (batch_size, sequence_length, vocab_size).
        labels (torch.Tensor): The true labels, typically of shape
                               (batch_size, sequence_length).
        vocab_size (int): The size of the vocabulary.

    Returns:
        torch.Tensor: The computed loss as a scalar tensor.
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def log_1_minus_p_loss(
    _logits: torch.Tensor,
    _labels: torch.Tensor,
    vocab_size: int,
    threshold: float = -15.0,
) -> torch.Tensor:
    """
    Compute the log(1 - P(label)) loss for a language model.

    This function calculates a loss based on the probability of not predicting the correct label,
    with a threshold to ignore tokens where the model is already sufficiently uncertain.

    Args:
        _logits (torch.Tensor): The predicted logits from the model, typically of shape
                                (batch_size, sequence_length, vocab_size).
        _labels (torch.Tensor): The true labels, typically of shape
                                (batch_size, sequence_length).
        vocab_size (int): The size of the vocabulary.
        threshold (float, optional): The threshold below which to ignore losses. Defaults to -15.0.

    Returns:
        torch.Tensor: The computed loss as a scalar tensor.
    """
    # Compute the log(sum(exp(logits))) for each token position

    logits = _logits[..., :-1, :].contiguous()
    labels = _labels[..., 1:].contiguous()
    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)

    log_sum_exp_all = torch.logsumexp(logits, dim=-1)

    # Temporarily replace -100 labels with 0 for the gather operation
    gather_labels = labels.clone()
    gather_labels[labels == -100] = 0

    # Get the logits corresponding to the labels
    logits_for_labels = torch.gather(logits, -1, gather_labels.unsqueeze(-1)).squeeze(
        -1
    )

    # Calculate log(P(label))
    log_p = logits_for_labels - log_sum_exp_all

    # Create a mask for the labels, so we can zero out the logits of true labels
    mask = torch.zeros_like(logits).scatter_(-1, gather_labels.unsqueeze(-1), 1.0)

    # Zero out the logits of true labels
    masked_logits = logits * (1 - mask) + mask * (
        -1e10
    )  # Large negative value to approximate zero when exponentiated

    # Compute the log(sum(exp(logits excluding true label))) for each token position
    log_sum_exp_without_true_label = torch.logsumexp(masked_logits, dim=-1)

    # Compute log(1 - P(label)) for each token position
    log_1_minus_p = log_sum_exp_without_true_label - log_sum_exp_all

    # Set losses for -100 labels to 0 (ignored values)
    ignored_values = labels == -100
    log_1_minus_p[ignored_values] = 0

    # Zero out the loss for tokens where log(P(label)) is less than the threshold
    below_threshold = log_p < threshold
    log_1_minus_p[below_threshold] = 0

    # Compute the mean of the log(1 - P(label)) values, excluding the ignored ones
    loss = -log_1_minus_p.sum() / (~ignored_values).sum().float()

    return loss


def max_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the negative mean entropy loss for the given logits.

    This function calculates the entropy of the softmax distribution of the input logits
    and returns the negative mean entropy as a loss value. Minimizing this loss
    encourages the model to produce more uniform (higher entropy) probability distributions.

    Args:
        logits (torch.Tensor): The input logits tensor.

    Returns:
        torch.Tensor: The negative mean entropy loss.
    """
    softmax = F.softmax(logits, dim=-1)
    log_softmax = F.log_softmax(logits, dim=-1)
    entropy = torch.sum(-softmax * log_softmax, dim=-1).mean()
    return entropy.mean() * -1


def _filter_dpo_inputs(
    inputs: Dict[str, torch.Tensor], chosen: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Filter inputs for Direct Preference Optimization (DPO) based on whether they are chosen or rejected.

    This function takes a dictionary of input tensors and filters them based on whether they
    are for the chosen or rejected option in a DPO setup.

    Args:
        inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors.
        chosen (bool, optional): A flag indicating whether to filter for chosen or rejected inputs.
                                 Defaults to False (i.e., rejected inputs).

    Returns:
        Dict[str, torch.Tensor]: A filtered dictionary containing only the relevant input tensors.
    """
    prefix = "chosen_" if chosen else "rejected_"
    if f"{prefix}input_ids" not in inputs:
        return inputs
    return {
        "input_ids": inputs[f"{prefix}input_ids"],
        "attention_mask": inputs[f"{prefix}attention_mask"],
        "labels": inputs[f"{prefix}labels"],
    }


def _filter_inputs(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Filter the input dictionary to keep only specific keys.

    This function takes a dictionary of input tensors and returns a new dictionary
    containing only the keys 'input_ids', 'attention_mask', and 'labels' if they exist
    in the original dictionary.

    Args:
        inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors.

    Returns:
        Dict[str, torch.Tensor]: A filtered dictionary containing only the specified keys.
    """
    return {
        k: v
        for k, v in inputs.items()
        if k in ["input_ids", "attention_mask", "labels"]
    }


def compute_lm_loss(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
    chosen: bool = False,
) -> torch.Tensor:
    """
    Compute the standard maximum next token objective.

    This function calculates the log probability loss for the next token prediction
    using the given model and inputs. It supports both standard inputs and
    Direct Preference Optimization (DPO) inputs.

    Args:
        model (torch.nn.Module): The model to use for prediction.
        inputs (Dict[str, torch.Tensor]): The input tensors for the model.
        device (Optional[torch.device]): The device object for distributed training. Defaults to None.
        chosen (bool): Flag to indicate whether to use chosen or rejected inputs for DPO. Defaults to False.

    Returns:
        torch.Tensor: The computed log probability loss.
    """
    outputs = model(
        **_filter_inputs(_filter_dpo_inputs(inputs, chosen)), output_hidden_states=False
    )
    return log_p_loss(
        outputs.logits,
        _filter_dpo_inputs(inputs, chosen).get("labels"),
        model.vocab_size,
    )

def pad_to_length(
    tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1
) -> torch.Tensor:
    """
    Pad a tensor to a specified length along a given dimension.

    Args:
        tensor (torch.Tensor): The input tensor to be padded.
        length (int): The desired length of the tensor along the specified dimension.
        pad_value (Union[int, float]): The value to use for padding.
        dim (int, optional): The dimension along which to pad. Defaults to -1 (last dimension).

    Returns:
        torch.Tensor: The padded tensor with the specified length along the given dimension.
    """
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value
                * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )

class DPOLoss(torch.nn.Module):
    """
    Direct Preference Optimization (DPO) Loss module: https://arxiv.org/abs/2305.18290.

    Based on the implementation in HF's TRL library:
    https://github.com/huggingface/trl/blob/5d1deb1445828cfd0e947cb3a7925b1c03a283fc/trl/trainer/dpo_trainer.py#L844

    Args:
        beta (float): Temperature parameter for the DPO loss, typically in the range of 0.1 to 0.5. Default is 0.1.
        label_smoothing (float): Parameter encoding uncertainty about the labels. Default is 0.
        loss_type (str): Type of loss function to be used. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair'].
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
        device: torch.device = None,
    ):
        super(DPOLoss, self).__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.device = device

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (batch and sequence length dim) and labels must have the same shape."
            )

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0
        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: torch.nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            device=self.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    @staticmethod
    def static_concatenated_forward(
        model: torch.nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        device: torch.device,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = DPOLoss.concatenated_inputs(
            batch,
            device=device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = DPOLoss.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    @staticmethod
    def compute_reference_log_probs(
        model, padded_batch: Dict, device: torch.device
    ) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""

        # compute reference logps
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = DPOLoss.static_concatenated_forward(model, padded_batch, device)

        return reference_chosen_logps, reference_rejected_logps

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(
                batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1]
            )
        else:
            max_length = max(
                batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
            )

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_length, pad_value=pad_value
                )
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = (
                batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            )
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps (torch.Tensor): Log probabilities of the policy model
                for the chosen responses. Shape: (batch_size)
            policy_rejected_logps (torch.Tensor): Log probabilities of the policy model
                for the rejected responses. Shape: (batch_size)
            reference_chosen_logps (torch.Tensor): Log probabilities of the reference model
                for the chosen responses. Shape: (batch_size)
            reference_rejected_logps (torch.Tensor): Log probabilities of the reference model
                for the rejected responses. Shape: (batch_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors:
                - losses: The DPO loss for each example in the batch.
                - chosen_rewards: Rewards for the chosen responses.
                - rejected_rewards: Rewards for the rejected responses.

        Raises:
            ValueError: If an unknown loss type is specified.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            chosen_kl = (
                (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            )
            rejected_kl = (
                (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)
            )

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_kl)),
                    1 - F.sigmoid(self.beta * (chosen_kl - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards


from contextlib import nullcontext


def dpo_loss_obj(
    policy_model: torch.nn.Module = None,
    ref_model: torch.nn.Module = None,
    batch: Dict[str, Union[List, torch.LongTensor]] = None,
    accelerator: Accelerator = None,
    gradient_accumulation_steps: int = None,
    scale: float = 1.0,
    backprop: bool = True,
):
    with torch.no_grad() if not backprop else nullcontext():
        dpo_loss = DPOLoss(beta=0.1, accelerator=accelerator, loss_type="ipo")
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = dpo_loss.concatenated_forward(policy_model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"].to(
                device
            )
            reference_rejected_logps = batch["reference_rejected_logps"].to(
                device
            )
        else:
            with torch.no_grad():
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = dpo_loss.concatenated_forward(ref_model, batch)

        losses, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        loss = losses.mean() / gradient_accumulation_steps * scale
        if backprop:
            accelerator.backward(loss)
    return loss.item(), reward_accuracies.mean().item() / gradient_accumulation_steps

def compute_dpo_loss(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: Dict[str, Union[List, torch.LongTensor]],
    device: torch.device,
) -> torch.Tensor:
    """
    Computes the DPO loss and returns it as a differentiable tensor.
    This function does NOT perform a backward pass.
    """
    # We can reuse the DPOLoss class for its helper methods
    dpo_loss_util = DPOLoss(beta=0.1, device=device, loss_type="ipo")

    # 1. Get log probabilities from the policy model (the one being trained)
    (
        policy_chosen_logps,
        policy_rejected_logps,
        _, _,
    ) = dpo_loss_util.concatenated_forward(policy_model, batch)

    # 2. Get log probabilities from the reference model
    with torch.no_grad():
        (
            reference_chosen_logps,
            reference_rejected_logps,
            _, _,
        ) = dpo_loss_util.concatenated_forward(ref_model, batch)

    # 3. Compute the per-sample losses using the DPOLoss forward method
    losses, _, _ = dpo_loss_util(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )

    # 4. Return the mean loss as a single scalar tensor
    return losses.mean()

def compute_kl_loss(model, ref_model, benign_batch):
    # 1. Get logits from both models
    with torch.no_grad():
        ref_outputs = ref_model(**benign_batch)
        ref_logits = ref_outputs.logits
    
    policy_outputs = model(**benign_batch)
    policy_logits = policy_outputs.logits

    # 2. Get log-probs for the policy and probs for the reference
    # We only need log_softmax for the policy, which is more stable
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_probs = F.softmax(ref_logits, dim=-1)
    
    # 3. Compute the forward cross-entropy: -sum(P * log(Q))
    # The negative sign is handled by convention (loss is minimized)
    cross_entropy = -(ref_probs * policy_log_probs).sum(dim=-1)
    
    # 4. Average over the batch and sequence length
    # We need to respect the attention mask to not average over padding
    mask = benign_batch['attention_mask']
    masked_loss = cross_entropy * mask
    
    # Average only over non-masked tokens
    kl_loss = masked_loss.sum() / mask.sum()

    return kl_loss