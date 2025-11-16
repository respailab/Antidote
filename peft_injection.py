import torch 
import torch.nn as nn 
import torch.nn.functional as F

class AdversarialPeftWrapper(nn.Module):
    def __init__(self, original_peft_layer, delta_adv):
        super().__init__()
        self.original_peft_layer = original_peft_layer
        # Always register as buffer for device management
        self.register_buffer("delta_adv", delta_adv, persistent=False)
        # Optionally, store input/output dims for safety
        self.in_features = original_peft_layer.in_features
        self.out_features = original_peft_layer.out_features

    def forward(self, x: torch.Tensor, *args, **kwargs):
        # 1. Get the standard output from the PEFT layer
        original_output = self.original_peft_layer(x, *args, **kwargs)
        
        # --- CRITICAL FIX START ---

        # 2. Reshape the input for F.linear if it's 3D
        original_shape = x.shape
        if x.dim() == 3:
            # Reshape from (batch, seq_len, in_features) to (batch * seq_len, in_features)
            x_reshaped = x.reshape(-1, original_shape[-1])
        else:
            x_reshaped = x

        # 3. Apply the adversarial delta in 2D
        delta_adv_device = self.delta_adv.to(x_reshaped.device, dtype=x_reshaped.dtype)
        delta_adv_device = delta_adv_device.reshape(-1, delta_adv_device.shape[-1])
        adversarial_perturbation_2d = F.linear(x_reshaped, delta_adv_device)
        # 4. Reshape the perturbation back to the original output's 3D shape
        # The output shape will be (batch, seq_len, out_features)
        adversarial_perturbation = adversarial_perturbation_2d.view(
            *original_shape[:-1], self.out_features
        )

        # --- CRITICAL FIX END ---

        # 5. Add the perturbation
        assert adversarial_perturbation.shape == original_output.shape, \
            f"Shape mismatch: Perturbation is {adversarial_perturbation.shape}, Output is {original_output.shape}"
            
        return original_output + adversarial_perturbation
def apply_adversarial_wrappers(model, lora_weights_dict: dict):
    """
    Replaces target PEFT layers with the AdversarialPeftWrapper.
    This version uses the correct LoRA matrix mathematics AND robust module retrieval.
    """
    original_modules = {}
    # The 'name' variable is the full, correct path to the module.
    for name, (U_gen, V_gen) in lora_weights_dict.items():
        try:
            original_peft_layer = model.get_submodule(name)
            parent_name = ".".join(name.split(".")[:-1])
            module_name = name.split(".")[-1]
            parent_module = model.get_submodule(parent_name)
            delta_adv = V_gen @ U_gen  # (B, in_dim, r) @ (B, r, out_dim) -> (B, in_dim, out_dim)

            expected_shape = (original_peft_layer.out_features, original_peft_layer.in_features)
            delta_adv_shape = (delta_adv.shape[-2], delta_adv.shape[-1])
            if delta_adv_shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for {name}: delta_adv has shape {delta_adv.shape}, "
                    f"but layer expects {expected_shape}."
                )    
            wrapped_layer = AdversarialPeftWrapper(original_peft_layer, delta_adv)
            setattr(parent_module, module_name, wrapped_layer)
            original_modules[name] = original_peft_layer
            
        except AttributeError:
            print(f"Warning: Could not find submodule for name '{name}'. Skipping.")
        except Exception as e:
            print(f"Error wrapping module {name}: {e}")
            
    return original_modules

# The 'restore_original_modules' function should be updated similarly for consistency and robustness.

def restore_original_modules(model, original_modules: dict):
    """Restores the original PEFT layers using robust retrieval."""
    for name, original_module in original_modules.items():
        try:
            # We still need the parent to perform the setattr operation
            parent_name = ".".join(name.split(".")[:-1])
            module_name = name.split(".")[-1]
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, module_name, original_module)
        except AttributeError:
            print(f"Warning: Could not find parent module to restore '{name}'. Skipping.")
        except Exception as e:
            print(f"Error restoring module {name}: {e}")