import torch
import torch.nn as nn
from collections import defaultdict

class ActivationCache:
    def __init__(self, model: nn.Module, target_modules: list, reshape: bool = True, capture_output: bool = False):
        self.model = model
        self.target_modules = target_modules
        self.reshape = reshape
        self.capture_output = capture_output
        self.activations = {}  # Simple dict: {module_name: tensor}
        self.hooks = []
        self.device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device('cpu')

    def _get_hook(self, module_name: str, is_output: bool = False):
        def hook(module, input, output):
            tensor = output if is_output else input[0]
            if self.reshape:
                tensor = tensor.reshape(-1, tensor.shape[-1])
            self.activations[module_name] = tensor.detach().to(self.device)  # Keep on same device for efficiency
        return hook

    def register_hooks(self):
        registered_count = 0
        
        for name, module in self.model.named_modules():
            # Get the actual module name (last part of the full name)
            module_parts = name.split('.')
            module_name = module_parts[-1]
            
            # Check if this module is in our target list
            if module_name in self.target_modules:
                hook_fn = self._get_hook(name, self.capture_output)
                self.hooks.append(module.register_forward_hook(hook_fn))
                self.activations[name] = None  # Pre-register key
                registered_count += 1

    def clear_cache(self):
        """Clear only the activation values, keep the keys"""
        for key in self.activations:
            self.activations[key] = None

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.remove_hooks()
        pass