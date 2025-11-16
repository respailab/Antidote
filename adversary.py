import torch 
import torch.nn as nn 
import torch.nn.functional as F 
class Adversary(nn.Module):
    def __init__(self, r, layer_configs: dict, enc_dim = 1024, num_heads = 16):
        """
        Initializes a general adversary for heterogeneous layers.
        :param r: LoRA rank.
        :param config: A config object with internal dimensions.
        :param layer_configs: A dict mapping a layer_type_name (e.g., "q_proj_3584_3584") 
                              to its (in_features, out_features) tuple.
        """
        super().__init__()
        self.r = r
        self.layer_configs = layer_configs
        
        # --- 1. Specialized INPUT Projection Heads ---
        # Projects from variable in_features -> fixed D_INTERNAL
        self.input_projs = nn.ModuleDict()
        unique_in_dims = set(cfg[0] for cfg in layer_configs.values())
        for in_dim in unique_in_dims:
            self.input_projs[str(in_dim)] = nn.Linear(in_dim, enc_dim)

        # --- 2. The SHARED CORE network (unchanged) ---
        self.aggregator = nn.MultiheadAttention(embed_dim=enc_dim, num_heads=num_heads) 
        self.body = nn.Sequential(
            ResidualFFN(enc_dim),
            ResidualFFN(enc_dim),
        )

        # --- 3. Specialized OUTPUT Heads (CRITICAL CHANGE) ---
        # Each head must generate U and V with the correct shapes.
        # U (LoRA A) -> (r, in_features)
        # V (LoRA B) -> (out_features, r)
        self.U_heads = nn.ModuleDict()
        self.V_heads = nn.ModuleDict()

        for config_name, (in_dim, out_dim) in layer_configs.items():
            # U_head projects from mlp_hidden_dim -> r * in_features
            self.U_heads[config_name] = nn.Linear(enc_dim, r * in_dim)
            # V_head projects from mlp_hidden_dim -> out_features * r
            self.V_heads[config_name] = nn.Linear(enc_dim, out_dim * r)
            
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, activations: torch.Tensor, config_name: str):
        in_dim, out_dim = self.layer_configs[config_name]
        # activations are of shape (B, L, V)

        # 1. Select correct INPUT head
        input_proj = self.input_projs[str(in_dim)]
        x = input_proj(activations) # (B, L, enc_dim)
        
        # 2. Process through SHARED CORE
        x_attended, _ = self.aggregator(x, x, x)  # (B, L, enc_dim)
        x_pooled = x_attended.mean(dim=0, keepdim=True)  # (1, L, enc_dim)

        # 3. Select correct OUTPUT heads
        U_head = self.U_heads[config_name]
        V_head = self.V_heads[config_name]
        
        U_flat = U_head(x_pooled) 
        V_flat = V_head(x_pooled)
        
        # 4. Reshape to final LoRA matrices with correct shapes
        U = U_flat.view(-1, self.r, in_dim)       # Shape: (L, r, in_dim)
        V = V_flat.view(-1, out_dim, self.r)      # Shape: (L, out_dim, r)
        
        return U, V

class ResidualFFN(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.up_proj = nn.Linear(dim, dim * 3)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(dim)
        self.down_proj = nn.Linear(dim * 3, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply layer normalization first
        x_norm = self.ln(x)
        # Feed-forward transformation
        o1 = self.up_proj(x_norm)
        o2 = self.gelu(o1)
        o3 = self.down_proj(o2)
        o3 = self.dropout(o3)
        # Residual connection (adding normalized input transformation to original input)
        return x + o3