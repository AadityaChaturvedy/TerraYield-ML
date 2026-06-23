import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) implementation.
    """
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)

    def forward(self, x):
        x = self.linear(x)
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)

class GRN(nn.Module):
    """
    Gated Residual Network (GRN) block.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None:
            self.context_projection = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_projection = None
            
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.glu = GLU(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        if input_dim != output_dim:
            self.skip_projection = nn.Linear(input_dim, output_dim)
        else:
            self.skip_projection = None
            
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x, context=None):
        # x: [..., input_dim]
        # context: [..., context_dim]
        h = self.linear1(x)
        if context is not None and self.context_projection is not None:
            h = h + self.context_projection(context)
        h = F.elu(h)
        h = self.linear2(h)
        h = self.dropout(h)
        h = self.glu(h)
        
        if self.skip_projection is not None:
            skip = self.skip_projection(x)
        else:
            skip = x
            
        return self.layer_norm(h + skip)

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) to select relevant features.
    """
    def __init__(self, num_features, input_dims, d_model, dropout=0.1, context_dim=None):
        super().__init__()
        self.num_features = num_features
        self.grns = nn.ModuleList([
            GRN(input_dims[i], d_model, d_model, dropout=dropout)
            for i in range(num_features)
        ])
        # Gating network
        self.gating = GRN(num_features * d_model, d_model, num_features, dropout=dropout, context_dim=context_dim)

    def forward(self, features_list, context=None):
        # features_list: list of length num_features, each with shape [batch, input_dim]
        # or [batch, time, input_dim]
        projected = []
        for i in range(self.num_features):
            feat = features_list[i]
            proj = self.grns[i](feat) # [..., d_model]
            projected.append(proj)
            
        # Concatenate along the last dimension
        concat = torch.cat(projected, dim=-1) # [..., num_features * d_model]
        
        # Compute weights
        weights = self.gating(concat, context) # [..., num_features]
        weights = torch.softmax(weights, dim=-1) # [..., num_features]
        weights = weights.unsqueeze(-1) # [..., num_features, 1]
        
        # Weighted sum of projected features
        stacked = torch.stack(projected, dim=-2) # [..., num_features, d_model]
        out = torch.sum(weights * stacked, dim=-2) # [..., d_model]
        
        return out, weights.squeeze(-1)

class TemporalFusionTransformer(nn.Module):
    """
    Adapted Temporal Fusion Transformer (TFT) for sequence-to-scalar crop yield prediction.
    """
    def __init__(self, 
                 state_vocab_size, 
                 district_vocab_size,
                 static_cont_dims,
                 temporal_dims,
                 d_model=64, 
                 n_heads=4, 
                 dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Embeddings for categorical static features
        self.state_emb = nn.Embedding(state_vocab_size, d_model)
        self.district_emb = nn.Embedding(district_vocab_size, d_model)
        
        # 2. Static variables list configuration
        # Categorical: State, District
        # Continuous static features (e.g., Lags, Historical Mean, Spatial Lag)
        self.num_static_features = 2 + len(static_cont_dims)
        static_input_dims = [d_model, d_model] + static_cont_dims
        
        # Static Variable Selection Network
        self.static_vsn = VariableSelectionNetwork(
            num_features=self.num_static_features,
            input_dims=static_input_dims,
            d_model=d_model,
            dropout=dropout
        )
        
        # 3. Time-varying inputs: Shared VSN across time steps (months)
        self.num_temporal_features = len(temporal_dims)
        self.temporal_vsn = VariableSelectionNetwork(
            num_features=self.num_temporal_features,
            input_dims=temporal_dims,
            d_model=d_model,
            dropout=dropout,
            context_dim=d_model # static context s
        )
        
        # 4. Sequence processing (LSTM)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.lstm_gate = GLU(d_model)
        self.lstm_norm = nn.LayerNorm(d_model)
        
        # 5. Temporal Self-Attention
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        
        # 6. Post-attention fusion and output
        self.fusion_grn = GRN(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=dropout,
            context_dim=d_model
        )
        
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, state, district, static_cont, temporal_seq):
        # state: [batch] (int)
        # district: [batch] (int)
        # static_cont: list of continuous static features, each [batch, 1]
        # temporal_seq: list of temporal features, each [batch, seq_len, feat_dim]
        
        batch_size = state.size(0)
        seq_len = temporal_seq[0].size(1)
        
        # Embed categories
        s_state = self.state_emb(state) # [batch, d_model]
        s_dist = self.district_emb(district) # [batch, d_model]
        
        # Group static features
        static_features = [s_state, s_dist] + static_cont
        
        # Static variable selection
        s_context, static_weights = self.static_vsn(static_features) # [batch, d_model]
        
        # Time-varying variable selection (applied to each time step)
        temporal_processed = []
        temporal_weights_list = []
        for t in range(seq_len):
            step_feats = [feat[:, t, :] for feat in temporal_seq] # list of [batch, feat_dim]
            step_out, step_weights = self.temporal_vsn(step_feats, context=s_context) # [batch, d_model]
            temporal_processed.append(step_out)
            temporal_weights_list.append(step_weights)
            
        temporal_seq_emb = torch.stack(temporal_processed, dim=1) # [batch, seq_len, d_model]
        
        # LSTM seq2seq processing
        h0 = s_context.unsqueeze(0) # [1, batch, d_model]
        c0 = torch.zeros_like(h0)
        lstm_out, _ = self.lstm(temporal_seq_emb, (h0, c0)) # [batch, seq_len, d_model]
        
        # Gating + Skip connection
        lstm_out = self.lstm_gate(lstm_out) # [batch, seq_len, d_model]
        lstm_out = self.lstm_norm(lstm_out + temporal_seq_emb)
        
        # Multi-head Self-Attention over time steps (months)
        attn_out, attn_weights = self.mha(lstm_out, lstm_out, lstm_out) # [batch, seq_len, d_model]
        attn_out = self.attn_norm(attn_out + lstm_out) # [batch, seq_len, d_model]
        
        # Fuse with static context at each step
        s_context_expanded = s_context.unsqueeze(1).expand(-1, seq_len, -1) # [batch, seq_len, d_model]
        fused = self.fusion_grn(attn_out, context=s_context_expanded) # [batch, seq_len, d_model]
        
        # Aggregate temporal sequence outputs
        # Average pooling over the months to get a single annual representation
        annual_representation = torch.mean(fused, dim=1) # [batch, d_model]
        
        # Predict yield
        pred_yield = self.output_layer(annual_representation) # [batch, 1]
        
        return pred_yield.squeeze(-1)
