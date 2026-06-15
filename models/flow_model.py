"""Conditional flow matching model for image-to-image translation.

Adapts the atlas-superresolution FlowModel for 2-D image data.  The two
targeted changes versus the original are:

  1. *Conditioning*: the (eta, phi, discrete_layer) detector-cell encoding is
     replaced by a continuous 2-D pixel-coordinate embedding ``pos_emb_net``.
  2. *Simplification*: all variable-length / graph-specific machinery
     (padding masks, masked means, attention masking) is removed.  Images have
     a fixed-size pixel grid so none of that overhead is needed.

Everything else — the transformer backbone, velocity head, adaLN modulation,
weight initialisation, and all tensor dimensions — is unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import TransformerEncoder
from models.diffusion_transformer import DiTEncoder
from models.attention import ScaledDotProductAttention
from models.dense import Dense
from models.utils import TimestepEmbedder, modulate

import torchdiffeq
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher


class FlowModel(nn.Module):
    """Conditional flow matching model for image-to-image translation.

    Architecture overview (dimensions from the default config)::

        Input per pixel
        ───────────────
        pos    (B, N, 2)  → pos_emb_net    → (B, N, 64)  ┐
        source (B, N, 1)  → source_emb_net → (B, N, 31)  │ cond_feat (B, N, 96)
                            raw source                (1)  ┘
        noisy_target (B, N, 1) → noisy_input_emb_net → (B, N, 64)

        Global context
        ──────────────
        time_emb  (B, 64)                           ┐ context
        mean(cond_feat, dim=1)  (B, 96)              ┘ (B, 160)

        Transformer
        ───────────
        feat_0_mlp: [cond_feat | noisy_emb] (B,N,160) → (B,N,256)
        DiTEncoder: (B,N,256) conditioned on context → (B,N,256)

        Velocity head
        ─────────────
        [transformer_out | cond_feat] (B,N,352)
        Optional adaLN modulation
        v_t_pred_net: (B,N,352) → (B,N,1)

    Batch dict keys used by ``forward`` and ``get_loss``:
        ``pos``    (B, N, 2)  normalised pixel coordinates in [-1, 1]
        ``source`` (B, N, 1)  log1p-transformed source image pixel values
        ``target`` (B, N, 1)  log1p-transformed target image pixel values
    """

    def __init__(self, model_config: dict):
        super().__init__()

        self.config = model_config
        self.n_steps = self.config['n_steps']
        self.sigma_min = self.config['sigma_min']
        self.flow_match_obj = TargetConditionalFlowMatcher(sigma=self.sigma_min)

        self.time_step_embedder = TimestepEmbedder(self.config['time_embedding_size'])
        self.h_dim = int(self.config['h_dim'])

        # context = time embedding only
        self.context_size = self.config['time_embedding_size']

        # --- per-pixel conditioning encoders ---
        pos_emb_config = dict(self.config['pos_emb'])
        pos_emb_config['context_size'] = self.context_size
        self.pos_emb_net = Dense(**pos_emb_config)

        source_emb_config = dict(self.config['source_emb'])
        source_emb_config['context_size'] = self.context_size
        self.source_emb_net = Dense(**source_emb_config)

        # cond_emb_dim: pos(64) + source_emb(31) + raw_source(1) = 96
        # Identical to atlas etaphi(32)+layer(32)+proxy_emb(31)+proxy_raw(1)
        self.cond_emb_dim = (
            pos_emb_config['output_size']
            + source_emb_config['output_size']
            + 1  # raw source pixel appended directly
        )

        # --- noisy input encoder ---
        noisy_input_emb_config = dict(self.config['noisy_input_emb'])
        noisy_input_emb_config['context_size'] = self.context_size
        self.noisy_input_emb_net = Dense(**noisy_input_emb_config)

        # context_size_plus = time_emb + global_cond = 64 + 96 = 160
        self.context_size_plus = self.context_size + self.cond_emb_dim

        # --- projection to h_dim ---
        feat_0_mlp_config = dict(self.config['feat_0_mlp'])
        if feat_0_mlp_config['input_size'] == -1:
            # pos_emb(64) + source_emb(31) + raw_source(1) + noisy_emb(64) = 160
            feat_0_mlp_config['input_size'] = (
                pos_emb_config['output_size']
                + source_emb_config['output_size']
                + 1
                + noisy_input_emb_config['output_size']
            )
        feat_0_mlp_config['context_size'] = self.context_size_plus
        self.feat_0_mlp = Dense(**feat_0_mlp_config)

        # --- transformer backbone (full self-attention, no masking) ---
        transformer_cfg = self.config['transformer']
        mha_config = {
            'num_heads': transformer_cfg['num_heads'],
            'attention': ScaledDotProductAttention(),
        }
        if transformer_cfg['type'] == 'DiT':
            self.transformer = DiTEncoder(
                embed_dim=self.h_dim,
                num_layers=transformer_cfg['num_transformer_layers'],
                mha_config=mha_config,
                dense_config=transformer_cfg['dense_config'],
                context_dim=self.context_size_plus,
            )
        elif transformer_cfg['type'] == 'GPT-2+Normformer':
            self.transformer = TransformerEncoder(
                embed_dim=self.h_dim,
                num_layers=transformer_cfg['num_transformer_layers'],
                mha_config=mha_config,
                dense_config=transformer_cfg['dense_config'],
                context_dim=self.context_size_plus,
            )
        else:
            raise ValueError(f"Unknown transformer type: {transformer_cfg['type']}")

        # --- velocity head: h_dim + cond_emb_dim = 256 + 96 = 352 ---
        self.v_t_input_dim = self.h_dim + self.cond_emb_dim
        if self.config.get('final_modulation', False):
            self.norm_v_t = nn.LayerNorm(self.v_t_input_dim)
            self.v_t_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.context_size_plus, 2 * self.v_t_input_dim, bias=True),
            )

        v_t_pred_config = dict(self.config['v_t_pred'])
        v_t_pred_config['input_size'] = self.v_t_input_dim
        v_t_pred_config['context_size'] = self.context_size_plus
        self.v_t_pred_net = Dense(**v_t_pred_config)

        self.initialize_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_weights(self):
        init_cfg = self.config['init_weights']

        if init_cfg.get('all_linear') == 'xavier_uniform':
            def _xavier(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            self.apply(_xavier)

        if init_cfg.get('time_step_embedder') == 'normal':
            nn.init.normal_(self.time_step_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.time_step_embedder.mlp[2].weight, std=0.02)

        if init_cfg.get('ln_modulation') == 'zero':
            for layer in self.transformer.layers:
                nn.init.constant_(layer.adaLN_modulation[1].weight, 0)
                nn.init.constant_(layer.adaLN_modulation[1].bias, 0)
            if self.config.get('final_modulation', False):
                nn.init.constant_(self.v_t_adaLN_modulation[1].weight, 0)
                nn.init.constant_(self.v_t_adaLN_modulation[1].bias, 0)

        if init_cfg.get('v_t_pred_linear') == 'zero':
            nn.init.constant_(self.v_t_pred_net.net[-1].weight, 0)
            nn.init.constant_(self.v_t_pred_net.net[-1].bias, 0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict,
        noisy_input: torch.Tensor,
        time_step: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the velocity field v_t.

        Args:
            batch:       dict with keys 'pos' and 'source'
            noisy_input: (B, N, 1)  noisy target pixels at time t
            time_step:   (B,)       scalar t in [0, 1]

        Returns:
            v_t: (B, N, 1) predicted velocity
        """
        time_emb = self.time_step_embedder(time_step)  # (B, time_emb_size)

        pos    = batch['pos']     # (B, N, 2)
        source = batch['source']  # (B, N, 1)

        # Per-pixel conditioning features
        pos_emb    = self.pos_emb_net(pos, context=time_emb)       # (B, N, 64)
        source_emb = self.source_emb_net(source, context=time_emb)  # (B, N, 31)
        cond_feat  = torch.cat([pos_emb, source_emb, source], dim=-1)  # (B, N, 96)

        # Global context: simple mean over all pixels (no masking needed)
        cond_feat_global = cond_feat.mean(dim=1)  # (B, 96)

        # Encode noisy input
        noisy_input_emb = self.noisy_input_emb_net(noisy_input, context=time_emb)  # (B, N, 64)

        # Global conditioning vector: [time | mean_cond]
        context = torch.cat([time_emb, cond_feat_global], dim=-1)  # (B, 160)

        # Project to transformer dimension
        feat = self.feat_0_mlp(
            torch.cat([cond_feat, noisy_input_emb], dim=-1),
            context=context,
        )  # (B, N, 256)

        # Full self-attention over all pixels (no attention masking)
        feat = self.transformer(q=feat, context=context)  # (B, N, 256)

        # Skip connection: append per-pixel conditioning
        feat = torch.cat([feat, cond_feat], dim=-1)  # (B, N, 352)

        if self.config.get('final_modulation', False):
            v_t_shift, v_t_scale = self.v_t_adaLN_modulation(context).chunk(2, dim=1)
            feat = modulate(self.norm_v_t(feat), v_t_shift, v_t_scale)

        return self.v_t_pred_net(feat, context=context)  # (B, N, 1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def get_loss(self, batch: dict) -> tuple:
        """Flow matching loss.

        Returns:
            loss:          scalar mean MSE over all pixels
            stats:         dict of scalar diagnostics
            loss_detached: per-element loss (B, N, 1), detached
        """
        target = batch['target']  # (B, N, 1)
        x_0 = torch.randn_like(target)
        t, xt, ut = self.flow_match_obj.sample_location_and_conditional_flow(x_0, target, t=None)

        vt = self.forward(batch, xt, t)
        loss_per = F.mse_loss(vt, ut, reduction='none')  # (B, N, 1)

        if not torch.isfinite(loss_per).all():
            raise RuntimeError("Non-finite loss detected")

        stats = {
            'ut_mean': ut.mean().item(), 'ut_std': ut.std().item(),
            'vt_mean': vt.mean().item(), 'vt_std': vt.std().item(),
            'loss_mean': loss_per.mean().item(), 'loss_max': loss_per.max().item(),
        }

        return loss_per.mean(), stats, loss_per.detach()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_samples(
        self,
        batch: dict,
        n_steps: int = None,
        method: str = 'dopri5',
    ) -> torch.Tensor:
        """ODE-integrate from Gaussian noise to the target distribution.

        Returns:
            x_1: (B, N, 1) predicted target pixels in transformed space
        """
        if n_steps is None:
            n_steps = self.n_steps

        source = batch['source']
        x_1 = torchdiffeq.odeint(
            lambda t, x_t: self.forward(
                batch, x_t,
                time_step=t * torch.ones(x_t.shape[0], device=source.device),
            ),
            torch.randn_like(source),
            torch.linspace(0, 1, n_steps, device=source.device),
            method=method,
            atol=1e-4,
            rtol=1e-4,
        )
        return x_1[-1]  # (B, N, 1)
