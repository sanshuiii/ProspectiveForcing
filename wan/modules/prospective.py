"""Shared recurrent future-feature draft module for Prospective Forcing."""

import torch
from torch import nn


class ProspectiveFutureDraft(nn.Module):
    """Predict one future latent chunk feature per recurrent application.

    WAN flattens a chunk in frame-major order.  Attention is intentionally
    restricted to the short temporal axis for each spatial token, keeping the
    phase-1 auxiliary branch small enough to train with the full DMD pipeline.
    The same cell is reused for every future horizon.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        max_horizons: int = 2,
        num_frames_per_chunk: int = 3,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if max_horizons < 1:
            raise ValueError("max_horizons must be positive")

        self.dim = dim
        self.max_horizons = max_horizons
        self.num_frames_per_chunk = num_frames_per_chunk
        hidden_dim = int(dim * mlp_ratio)

        self.horizon_embedding = nn.Embedding(max_horizons, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=0.0, batch_first=True
        )
        self.noise_gate = nn.Linear(dim, dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, dim),
        )
        # Start as a conservative residual update instead of overwriting the
        # already useful ODE-initialized backbone feature.
        self.residual_logit = nn.Parameter(torch.tensor([-2.0]))

    def forward(
        self,
        previous_feature: torch.Tensor,
        future_noise_feature: torch.Tensor,
        horizon_index: int,
    ) -> torch.Tensor:
        if previous_feature.shape != future_noise_feature.shape:
            raise ValueError(
                "previous and future-noise features must have identical shapes; "
                f"got {previous_feature.shape} and {future_noise_feature.shape}"
            )
        if previous_feature.ndim != 3:
            raise ValueError("features must have shape [batch, tokens, dim]")
        if not 0 <= horizon_index < self.max_horizons:
            raise ValueError(
                f"horizon_index={horizon_index} outside [0, {self.max_horizons})"
            )

        batch, tokens, dim = previous_feature.shape
        if dim != self.dim or tokens % self.num_frames_per_chunk != 0:
            raise ValueError(
                f"expected dim={self.dim} and tokens divisible by "
                f"{self.num_frames_per_chunk}; got {previous_feature.shape}"
            )

        horizon = self.horizon_embedding.weight[horizon_index].view(1, 1, dim)
        fused = self.input_norm(previous_feature + future_noise_feature + horizon)

        spatial_tokens = tokens // self.num_frames_per_chunk
        temporal = fused.view(
            batch, self.num_frames_per_chunk, spatial_tokens, dim
        ).permute(0, 2, 1, 3).reshape(
            batch * spatial_tokens, self.num_frames_per_chunk, dim
        )
        temporal = self.temporal_norm(temporal)
        attended, _ = self.temporal_attn(
            temporal, temporal, temporal, need_weights=False
        )
        attended = attended.view(
            batch, spatial_tokens, self.num_frames_per_chunk, dim
        ).permute(0, 2, 1, 3).reshape(batch, tokens, dim)

        gate = torch.sigmoid(self.noise_gate(future_noise_feature + horizon))
        scale = torch.sigmoid(self.residual_logit)
        state = previous_feature + scale * gate * attended
        return state + scale * self.mlp(self.mlp_norm(state))


class ProspectiveParallelFutureDraft(nn.Module):
    """Predict every horizon from the same anchor with an independent head."""

    parallel_horizons = True

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        max_horizons: int = 2,
        num_frames_per_chunk: int = 3,
    ) -> None:
        super().__init__()
        if max_horizons < 1:
            raise ValueError("max_horizons must be positive")
        self.max_horizons = max_horizons
        self.heads = nn.ModuleList([
            ProspectiveFutureDraft(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                max_horizons=1,
                num_frames_per_chunk=num_frames_per_chunk,
            )
            for _ in range(max_horizons)
        ])

    def forward(
        self,
        anchor_feature: torch.Tensor,
        future_noise_feature: torch.Tensor,
        horizon_index: int,
    ) -> torch.Tensor:
        if not 0 <= horizon_index < self.max_horizons:
            raise ValueError(
                f"horizon_index={horizon_index} outside [0, {self.max_horizons})"
            )
        return self.heads[horizon_index](
            anchor_feature,
            future_noise_feature,
            horizon_index=0,
        )


class ProspectiveContextFutureDraft(ProspectiveFutureDraft):
    """Predict one future chunk from the last ``C`` hidden chunks.

    The output query remains the E2 last chunk plus the horizon noise.  Earlier
    chunks only extend the temporal key/value context.  With ``C=1`` this
    reduces exactly to :class:`ProspectiveFutureDraft`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        max_horizons: int = 2,
        num_frames_per_chunk: int = 3,
        context_num_chunks: int = 1,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            max_horizons=max_horizons,
            num_frames_per_chunk=num_frames_per_chunk,
        )
        if context_num_chunks < 1:
            raise ValueError("context_num_chunks must be positive")
        self.context_num_chunks = context_num_chunks

    def forward(
        self,
        previous_feature: torch.Tensor,
        future_noise_feature: torch.Tensor,
        horizon_index: int,
    ) -> torch.Tensor:
        if previous_feature.ndim != 3 or future_noise_feature.ndim != 3:
            raise ValueError("features must have shape [batch, tokens, dim]")
        if previous_feature.shape[0] != future_noise_feature.shape[0]:
            raise ValueError("context and future noise must share the batch size")
        if (
            previous_feature.shape[-1] != self.dim
            or future_noise_feature.shape[-1] != self.dim
        ):
            raise ValueError(f"all features must use dim={self.dim}")
        if not 0 <= horizon_index < self.max_horizons:
            raise ValueError(
                f"horizon_index={horizon_index} outside [0, {self.max_horizons})"
            )

        batch, context_tokens, dim = previous_feature.shape
        context_frames = self.context_num_chunks * self.num_frames_per_chunk
        if context_tokens % context_frames != 0:
            raise ValueError(
                f"context tokens must be divisible by {context_frames}; "
                f"got {context_tokens}"
            )
        spatial_tokens = context_tokens // context_frames
        expected_noise_tokens = self.num_frames_per_chunk * spatial_tokens
        if future_noise_feature.shape[1] != expected_noise_tokens:
            raise ValueError(
                f"expected {expected_noise_tokens} future-noise tokens, got "
                f"{future_noise_feature.shape[1]}"
            )

        anchor = previous_feature.view(
            batch, context_frames, spatial_tokens, dim
        )
        noise = future_noise_feature.view(
            batch, self.num_frames_per_chunk, spatial_tokens, dim
        )
        horizon = self.horizon_embedding.weight[horizon_index].view(
            1, 1, 1, dim
        )

        prefix = anchor[:, :-self.num_frames_per_chunk] + horizon
        last = anchor[:, -self.num_frames_per_chunk:] + noise + horizon
        fused_context = torch.cat([prefix, last], dim=1)
        fused_context = self.input_norm(fused_context)
        query = fused_context[:, -self.num_frames_per_chunk:]

        query = query.permute(0, 2, 1, 3).reshape(
            batch * spatial_tokens, self.num_frames_per_chunk, dim
        )
        context = fused_context.permute(0, 2, 1, 3).reshape(
            batch * spatial_tokens, context_frames, dim
        )
        query = self.temporal_norm(query)
        context = self.temporal_norm(context)
        attended, _ = self.temporal_attn(
            query, context, context, need_weights=False
        )
        attended = attended.view(
            batch, spatial_tokens, self.num_frames_per_chunk, dim
        ).permute(0, 2, 1, 3).reshape(
            batch, expected_noise_tokens, dim
        )

        last_feature = previous_feature[:, -expected_noise_tokens:]
        flat_horizon = horizon.view(1, 1, dim)
        gate = torch.sigmoid(
            self.noise_gate(future_noise_feature + flat_horizon)
        )
        scale = torch.sigmoid(self.residual_logit)
        state = last_feature + scale * gate * attended
        return state + scale * self.mlp(self.mlp_norm(state))


class ProspectiveParallelContextFutureDraft(nn.Module):
    """Independent E2 heads with a configurable hidden-context length."""

    parallel_horizons = True

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        max_horizons: int = 2,
        num_frames_per_chunk: int = 3,
        context_num_chunks: int = 1,
    ) -> None:
        super().__init__()
        if max_horizons < 1:
            raise ValueError("max_horizons must be positive")
        self.max_horizons = max_horizons
        self.context_num_chunks = context_num_chunks
        self.heads = nn.ModuleList([
            ProspectiveContextFutureDraft(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                max_horizons=1,
                num_frames_per_chunk=num_frames_per_chunk,
                context_num_chunks=context_num_chunks,
            )
            for _ in range(max_horizons)
        ])

    def forward(
        self,
        anchor_feature: torch.Tensor,
        future_noise_feature: torch.Tensor,
        horizon_index: int,
    ) -> torch.Tensor:
        if not 0 <= horizon_index < self.max_horizons:
            raise ValueError(
                f"horizon_index={horizon_index} outside [0, {self.max_horizons})"
            )
        return self.heads[horizon_index](
            anchor_feature, future_noise_feature, horizon_index=0
        )


class ProspectiveParallelMultiLayerFutureDraft(nn.Module):
    """Next-Forcing-style multi-layer fusion feeding independent E2 heads."""

    parallel_horizons = True
    expects_multilayer_features = True

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        max_horizons: int = 2,
        num_frames_per_chunk: int = 3,
        feature_tap_layers=(5, 16, 27, 40),
    ) -> None:
        super().__init__()
        taps = tuple(int(layer) for layer in feature_tap_layers)
        if not taps or any(layer < 1 for layer in taps):
            raise ValueError("feature_tap_layers must be positive and non-empty")
        if len(set(taps)) != len(taps) or tuple(sorted(taps)) != taps:
            raise ValueError("feature_tap_layers must be sorted and unique")
        if max_horizons < 1:
            raise ValueError("max_horizons must be positive")
        self.max_horizons = max_horizons
        self.feature_tap_layers = taps
        fused_dim = len(taps) * dim
        self.expected_input_dim = fused_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(fused_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.heads = nn.ModuleList([
            ProspectiveFutureDraft(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                max_horizons=1,
                num_frames_per_chunk=num_frames_per_chunk,
            )
            for _ in range(max_horizons)
        ])

    def forward(
        self,
        anchor_feature: torch.Tensor,
        future_noise_feature: torch.Tensor,
        horizon_index: int,
    ) -> torch.Tensor:
        if anchor_feature.ndim != 3:
            raise ValueError("multi-layer features must have shape [B, T, L*d]")
        if anchor_feature.shape[-1] != self.expected_input_dim:
            raise ValueError(
                "multi-layer feature dimension does not match configured taps"
            )
        if not 0 <= horizon_index < self.max_horizons:
            raise ValueError(
                f"horizon_index={horizon_index} outside [0, {self.max_horizons})"
            )
        fused = self.feature_fusion(anchor_feature)
        return self.heads[horizon_index](
            fused, future_noise_feature, horizon_index=0
        )


class ProspectiveFutureSuffixDraft(nn.Module):
    """Predict the last ``O`` chunks of one future rolling window.

    Only the final E2 hidden chunk is used as the backbone context.  A head at
    future offset ``h`` repeats that anchor for the still-known suffix slots
    and repeats its own next-noise condition for the ``h`` entering slots.
    This preserves E2's independent-head rule: q2 never consumes q1.
    """

    suffix_prediction = True

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        future_offset_chunks: int = 1,
        output_num_chunks: int = 3,
        num_frames_per_chunk: int = 3,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if output_num_chunks < 1:
            raise ValueError("output_num_chunks must be positive")
        if not 1 <= future_offset_chunks <= output_num_chunks:
            raise ValueError(
                "future_offset_chunks must be in "
                f"[1, {output_num_chunks}]; got {future_offset_chunks}"
            )
        if num_frames_per_chunk < 1:
            raise ValueError("num_frames_per_chunk must be positive")

        self.dim = dim
        self.future_offset_chunks = future_offset_chunks
        self.output_num_chunks = output_num_chunks
        self.num_frames_per_chunk = num_frames_per_chunk
        self.output_frames = output_num_chunks * num_frames_per_chunk
        hidden_dim = int(dim * mlp_ratio)

        self.output_frame_embedding = nn.Embedding(self.output_frames, dim)
        self.context_frame_embedding = nn.Embedding(
            2 * num_frames_per_chunk, dim
        )
        self.horizon_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.temporal_cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=0.0, batch_first=True
        )
        self.noise_gate = nn.Linear(dim, dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, dim),
        )
        self.residual_logit = nn.Parameter(torch.tensor([-2.0]))

    def forward(
        self,
        anchor_feature: torch.Tensor,
        future_noise_feature: torch.Tensor,
    ) -> torch.Tensor:
        if anchor_feature.ndim != 3 or future_noise_feature.ndim != 3:
            raise ValueError("features must have shape [batch, tokens, dim]")
        if anchor_feature.shape != future_noise_feature.shape:
            raise ValueError(
                "E2 anchor and future-noise chunks must have identical shapes"
            )
        batch, anchor_tokens, dim = anchor_feature.shape
        if dim != self.dim or anchor_tokens % self.num_frames_per_chunk != 0:
            raise ValueError(
                f"expected dim={self.dim} and tokens divisible by "
                f"{self.num_frames_per_chunk}; got {anchor_feature.shape}"
            )

        spatial_tokens = anchor_tokens // self.num_frames_per_chunk
        anchor = anchor_feature.view(
            batch, self.num_frames_per_chunk, spatial_tokens, dim
        )
        noise = future_noise_feature.view(
            batch, self.num_frames_per_chunk, spatial_tokens, dim
        )
        known_chunks = self.output_num_chunks - self.future_offset_chunks
        base_parts = []
        if known_chunks:
            base_parts.append(anchor.repeat(1, known_chunks, 1, 1))
        base_parts.append(
            noise.repeat(1, self.future_offset_chunks, 1, 1)
        )
        base = torch.cat(base_parts, dim=1)

        output_position = self.output_frame_embedding.weight.view(
            1, self.output_frames, 1, dim
        )
        query = (
            base
            + output_position
            + self.horizon_embedding.view(1, 1, 1, dim)
        )
        context = torch.cat([anchor, noise], dim=1)
        context_position = self.context_frame_embedding.weight.view(
            1, 2 * self.num_frames_per_chunk, 1, dim
        )
        context = context + context_position

        query = query.permute(0, 2, 1, 3).reshape(
            batch * spatial_tokens, self.output_frames, dim
        )
        context = context.permute(0, 2, 1, 3).reshape(
            batch * spatial_tokens, 2 * self.num_frames_per_chunk, dim
        )
        normalized_context = self.context_norm(context)
        attended, _ = self.temporal_cross_attn(
            self.query_norm(query),
            normalized_context,
            normalized_context,
            need_weights=False,
        )
        attended = attended.view(
            batch, spatial_tokens, self.output_frames, dim
        ).permute(0, 2, 1, 3)

        gate_noise_parts = []
        if known_chunks:
            gate_noise_parts.append(noise[:, :1].expand(
                -1,
                known_chunks * self.num_frames_per_chunk,
                -1,
                -1,
            ))
        gate_noise_parts.append(
            noise.repeat(1, self.future_offset_chunks, 1, 1)
        )
        gate_noise = torch.cat(gate_noise_parts, dim=1)
        gate = torch.sigmoid(self.noise_gate(gate_noise + output_position))
        scale = torch.sigmoid(self.residual_logit)
        state = base + scale * gate * attended
        state = state + scale * self.mlp(self.mlp_norm(state))
        return state.reshape(
            batch, self.output_frames * spatial_tokens, dim
        )


class ProspectiveParallelFutureSuffixDraft(nn.Module):
    """Independent E2-context heads predicting a configurable suffix."""

    parallel_horizons = True
    suffix_prediction = True

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        max_horizons: int = 2,
        num_frames_per_chunk: int = 3,
        output_num_chunks: int = 3,
    ) -> None:
        super().__init__()
        if max_horizons < 1:
            raise ValueError("max_horizons must be positive")
        if max_horizons > output_num_chunks:
            raise ValueError(
                "future horizons cannot exceed the predicted suffix length"
            )
        self.max_horizons = max_horizons
        self.output_num_chunks = output_num_chunks
        self.heads = nn.ModuleList([
            ProspectiveFutureSuffixDraft(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                future_offset_chunks=horizon + 1,
                output_num_chunks=output_num_chunks,
                num_frames_per_chunk=num_frames_per_chunk,
            )
            for horizon in range(max_horizons)
        ])

    def forward(
        self,
        anchor_feature: torch.Tensor,
        future_noise_feature: torch.Tensor,
        horizon_index: int,
    ) -> torch.Tensor:
        if not 0 <= horizon_index < self.max_horizons:
            raise ValueError(
                f"horizon_index={horizon_index} outside [0, {self.max_horizons})"
            )
        return self.heads[horizon_index](
            anchor_feature, future_noise_feature
        )


class ProspectiveFutureWindowDraft(nn.Module):
    """Predict one complete future rolling window for one fixed horizon.

    The anchor contains ``window_num_chunks`` complete chunks.  The future
    noise contains the single newly entering chunk associated with this head.
    Attention is independent for every spatial token and uses the complete
    anchor window as context.  A shifted anchor plus the noise-conditioned
    slots form the output queries.
    """

    window_prediction = True

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        future_offset_chunks: int = 1,
        window_num_chunks: int = 5,
        num_frames_per_chunk: int = 3,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if not 1 <= future_offset_chunks < window_num_chunks:
            raise ValueError(
                "future_offset_chunks must be in "
                f"[1, {window_num_chunks}); got {future_offset_chunks}"
            )
        if num_frames_per_chunk < 1:
            raise ValueError("num_frames_per_chunk must be positive")

        self.dim = dim
        self.future_offset_chunks = future_offset_chunks
        self.window_num_chunks = window_num_chunks
        self.num_frames_per_chunk = num_frames_per_chunk
        self.window_frames = window_num_chunks * num_frames_per_chunk
        self.future_offset_frames = future_offset_chunks * num_frames_per_chunk
        hidden_dim = int(dim * mlp_ratio)

        self.output_frame_embedding = nn.Embedding(self.window_frames, dim)
        self.context_frame_embedding = nn.Embedding(
            self.window_frames + num_frames_per_chunk, dim
        )
        self.horizon_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.temporal_cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=0.0, batch_first=True
        )
        self.noise_gate = nn.Linear(dim, dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, dim),
        )
        self.residual_logit = nn.Parameter(torch.tensor([-2.0]))

    def forward(
        self,
        anchor_feature: torch.Tensor,
        future_noise_feature: torch.Tensor,
    ) -> torch.Tensor:
        if anchor_feature.ndim != 3 or future_noise_feature.ndim != 3:
            raise ValueError("features must have shape [batch, tokens, dim]")
        if anchor_feature.shape[0] != future_noise_feature.shape[0]:
            raise ValueError("anchor and future noise must have the same batch size")
        if (
            anchor_feature.shape[-1] != self.dim
            or future_noise_feature.shape[-1] != self.dim
        ):
            raise ValueError(f"both inputs must use dim={self.dim}")
        if anchor_feature.shape[1] % self.window_frames != 0:
            raise ValueError(
                f"anchor tokens must be divisible by {self.window_frames}"
            )

        batch, anchor_tokens, dim = anchor_feature.shape
        spatial_tokens = anchor_tokens // self.window_frames
        expected_noise_tokens = self.num_frames_per_chunk * spatial_tokens
        if future_noise_feature.shape[1] != expected_noise_tokens:
            raise ValueError(
                f"expected {expected_noise_tokens} future-noise tokens, got "
                f"{future_noise_feature.shape[1]}"
            )

        anchor = anchor_feature.view(
            batch, self.window_frames, spatial_tokens, dim
        )
        noise = future_noise_feature.view(
            batch, self.num_frames_per_chunk, spatial_tokens, dim
        )

        # Horizon one needs one new chunk; horizon two needs two output chunks.
        # W2 deliberately keeps the E2 independence rule, so the second head
        # fills both unknown query slots from its own n2 condition.  Learned
        # output positions distinguish those slots.
        repeats = self.future_offset_frames // self.num_frames_per_chunk
        expanded_noise = noise.repeat(1, repeats, 1, 1)
        base = torch.cat(
            [anchor[:, self.future_offset_frames:], expanded_noise], dim=1
        )
        if base.shape[1] != self.window_frames:
            raise RuntimeError(f"invalid window query length: {base.shape[1]}")

        output_position = self.output_frame_embedding.weight.view(
            1, self.window_frames, 1, dim
        )
        query = base + output_position + self.horizon_embedding.view(1, 1, 1, dim)

        context = torch.cat([anchor, noise], dim=1)
        context_position = self.context_frame_embedding.weight.view(
            1, self.window_frames + self.num_frames_per_chunk, 1, dim
        )
        context = context + context_position

        query = query.permute(0, 2, 1, 3).reshape(
            batch * spatial_tokens, self.window_frames, dim
        )
        context = context.permute(0, 2, 1, 3).reshape(
            batch * spatial_tokens,
            self.window_frames + self.num_frames_per_chunk,
            dim,
        )
        attended, _ = self.temporal_cross_attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        attended = attended.view(
            batch, spatial_tokens, self.window_frames, dim
        ).permute(0, 2, 1, 3)

        expanded_gate_noise = expanded_noise
        if self.future_offset_frames < self.window_frames:
            prefix = expanded_noise[:, :1].expand(
                -1, self.window_frames - self.future_offset_frames, -1, -1
            )
            expanded_gate_noise = torch.cat([prefix, expanded_noise], dim=1)
        gate = torch.sigmoid(
            self.noise_gate(expanded_gate_noise + output_position)
        )
        scale = torch.sigmoid(self.residual_logit)
        state = base + scale * gate * attended
        state = state + scale * self.mlp(self.mlp_norm(state))
        return state.reshape(batch, anchor_tokens, dim)


class ProspectiveParallelFutureWindowDraft(nn.Module):
    """Two independent E2-style heads, each predicting a complete window."""

    parallel_horizons = True
    window_prediction = True

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        max_horizons: int = 2,
        num_frames_per_chunk: int = 3,
        window_num_chunks: int = 5,
    ) -> None:
        super().__init__()
        if max_horizons < 1:
            raise ValueError("max_horizons must be positive")
        if max_horizons >= window_num_chunks:
            raise ValueError("future horizons must be shorter than the window")
        self.max_horizons = max_horizons
        self.heads = nn.ModuleList([
            ProspectiveFutureWindowDraft(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                future_offset_chunks=horizon + 1,
                window_num_chunks=window_num_chunks,
                num_frames_per_chunk=num_frames_per_chunk,
            )
            for horizon in range(max_horizons)
        ])

    def forward(
        self,
        anchor_feature: torch.Tensor,
        future_noise_feature: torch.Tensor,
        horizon_index: int,
    ) -> torch.Tensor:
        if not 0 <= horizon_index < self.max_horizons:
            raise ValueError(
                f"horizon_index={horizon_index} outside [0, {self.max_horizons})"
            )
        return self.heads[horizon_index](
            anchor_feature, future_noise_feature
        )
