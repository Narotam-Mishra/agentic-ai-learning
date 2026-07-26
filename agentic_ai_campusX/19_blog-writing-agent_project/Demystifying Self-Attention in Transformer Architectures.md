# Demystifying Self-Attention in Transformer Architectures

> **[IMAGE GENERATION FAILED]** Connectivity and path lengths for RNNs, CNNs, and self-attention over a 5-token sequence. Self-attention provides direct pairwise edges between all tokens within one layer.
>
> **Alt:** Diagram comparing RNN, CNN, and self-attention connectivity patterns over a short sequence
>
> **Prompt:** Side-by-side technical diagram with three panels labeled RNN, CNN, and Self-Attention. Each panel shows five token nodes in a row (t1–t5). RNN panel: arrows only from left to right connecting consecutive tokens, with a highlighted long path from t1 to t5 going through all intermediates. CNN panel: local window connections (e.g., kernel size 3) with overlapping receptive fields; long-range interaction from t1 to t5 requires multiple stacked layers indicated schematically. Self-Attention panel: every token connected to every other token with faint directed edges, and a highlighted direct edge from t1 to t5 to emphasize constant path length. Clean, minimal, vector-style, white background, readable labels.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image\nPlease retry in 49.776915487s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '49s'}]}}


## Why Self-Attention? From RNN Limits to Transformer Intuition

RNNs and CNNs both struggle with **long-range dependencies** in sequences, but for different structural reasons.

RNNs process tokens one step at a time, passing a hidden state forward. In theory, this state can carry information arbitrarily far. In practice, gradients vanish or explode over long chains, and the model is biased toward recent context. Even gated variants (LSTMs/GRUs) still rely on a single, compressed state vector to summarize everything seen so far. If token *i* needs information from token *j* far away, that information must flow through every intermediate step.

CNNs remove sequential recurrence and allow parallel computation, but each convolution only sees a local window. Capturing long context requires stacking many layers or using dilated kernels. The “effective receptive field” grows, yet interactions between distant positions are still indirect and mediated by multiple layers of local operations.

Self-attention replaces these indirect pathways with **direct pairwise interactions** between all positions in a sequence. Each position can “look at” every other position in a single layer, computing attention weights that say “how relevant is token *j* to token *i* right now?” No matter how far apart tokens are, the path length between them (in terms of computation graph hops) is 1 per layer.

This change also transforms how we use hardware. RNNs are **inherently sequential**: each step depends on the previous hidden state, limiting both training and inference throughput. CNNs allow more parallelism across positions, but depth still introduces some sequential dependence across layers.

Self-attention, by contrast, lets you process all positions in a layer **in parallel**:

- For a given sequence, you compute queries/keys/values for all tokens at once.
- The attention matrix (all pairwise scores) is computed as large batched matrix multiplications, which map well to GPUs/TPUs.
- There’s no time-step loop inside a layer, only a depth-wise loop across layers.

The result is higher throughput and better hardware utilization, especially for long sequences where the parallelism over positions is substantial. The main trade-off is quadratic cost in sequence length for dense self-attention, which we’ll revisit later.

Conceptually, self-attention turns the sequence into a **fully connected directed graph** at each layer:

- Nodes = token representations.
- Edges = attention scores from one token to another.
- Edge weights are **content-dependent**: they’re computed from the current token embeddings, not from fixed distances or kernel shapes.

Each layer refines this graph: tokens rewrite their representations as a weighted sum of information from their neighbors in this learned graph.

It’s also useful to distinguish **self-attention** from **cross-attention**:

- Self-attention: queries, keys, and values all come from the **same** sequence. This is what lets a sentence “reason about itself.”
- Cross-attention: queries come from one sequence (e.g., a decoder), keys/values from another (e.g., an encoder). This is used for sequence-to-sequence tasks like translation.

This article focuses on **self-attention within a single sequence**, because that’s the core building block that replaces RNN/CNN mechanisms for modeling contextual relationships.

One subtle but crucial property: self-attention is **permutation-equivariant** over positions. If you shuffle the tokens and apply the same self-attention layer, the outputs are shuffled in the same way. The mechanism itself does not know about order; it only knows about pairwise content relationships.

To make the model sensitive to word order or token position, we must **inject positional information separately** (e.g., via positional encodings added to the input embeddings). Without this, a Transformer could not distinguish “dog bites man” from “man bites dog” purely from self-attention.

## Unpacking the Self-Attention Mechanism: Queries, Keys, and Values

> **[IMAGE GENERATION FAILED]** Dataflow in single-head scaled dot-product self-attention: from input sequence X to Q, K, V projections, score matrix, softmax weights, and the final weighted value sums.
>
> **Alt:** Flow diagram of single-head self-attention from input X through Q, K, V projections to attention weights and output
>
> **Prompt:** Technical block diagram of single-head scaled dot-product self-attention. Start with a matrix X (shape B×L×D_model) on the left. From X, three arrows go into three linear blocks labeled W_Q, W_K, W_V producing Q, K, V (with their shapes indicated). Q and K feed into a matrix multiply block labeled Q K^T / sqrt(d_k), outputting a scores matrix (L×L). This goes into a softmax block over the last dimension, producing attention weights (L×L). The weights then multiply V in another matmul block to produce output O (B×L×d_head). Show tensor shapes near each stage and arrows indicating flow. Clean, uncluttered, white background.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image\nPlease retry in 49.3308603s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '49s'}]}}


At a single layer, self-attention takes a sequence of vectors and returns a sequence of *contextualized* vectors of the same length.

### From inputs to Q, K, V

Assume a batch of input sequences already embedded and (optionally) position-encoded:

- Shape: `X ∈ ℝ^{B × L × D_model}`
  - `B`: batch size  
  - `L`: sequence length  
  - `D_model`: model (embedding) dimension

Self-attention first computes **queries**, **keys**, and **values** via learned linear projections:

- Weight matrices:
  - `W_Q ∈ ℝ^{D_model × D_k}`
  - `W_K ∈ ℝ^{D_model × D_k}`
  - `W_V ∈ ℝ^{D_model × D_v}`

For each token vector `x_t`:

- `q_t = x_t W_Q`
- `k_t = x_t W_K`
- `v_t = x_t W_V`

In batched tensor form:

- `Q = X W_Q` → shape `B × L × D_k`
- `K = X W_K` → shape `B × L × D_k`
- `V = X W_V` → shape `B × L × D_v`

In multi-head attention, `D_k = H * d_head` and similarly for `D_v`, then you reshape to:

- `Q, K, V`: `B × H × L × d_head`

### Scaled dot-product attention scores

For a given head, attention scores are pairwise dot products between queries and keys:

- For each sequence in the batch, for each head:
  - `scores[i, j] = q_i ⋅ k_j`  for `i, j ∈ {1…L}`

In tensor notation (single head):

- `Q ∈ ℝ^{B × L × d_head}`
- `K ∈ ℝ^{B × L × d_head}`
- `scores = Q K^T` → shape `B × L × L`

To avoid large magnitudes (and thus overly peaked softmax and unstable gradients when `d_head` is large), scores are **scaled**:

- `scores = (Q K^T) / sqrt(d_head)`

This scaling keeps the variance of the scores roughly constant as `d_head` grows, preventing gradients from vanishing/exploding due to softmax saturation.

### From scores to weighted sums

To turn scores into **attention weights**, apply a row-wise softmax over the key dimension:

- For each query position `i`:
  - `α_i = softmax(scores[i, :])`
  - `α_i ∈ ℝ^{L}`, `∑_j α_i[j] = 1`

Tensor-wise:

- `α = softmax(scores, dim=-1)` → shape `B × L × L`

Then compute a weighted sum of values:

- `output[i] = ∑_j α_i[j] v_j`

In tensors:

- `V ∈ ℝ^{B × L × d_head}`
- `O = α V` → matrix multiplication over the `L` dimension
  - `O ∈ ℝ^{B × L × d_head}`

For multi-head attention, you do this per head:

- `Q, K, V ∈ ℝ^{B × H × L × d_head}`
- `scores = (Q @ K.transpose(-2, -1)) / sqrt(d_head)`  
  → `B × H × L × L`
- `α = softmax(scores, dim=-1)`  
  → `B × H × L × L`
- `O = α @ V`  
  → `B × H × L × d_head`

Then you reshape/concatenate heads:

- `O_merged ∈ ℝ^{B × L × (H * d_head)}`  
- Final linear projection with `W_O ∈ ℝ^{(H * d_head) × D_model}` brings it back to `B × L × D_model`.

### Masking: causality and padding

Before softmax, you can inject masks into `scores` to control which tokens attend to which:

- **Padding mask**: prevent attention to padded positions.
  - Mask `M_pad ∈ {0, -∞}^{B × 1 × 1 × L}` (broadcast across query positions and heads).
- **Causal mask**: enforce that position `i` can only attend to `j ≤ i`.
  - Mask `M_causal ∈ {0, -∞}^{1 × 1 × L × L}`, with `-∞` above the diagonal.

You combine and add to scores:

- `scores_masked = scores + M`

Entries with `-∞` become near-zero after softmax, effectively ignored.

### Minimal code sketch (single head)

```python
import torch
import math

def self_attention(X, W_Q, W_K, W_V, mask=None):
    # X: [B, L, D_model]
    B, L, D_model = X.shape
    d_head = W_Q.shape[-1]

    Q = X @ W_Q   # [B, L, d_head]
    K = X @ W_K   # [B, L, d_head]
    V = X @ W_V   # [B, L, d_head]

    scores = Q @ K.transpose(-2, -1)  # [B, L, L]
    scores = scores / math.sqrt(d_head)

    if mask is not None:
        # mask: [B, 1, L] or [B, L, L], pre-broadcasted to [B, L, L]
        scores = scores + mask  # e.g., 0 or -1e9 in masked positions

    weights = torch.softmax(scores, dim=-1)  # [B, L, L]
    O = weights @ V  # [B, L, d_head]
    return O
```

### Edge cases and performance notes

- **Very long sequences**: `scores` is `O(L²)` in memory and compute. Long `L` may require approximations or memory-efficient kernels.
- **Extreme score magnitudes**: insufficient scaling or missing mask can cause `NaN` after softmax; using large negative values (not `-inf` literal in some backends) for masks is common to avoid numerical issues.
- **Broadcasting bugs**: masks with incorrect shape (e.g., `[B, L]` where `[B, 1, 1, L]` is expected) can silently produce wrong patterns of attention; always check final `scores.shape` and `mask.shape`.

## Multi-Head Self-Attention: Why One Head Isn’t Enough

> **[IMAGE GENERATION FAILED]** Multi-head self-attention: multiple parallel attention heads operate on the same input X in different subspaces, and their outputs are concatenated and linearly projected back to the model dimension.
>
> **Alt:** Diagram of multi-head self-attention showing parallel heads and concatenation back to model dimension
>
> **Prompt:** Architectural diagram of multi-head self-attention. On the left, an input tensor X (B×L×d_model). It fans out into h parallel branches labeled Head 1, Head 2, ..., Head h. Each branch contains a small box labeled Q,K,V + scaled dot-product attention, with input shape B×L×d_head and output B×L×d_head. All head outputs then converge into a concat block forming B×L×(h·d_head), followed by a linear projection block W_O back to B×L×d_model. Indicate dimensions near tensors and use clear labels. Clean vector style on white background.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image\nPlease retry in 48.916550255s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '48s'}]}}


A single self-attention head takes an input sequence of hidden states and produces a new sequence where each position is a weighted mix of all positions.

Let the input be:

- Shape: `(batch_size, seq_len, d_model)`

A single head uses three learned linear maps:

- `W_Q ∈ ℝ^{d_model × d_k}`
- `W_K ∈ ℝ^{d_model × d_k}`
- `W_V ∈ ℝ^{d_model × d_v}`

Compute:

- `Q = X W_Q` → `(batch, seq_len, d_k)`
- `K = X W_K` → `(batch, seq_len, d_k)`
- `V = X W_V` → `(batch, seq_len, d_v)`

Then scaled dot-product attention:

- Scores: `S = Q Kᵀ / sqrt(d_k)` → `(batch, seq_len, seq_len)`
- Weights: `A = softmax(S, dim=-1)`
- Output: `O = A V` → `(batch, seq_len, d_v)`

Multi-head self-attention just replicates this structure `h` times in parallel, each with its own parameters:

- For head `i`:
  - `W_Q^i, W_K^i, W_V^i` all independent
- Each head sees the same `X` but projects it into a different learned subspace.

Intuitively, each head can specialize:

- One head might focus on local patterns (nearby tokens).
- Another on long-range dependencies.
- Others on syntactic roles (subject–verb links) vs. semantic similarity (coreference, topic).

Because each head has its own projections, these “views” are linearly separated and can evolve independently during training, instead of forcing a single attention map to encode all relationships at once.

After computing per-head outputs:

- `O_i` for head `i` has shape `(batch, seq_len, d_v)`
- Concatenate along the feature dimension:

  - `O_concat = concat(O_1, …, O_h)` → `(batch, seq_len, h * d_v)`

Then project back to the model dimension:

- `W_O ∈ ℝ^{(h * d_v) × d_model}`
- Final output: `Y = O_concat W_O` → `(batch, seq_len, d_model)`

This preserves the original dimensionality while letting the model mix information across heads.

**Trade-offs:**

- Parameters:
  - Each head adds its own `W_Q^i, W_K^i, W_V^i`.
  - Total parameters grow roughly linearly with `h` (for fixed `d_k`, `d_v`), or stay similar if `d_k`, `d_v` shrink as `h` grows (common in practice).
- Compute:
  - Attention scores scale as `O(h * seq_len² * d_k)`.
  - More heads → more FLOPs and memory for attention matrices.
- Benefits:
  - Higher modeling capacity: multiple, specialized attention patterns.
  - Optimization: easier for SGD/Adam to “assign” roles to heads than to force one large attention map to do everything, which can lead to entangled gradients and slower convergence.

Typical configurations pick:

- `d_model` divisible by the number of heads, `h` (e.g., 4, 8, 12, 16,…).
- Per-head dimension `d_k = d_v = d_model / h`.
- This keeps `h * d_v = d_model`, so parameter count and compute remain balanced while enabling parallelism.
- Choosing `h` and `d_k` as multiples of 8/16 helps align with vector widths and tensor-core layouts, improving memory access patterns and throughput.

Conceptually:

- Too few heads: bottlenecked expressiveness; one head must mix many distinct relationships.
- Too many heads (with very small `d_k`): each head is too “thin” to capture rich structure; overhead from per-head operations increases.

A minimal code sketch:

```python
import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape  # batch, seq_len, d_model

        # project to Q, K, V of shape (B, num_heads, T, d_head)
        q = self.W_q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # scaled dot-product attention per head
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, h, T, T)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # (B, h, T, d_head)

        # combine heads: (B, T, h * d_head) == (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)
```

This layout (batch × heads × seq × head_dim) is designed to match hardware-friendly tensor operations while exposing the benefits of multi-head specialization.

## A Minimal Self-Attention Implementation: From Equations to Code

Below is a compact, framework-style sketch in PyTorch-like pseudocode. The goal is to mirror the equations while making shapes and data flow very explicit.

### Scaled Dot-Product Attention

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None

    def forward(self, Q, K, V, mask=None):
        """
        Q: (batch, seq_q, d_k)
        K: (batch, seq_k, d_k)
        V: (batch, seq_k, d_v)
        mask: (batch, 1, seq_q, seq_k) or (batch, seq_q, seq_k), with 0 for masked positions
        """
        d_k = Q.size(-1)

        # (batch, seq_q, d_k) @ (batch, d_k, seq_k) -> (batch, seq_q, seq_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Ensure broadcastable; mask == 0 gets large negative
            while mask.dim() < scores.dim():
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # along seq_k dimension

        if self.dropout is not None:
            attn = self.dropout(attn)

        # (batch, seq_q, seq_k) @ (batch, seq_k, d_v) -> (batch, seq_q, d_v)
        output = torch.matmul(attn, V)
        return output, attn
```

### Multi-Head Self-Attention

We now wrap this into a multi-head module. Shapes are annotated to make the head reshaping explicit.

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear projections for Q, K, V and final output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(dropout_p)
        self.dropout = nn.Dropout(dropout_p)

    def _split_heads(self, x):
        """
        x: (batch, seq, d_model)
        returns: (batch, num_heads, seq, d_head)
        """
        b, seq, _ = x.size()
        x = x.view(b, seq, self.num_heads, self.d_head)
        x = x.permute(0, 2, 1, 3)
        return x

    def _merge_heads(self, x):
        """
        x: (batch, num_heads, seq, d_head)
        returns: (batch, seq, d_model)
        """
        b, h, seq, d_h = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(b, seq, h * d_h)

    def forward(self, x, mask=None):
        """
        x: (batch, seq, d_model)
        mask: (batch, seq, seq) or (batch, 1, seq, seq)
        """
        # Project to Q, K, V
        Q = self._split_heads(self.W_q(x))  # (b, h, seq, d_head)
        K = self._split_heads(self.W_k(x))  # (b, h, seq, d_head)
        V = self._split_heads(self.W_v(x))  # (b, h, seq, d_head)

        if mask is not None:
            # Expand mask to (batch, 1, seq, seq) if needed
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

        # Merge batch and head dims for dot-product attention
        b, h, seq, d_h = Q.size()
        Q_ = Q.reshape(b * h, seq, d_h)
        K_ = K.reshape(b * h, seq, d_h)
        V_ = V.reshape(b * h, seq, d_h)
        mask_ = None
        if mask is not None:
            mask_ = mask.expand(b, h, seq, seq).reshape(b * h, seq, seq)

        context, attn = self.attn(Q_, K_, V_, mask_)  # context: (b*h, seq, d_head)

        # Restore heads dimension
        context = context.view(b, h, seq, d_h)
        context = self._merge_heads(context)  # (b, seq, d_model)

        out = self.W_o(context)
        out = self.dropout(out)
        return out, attn
```

### Verifying Shapes with a Tiny Example

```python
def demo():
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 3
    d_model = 8
    num_heads = 2

    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len, seq_len)  # no masking

    mha = MultiHeadSelfAttention(d_model, num_heads, dropout_p=0.0)

    print("Input x:", x.shape)
    out, attn = mha(x, mask)

    print("Output:", out.shape)          # (2, 3, 8)
    print("Attention weights:", attn.shape)  # (b*h, seq, seq) = (4, 3, 3)

demo()
```

When debugging, you can temporarily add prints inside `_split_heads` and `_merge_heads` to confirm shapes like `(batch, num_heads, seq, d_head)` and `(batch, seq, d_model)`.

### Common Pitfalls and Edge Cases

- **Missing scaling factor**  
  Forgetting `1 / sqrt(d_k)` in the score computation makes gradients unstable and pushes softmax toward saturation.

- **Incorrect broadcasting in scores**  
  Using `Q @ K` without transposing `K` yields shape errors or silently wrong shapes if dimensions happen to align. Always check `scores.shape == (batch, seq_q, seq_k)`.

- **Mask misalignment**  
  - Wrong mask shape can broadcast along the batch or head dimensions incorrectly, masking the wrong tokens.
  - Using `1` for masked and `0` for unmasked while doing `masked_fill(mask == 0, -inf)` reverses the effect.

- **Very long sequences**  
  Memory and compute scale as `O(seq_len^2)`. For very long `seq_len`, this becomes a bottleneck; profiling attention separately is useful.

- **Non-divisible head dimensions**  
  If `d_model % num_heads != 0`, the `view` in `_split_heads` fails. Guard with an assertion.

### Where Dropout and LayerNorm Plug In

A typical Transformer-style block around self-attention (simplified):

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout_p=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout_p)
        self.ln1 = nn.LayerNorm(d_model)
        # feed-forward block omitted; would be here
        # self.ffn = ...
        # self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention + residual + layer norm
        attn_out, _ = self.attn(x, mask)     # (batch, seq, d_model)
        x = self.ln1(x + attn_out)           # residual connection

        # FFN + residual + ln2 would follow here
        return x
```

- **Dropout** commonly appears:
  - On attention weights (inside `ScaledDotProductAttention`).
  - After `W_o` before adding the residual.
- **LayerNorm** wraps the residual path, stabilizing training by normalizing across features per token.

## Handling Position and Structure: Why Self-Attention Needs Extra Signals

Consider a vanilla self-attention layer that only sees a bag of token embeddings. If you permute the input sequence (shuffle the tokens) but keep the same set of vectors, the attention output is just permuted in the same way. There is no notion of “first,” “next,” or “previous” inside the mechanism itself—only similarity between vectors. For many sequence tasks this is fatal:  

- In language, “dog bites man” vs. “man bites dog” should not collapse to the same internal representation.  
- In time series, swapping day 1 and day 10 destroys trends and causality.  

Pure self-attention without positional information is permutation-invariant, but most sequence modeling problems are order-sensitive.

The standard fix is simple but powerful: inject position information into the token representations before attention. Each token embedding `x_t` is combined with a positional signal `p_t`:

```python
x_pos = x_embed + pos_embed  # or concat, depending on design
attn_out = self_attention(x_pos)
```

The attention block itself is unchanged; queries, keys, and values are computed as usual. But because `x_pos` now depends on both token identity and position, permuting the sequence changes the pairwise similarities and therefore the attention pattern. This breaks permutation invariance while retaining the same self-attention machinery.

There are two main ways to encode position:

- **Absolute positions**: each index `t` (0, 1, 2, …) gets its own embedding or deterministic vector. The model can learn that “position 0” behaves like a start-of-sequence, or that early positions differ from later ones.
- **Relative positions**: the model cares about “how far apart” tokens are (offsets like -1, +2, +10) rather than their absolute indices. This naturally supports patterns like “attend to the previous token” or “attend within a small window,” and can generalize better across sequence lengths.

Self-attention can integrate relative information by modifying how attention scores are computed, e.g., adding learned biases or embeddings that depend on the distance between query and key positions. This lets a head specialize in relations like “previous word,” “same segment,” or “nearby window,” not just global similarity.

Positional choices strongly shape what dependencies heads tend to learn:

- Position-free (or very weak) signals encourage global, order-agnostic patterns—good for sets, bad for syntax or temporal causality.
- Strong local relative biases push heads to focus on nearby tokens, which often improves efficiency and inductive bias for language or time series.
- More flexible or long-range encodings make it easier for heads to track dependencies over hundreds or thousands of steps, at the cost of computation and potential overfitting to spurious long-range correlations.

Beyond 1D sequences, you can impose **structural biases** by designing positional schemes or masks that reflect the data’s shape:

- Grids (images): 2D positions, or relative row/column offsets, so attention respects spatial locality.
- Trees (parses, ASTs): tree-based distances or ancestor relationships as positional/relative features.
- Graphs: adjacency-based masks and edge-aware positional terms so tokens mainly attend along graph edges.

In all these cases, self-attention remains the same algebraic operation. What changes is the positional and structural context you feed into it, which determines whether the model understands “where” and “how” tokens relate, not just “what” they are.

## Performance and Scaling: Cost, Memory, and Optimization Concerns

Self-attention is powerful but expensive. Understanding its scaling behavior is essential when you push to longer sequences or larger models.

### Time and memory complexity

Consider a single self-attention layer with:

- Batch size: `B`
- Sequence length: `L`
- Model dimension: `d_model`
- Number of heads: `H`
- Per-head dimension: `d_k = d_model / H`

Key costs:

1. **Projection to Q, K, V**

   You multiply the input `X ∈ ℝ^{B×L×d_model}` by three linear layers:

   - Time: `O(B * L * d_model^2)` if implemented as three separate `d_model×d_model` projections.
   - Memory (activations): `O(B * L * d_model)` for `Q`, `K`, `V`.

2. **Attention scores and softmax**

   Per head, scores are `S = QKᵀ`, shape `B × H × L × L`.

   - Time: `O(B * H * L^2 * d_k)` (multiplying `L×d_k` by `d_k×L`).
   - Memory: `O(B * H * L^2)` for scores (plus some for softmax).

3. **Weighted sum**

   `A = softmax(S) V` also costs:

   - Time: `O(B * H * L^2 * d_k)`.
   - Memory: `O(B * L * d_model)` for the output.

Overall, ignoring constant factors, **self-attention time is dominated by `O(B * L^2 * d_model)`**, and the **quadratic `L^2` term** is the main scaling problem. Activation memory is dominated by the attention scores at `O(B * H * L^2)`.

### Attention scores dominate memory

The `L×L` score matrix per head is the key memory hog:

- Memory roughly scales as `B * H * L^2 * bytes_per_element`.
- Doubling `L` (sequence length) roughly **quadruples** score-memory usage.
- For long sequences, this can exceed device memory even when the parameter count fits comfortably.

This is why “maximum context length” is often a memory, not just a modeling, constraint. Mixed precision and activation checkpointing can help, but the `L^2` scaling remains.

### Practical knobs to stay within budgets

To fit your hardware and latency targets, you can:

- **Reduce sequence length `L`**
  - Truncate or window inputs.
  - Pros: Quadratically reduces attention cost and memory.
  - Cons: May lose long-range dependencies; needs careful task-specific handling.

- **Reduce model width `d_model` or `H`**
  - Pros: Linear reduction in compute and some memory; often easiest way to fit a GPU.
  - Cons: Less capacity per token; can degrade quality.

- **Reduce batch size `B`**
  - Pros: Linear memory reduction; often the first lever for OOM.
  - Cons: Noisy gradients, slower wall-clock throughput unless you adjust learning rate or accumulate gradients.

- **Fewer layers**
  - Pros: Linear reduction in compute and activations.
  - Cons: Lower representational depth; often hurts performance more sharply than modest width/batch changes.

Typical trade-off: For a fixed device, you choose among longer context, larger model, or larger batch—but not all three.

### Ideas behind more efficient attention variants

Many “efficient attention” proposals tackle the `L^2` issue:

- **Sparse patterns**
  - Limit each position to attend only to a subset (e.g., local windows, strided or block patterns).
  - Aim: Reduce effective complexity to near-linear or `O(L log L)` while preserving enough global context via carefully designed sparsity.

- **Low-rank / kernel-based approximations**
  - Approximate the `L×L` attention matrix using lower-rank structure or kernel tricks so you can rewrite attention as products that scale as `O(L * d_model)` or `O(L * r)` for small rank `r`.
  - Core idea: You never form the full `L×L` matrix explicitly.

- **Chunking / block processing**
  - Process sequences in chunks, computing attention only within or between limited chunks.
  - Often combined with caching or recurrent-style mechanisms to approximate full-context attention over time.

All of these trade exactness for better scaling. The details differ, but the shared theme is: avoid storing or computing the full dense `L×L` scores.

### Implementation pitfalls and profiling

Even with the same theoretical complexity, implementations can vary significantly in speed:

- **Excessive tensor reshaping or transposing**
  - Repeated, unnecessary `view`, `permute`, or `contiguous` calls can cause data copies and break fusion opportunities in kernels.

- **Non-contiguous memory layouts**
  - Operations on non-contiguous tensors can trigger implicit copies or slow memory access.

- **Small, fragmented ops**
  - Implementing attention as many tiny operations (e.g., manual loops over heads or sequence positions) prevents kernel fusion and underutilizes hardware.

A minimal, reasonably efficient sketch in a typical tensor framework looks like:

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: (B, H, L, d_k)
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / d_k**0.5  # (B, H, L, L)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn = scores.softmax(dim=-1)  # (B, H, L, L)
    out = attn @ v                # (B, H, L, d_k)
    return out
```

For real systems:

- Use fused, library-provided attention kernels when possible.
- Profile end-to-end with built-in profilers to identify:
  - Time spent in attention vs. feed-forward layers.
  - Memory-bound vs. compute-bound regions.
  - Hotspots caused by reshaping, casting, or Python-side loops.

Grounding your design in these complexity and implementation facts helps you choose architectures and configurations that scale without surprise OOMs or latency spikes.

## Edge Cases, Failure Modes, and Debugging Self-Attention

Self-attention layers are deceptively simple mathematically, but they fail in characteristic ways. Recognizing the patterns early saves a lot of training time and head‑scratching.

### Numerical stability: exploding scores and near one-hot softmax

The raw attention scores are dot products between queries and keys. Without care, these scores can grow large in magnitude, causing the softmax to:

- Saturate (e.g., one value ≈ 1, all others ≈ 0)
- Produce `NaN` due to overflow in `exp`

Mitigations that should be standard:

- **Scaling**: divide scores by `sqrt(d_k)` before softmax. This keeps logits in a reasonable range as dimensionality grows.
- **Normalization**:
  - Layer normalization before/after attention blocks stabilizes activations.
  - Gradient clipping prevents rare but catastrophic updates.
- **Initialization**:
  - Use variance-preserving initializers so `Q`, `K`, `V` magnitudes don’t explode.
  - Avoid large bias terms in projection layers that could shift scores.

A basic numerical check: log the mean and max of attention logits before softmax. If max values are consistently large (e.g., > 30) or grow over training, you’re heading toward saturation.

### Attention collapse: always focusing on a few tokens

A common failure is **attention collapse**, where heads put almost all mass on:

- A single position (e.g., the first token)
- A special token (e.g., CLS, BOS)
- The token itself (identity mapping)

Symptoms:

- Attention heatmaps show vertical stripes (every token attends to the same index).
- Head entropy (per row of the attention matrix) is very low across the board.

Detection strategies:

- Visualize attention for a few representative sequences.
- Compute attention entropy:
  - High entropy → diffuse attention
  - Very low entropy across many heads/layers → collapse

Countermeasures:

- Add or tune regularization (dropout in attention, weight decay).
- Encourage diversity across heads (e.g., loss terms or architectural changes, if you control the model).
- Inspect data preprocessing; sometimes a special token dominates because it encodes too much information.

### Padding and masking bugs

Masks ensure the model ignores padded positions and (for causal models) the future. When masks are wrong, self-attention misbehaves in subtle ways.

Typical manifestations:

- The model **attends heavily to pad tokens**:
  - Outputs depend on how many pads are present.
  - Performance degrades with variable-length batches but looks fine for fixed-length data.
- Sequence-to-sequence models “leak” by attending to future or padded positions in the encoder/decoder.

Debugging steps:

- Manually construct **tiny synthetic inputs**:
  - Batch of size 1–2, sequence length 3–5.
  - Clear, distinct token embeddings (e.g., one-hot or small integers).
  - Explicit masks (e.g., last two positions padded).
- Print:
  - The mask tensor (confirm `True`/`1` means “keep” or “mask out” as expected).
  - The attention matrix for each head.
- Ensure masked positions:
  - Receive near-zero attention probability.
  - Do not affect the attended summary (test by toggling pad tokens and checking output invariance).

Shape mismatches also cause silent errors: a mask broadcast in the wrong dimension can effectively mask the wrong tokens or even whole heads.

### Overfitting patterns specific to attention

Attention heads can overfit in ways that look “working” but don’t generalize:

- **Identity heads**: each position attends almost exclusively to itself.
- **Position-only heads**: attention patterns depend only on positions (e.g., always attend to previous token) and ignore content.
- **Shortcut heads**: a head always attends to a special token that carries global info, ignoring local structure.

Hints you’re seeing this:

- Validation loss diverges from training loss, while attention patterns remain rigid.
- Attention maps look nearly identical across very different inputs.

Mitigations:

- Regularization:
  - Attention dropout.
  - Weight decay on projection matrices.
- Architectural hygiene:
  - Limit the number of heads to what’s justified by data.
  - Periodically evaluate **head importance** (e.g., by zeroing heads and measuring impact) and prune unhelpful ones.

### Practical observability and tests

Treat self-attention as a first-class observable component:

- **Log attention entropy**:
  - Per head, per layer.
  - Watch for:
    - Entropy collapsing to near zero (over-confident, one-hot behavior).
    - Entropy blowing up if logits are noisy/undertrained.
- **Visualize heads**:
  - For a fixed set of diagnostic prompts/inputs, render attention matrices:
    - Rows: query positions.
    - Columns: key positions.
  - Look for systematic patterns (diagonals, local bands, global tokens) and anomalies (attention to pads, random noise).
- **Unit tests**:
  - **Shape tests**: ensure `Q`, `K`, `V`, masks, and outputs have expected shapes and broadcasting semantics.
  - **Mask behavior tests**:
    - Changing padded tokens should not change outputs.
    - For causal masks, attending to future positions should be impossible.

A small test harness can already catch a lot:

```python
import torch
import torch.nn.functional as F

def run_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return attn @ v, attn

# Example: verify padded token does not influence output
torch.manual_seed(0)
seq_len, d = 4, 8
x = torch.randn(1, seq_len, d)
pad_mask = torch.tensor([[1, 1, 1, 0]])  # last token is pad

Wq = torch.randn(d, d)
Wk = torch.randn(d, d)
Wv = torch.randn(d, d)

def project(x):
    return x @ Wq, x @ Wk, x @ Wv

q, k, v = project(x)
mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, T]

y1, attn1 = run_attention(q, k, v, mask)

# Change padded token embedding
x2 = x.clone()
x2[0, -1] += 10.0  # large change in pad
q2, k2, v2 = project(x2)
y2, attn2 = run_attention(q2, k2, v2, mask)

print("Max diff in non-pad outputs:", (y1[0, :-1] - y2[0, :-1]).abs().max().item())
```

This kind of minimal test can quickly reveal mask mistakes, attention collapse, and numerical issues long before you scale to full training.

## Putting It All Together: Self-Attention in a Transformer Block

A standard Transformer block wraps self-attention inside a few other key components:

- **Multi-head self-attention sublayer**
- **Residual (skip) connections**
- **Layer normalization**
- **Position-wise feed-forward network (FFN)** applied independently to each token
- Often **dropout** around/inside these sublayers

A common ordering (pre-norm variant) looks like:

1. `x` (sequence of token embeddings with positional encodings added)
2. `x₁ = x + SelfAttention(LayerNorm(x))`
3. `x₂ = x₁ + FeedForward(LayerNorm(x₁))`

So the dataflow through a single block is:

1. **Input embeddings**: You start with a matrix `X ∈ ℝ^{T×d_model}` where each row is a token embedding plus positional information.
2. **Self-attention**:
   - Compute queries, keys, values from `X`.
   - Apply attention (with any required masks).
   - Concatenate heads and project back to `d_model`, giving `SA(X)`.
3. **Residual + norm**:
   - Add `X` back: `Y = X + SA(X)` to stabilize gradients and preserve the original signal.
   - Normalize `Y` across the feature dimension for each token.
4. **Feed-forward network**:
   - Apply a 2-layer MLP to each token independently: `FFN(norm(Y))`.
   - Add another residual and normalization: `Z = Y + FFN(norm(Y))`.

Stacking many such blocks lets the model progressively build more abstract interactions:

- Lower layers capture local and syntactic relationships.
- Higher layers aggregate longer-range and task-specific patterns.
- Because each layer attends over the entire (visible) sequence, depth compounds these relationships instead of just widening a fixed receptive field.

Self-attention’s role changes with architecture type:

- **Encoder-only** (e.g., for classification):
  - Self-attention is **bidirectional**: each token can attend to all others.
  - No causal mask; masks are usually just for padding.
- **Decoder-only** (e.g., autoregressive generation):
  - Self-attention is **causal**: tokens can only attend to past and current positions.
  - A triangular mask enforces left-to-right generation.
- **Encoder–decoder**:
  - **Encoder blocks**: bidirectional self-attention (like encoder-only).
  - **Decoder blocks**: 
    - Causal self-attention over decoded tokens.
    - Plus a **cross-attention** sublayer where decoder queries attend to encoder outputs (keys/values).

To summarize, you should now be able to:

- Write down the components of a Transformer block and their order.
- Trace how embeddings and positional information flow through self-attention, residuals, norm, and FFN.
- Reason about what stacking more blocks buys you in terms of interaction depth.
- Distinguish encoder, decoder, and encoder–decoder attention patterns (especially masking).
- Use this mental model to interpret architecture diagrams, decode notation in papers, and map configuration options to actual behavior when implementing or modifying Transformer models.
