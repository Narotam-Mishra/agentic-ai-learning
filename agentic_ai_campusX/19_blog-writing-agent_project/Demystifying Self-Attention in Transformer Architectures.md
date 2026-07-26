# Demystifying Self-Attention in Transformer Architectures

## From Recurrent Models to Self-Attention: Why Transformers Needed a New Core

> **[IMAGE GENERATION FAILED]** RNNs pass information through a chain, CNNs use local fixed windows, while self-attention allows every token to connect directly to every other token in a single layer.
>
> **Alt:** Diagram comparing RNN, CNN, and self-attention connectivity for a short sentence
>
> **Prompt:** clean technical diagram with three panels labeled RNN, CNN, Self-Attention, each panel showing the same short sequence of 6 tokens. RNN panel: arrows only from token t-1 to t along the sequence. CNN panel: each token connected to a small local neighborhood (e.g., window size 3) with fixed pattern. Self-Attention panel: dense all-to-all connections between tokens (faint gray lines), highlighting that every token can directly attend to every other token. Minimal, monochrome or two-color scheme, suitable for a machine learning blog, no decorative elements.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your prepayment credits are depleted. Please go to AI Studio at https://ai.studio/projects to manage your project and billing. Learn more at https://ai.google.dev/gemini-api/docs/billing#prepay. ', 'status': 'RESOURCE_EXHAUSTED'}}


RNNs and CNNs were the dominant tools for sequence modeling before Transformers.

- **RNNs** process inputs step by step. At time step *t*, the hidden state depends on step *t–1*. This gives them an implicit notion of order, but:
  - Long-range dependencies must be carried through many recurrent updates, leading to vanishing/exploding gradients and information loss.
  - Computation is inherently **sequential**: you can’t process token *t* before token *t–1*, which limits parallelism.

- **CNNs for sequences** (1D convolutions) process multiple positions in parallel, but each layer has a **fixed receptive field** (e.g., kernel size 3). To connect distant tokens, you stack many layers or use dilations:
  - Long-range interactions require depth and careful design of dilation patterns.
  - The connectivity is mostly **static** and position-local, not directly conditioned on the specific content at each position.

### A concrete example: “long-distance” interaction

Consider the sentence:

> “If the deployment fails, **roll back**.”

The word “If” at the beginning and the phrase “roll back” at the end are tightly coupled: “roll back” is conditional on “If the deployment fails”.

- In an **RNN**, information must flow:
  - “If” → “the” → “deployment” → “fails,” → “roll” → “back”.
  - To decide how to interpret “roll back”, the model relies on the hidden state that has been passed through all intermediate tokens. That’s multiple recurrent transitions where signal can degrade.

- In a **self-attention layer**, when computing the representation for “roll” or “back”, the model can directly “look at” all tokens, including “If”, in a *single* step. There is no need for a multi-hop chain just to connect distant positions.

This is a core motivation: **any token can directly consult any other token in O(1) layers**, regardless of distance in the sequence.

### What “attention” means at a high level

Self-attention is a learned, content-dependent **weighted aggregation of all tokens**, conditioned on a particular “query” token.

Conceptually, for each token \(i\):

1. Compute a **query** vector \(q_i\).
2. For every token \(j\), compute a **key** \(k_j\) and **value** \(v_j\).
3. Compute similarity scores \(s_{ij} = q_i \cdot k_j\), normalize them (softmax), and use them as weights.
4. The new representation of token \(i\) is a **weighted sum** of all \(v_j\), where weights encode “how relevant is token \(j\) to token \(i\)?”.

All tokens do this in parallel, so in one layer each token can pull information from everywhere in the sequence.

### Encoder-only, decoder-only, and encoder–decoder: self-attention’s roles

> **[IMAGE GENERATION FAILED]** Self-attention appears in encoder-only stacks (bidirectional), decoder-only stacks (causal), and encoder–decoder models where decoder self-attention is combined with cross-attention to encoder outputs.
>
> **Alt:** High-level diagram of encoder-only, decoder-only, and encoder–decoder Transformer stacks with self-attention and cross-attention
>
> **Prompt:** technical block diagram comparing three Transformer variants side by side: Encoder-only, Decoder-only, Encoder–Decoder. Each variant drawn as a vertical stack of layers. For Encoder-only: single stack labeled Encoder with bidirectional self-attention blocks over input tokens. For Decoder-only: single stack labeled Decoder with causal self-attention blocks over generated tokens. For Encoder–Decoder: left stack labeled Encoder with self-attention over source tokens, right stack labeled Decoder with self-attention over target tokens plus cross-attention blocks that take encoder outputs as keys and values. Use arrows to indicate flow of queries, keys, and values conceptually but keep text minimal, clean style.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your prepayment credits are depleted. Please go to AI Studio at https://ai.studio/projects to manage your project and billing. Learn more at https://ai.google.dev/gemini-api/docs/billing#prepay. ', 'status': 'RESOURCE_EXHAUSTED'}}


Self-attention is the same core mechanism, deployed in different ways:

- **Encoder-only** (e.g., for classification, embeddings):
  - You have an input sequence (text, code, etc.).
  - Each layer uses **bidirectional self-attention** over the entire input.
  - Goal: produce rich contextual representations of each token (or a pooled representation of the whole sequence).

- **Decoder-only** (e.g., autoregressive language models):
  - The model generates tokens left-to-right.
  - Each layer uses **causal self-attention**: token *t* can attend only to tokens ≤ *t*.
  - Goal: predict the next token given all prior ones, with direct access to *all* previous positions.

- **Encoder–decoder** (e.g., classic sequence-to-sequence):
  - Encoder: same as encoder-only, builds source-sequence representations using self-attention.
  - Decoder: uses **self-attention over generated tokens so far**, plus **cross-attention** to the encoder’s outputs.
  - Goal: map one sequence to another, letting each generated token attend both to prior outputs and all input tokens.

### Self-attention as learned connectivity

RNNs impose a **fixed chain** of information flow, and CNNs impose a **fixed local neighborhood**. Self-attention instead lets the model **learn a data-dependent connectivity pattern**:

- For each token, the model decides which other tokens matter *for this example*.
- This pattern is recomputed at every layer, enabling complex relational structures to form in a few layers.
- Long-range and local dependencies are treated uniformly: “distance” in the input sequence doesn’t restrict who can talk to whom.

This shift—from rigid, position-driven connections to **flexible, content-driven connections**—is why self-attention became the new core of Transformer architectures.

## Mechanics of Self-Attention: Queries, Keys, Values, and Weights

> **[IMAGE GENERATION FAILED]** Scaled dot-product self-attention for a single head: input X is projected to Q, K, V, scores QKᵀ are scaled, masked, passed through softmax to get attention A, and finally multiplied by V to produce output O.
>
> **Alt:** Flow diagram of scaled dot-product self-attention from X to Q, K, V to attention weights and output O
>
> **Prompt:** step-by-step flow diagram for single-head scaled dot-product self-attention. Start with matrix X on the left, arrows into three linear projection blocks labeled W_Q, W_K, W_V producing Q, K, V. From Q and K arrows into a node labeled Q K^T / sqrt(d_k) producing a scores matrix. Next box for optional Masking, then a Softmax box producing attention matrix A. Finally, arrow from A and V into a node labeled A V giving output O. Show matrix shapes lightly annotated, e.g., (T, d_model), (T, d_k), (T, T), (T, d_v). Clean, minimalist, vector-style graphic on light background, no decorative icons.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your prepayment credits are depleted. Please go to AI Studio at https://ai.studio/projects to manage your project and billing. Learn more at https://ai.google.dev/gemini-api/docs/billing#prepay. ', 'status': 'RESOURCE_EXHAUSTED'}}


At the level of a single attention head, self-attention is just a series of batched matrix multiplications and elementwise operations.

### Input representation: the X matrix

Assume we have:

- batch size: `B`
- sequence length: `T`
- embedding dimension: `d_model`

We collect token embeddings into a tensor:

- `X` with shape `(B, T, d_model)`

Typically, `X` is the sum of token embeddings and positional encodings, and it flows into a single self-attention layer as the input that will be mixed across positions.

For many implementations, computations are done per batch, but conceptually you can think in terms of one sequence:

- `X` (for one sequence) is a matrix of shape `(T, d_model)`
- the `t`‑th row `X[t]` is the embedding of token `t`.

### Q, K, V projections

A single attention head uses three learned linear projections:

- `W_Q` of shape `(d_model, d_k)`
- `W_K` of shape `(d_model, d_k)`
- `W_V` of shape `(d_model, d_v)`

Typical choice: `d_k = d_v = d_model / num_heads`, but that’s a design choice.

We compute:

- `Q = X · W_Q` → shape `(T, d_k)`
- `K = X · W_K` → shape `(T, d_k)`
- `V = X · W_V` → shape `(T, d_v)`

Batched, with shapes:

- `X`: `(B, T, d_model)`
- `W_Q`: `(d_model, d_k)` ⇒ `Q`: `(B, T, d_k)`
- `W_K`: `(d_model, d_k)` ⇒ `K`: `(B, T, d_k)`
- `W_V`: `(d_model, d_v)` ⇒ `V`: `(B, T, d_v)`

Each token’s embedding is mapped to a query vector, a key vector, and a value vector.

### Computing attention scores and weights

For one sequence (drop batch index), self-attention between all token positions is:

1. **Raw scores**  
   Compute similarity between each query and each key:

   \[
   S = Q K^\top
   \]

   - `Q`: `(T, d_k)`
   - `K^T`: `(d_k, T)`
   - `S`: `(T, T)` where `S[i, j]` is the score of token `i` attending to token `j`.

2. **Scaling**  
   To stabilize gradients when `d_k` is large:

   \[
   \tilde{S} = \frac{S}{\sqrt{d_k}}
   \]

3. **Masking (optional but common)**  
   - **Causal mask** (decoder): disallow attending to “future” tokens (`j > i`) by setting those scores to a large negative number (e.g., `-1e9`).
   - **Padding mask**: disallow attending to padded positions.

   Conceptually:

   \[
   \tilde{S}_{\text{masked}}[i, j] =
   \begin{cases}
   \tilde{S}[i, j] & \text{if allowed}\\
   -\infty & \text{if masked}
   \end{cases}
   \]

4. **Softmax → attention weights**  
   Apply softmax row-wise over the last dimension (across `j` for each `i`):

   \[
   A[i, j] = \frac{\exp(\tilde{S}_{\text{masked}}[i, j])}{\sum_{k=1}^{T} \exp(\tilde{S}_{\text{masked}}[i, k])}
   \]

   - `A`: `(T, T)`  
   Each row `A[i]` is a probability distribution over all positions `j` that token `i` can attend to.

### Output: mixing values with attention weights

The head’s output is a weighted sum of the values for each token:

\[
O = A V
\]

- `A`: `(T, T)`
- `V`: `(T, d_v)`
- `O`: `(T, d_v)`

Row-wise interpretation for token `i`:

\[
O[i] = \sum_{j=1}^{T} A[i, j] \cdot V[j]
\]

So the new representation of token `i` is a mixture of value vectors from all positions, weighted by how much `i` attends to them.

Batched shapes:

- `A`: `(B, T, T)`
- `V`: `(B, T, d_v)`
- `O`: `(B, T, d_v)`

This `O` is the per-head output, which in a full multi-head module would be concatenated across heads and linearly projected back to `d_model`.

### Minimal code sketch for a single head (no masks)

```python
import torch
import math

def single_head_self_attention(X, W_Q, W_K, W_V):
    # X: (B, T, d_model)
    # W_Q, W_K, W_V: (d_model, d_k/d_v)
    Q = X @ W_Q  # (B, T, d_k)
    K = X @ W_K  # (B, T, d_k)
    V = X @ W_V  # (B, T, d_v)

    # scores: (B, T, T)
    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(Q.size(-1))

    # attention weights: (B, T, T)
    A = torch.softmax(scores, dim=-1)

    # output: (B, T, d_v)
    O = A @ V
    return O
```

### Why it’s called “self”-attention

In **self-attention**, the same sequence `X` generates:

- queries (`Q`)
- keys (`K`)
- values (`V`)

So each token attends to **itself and other tokens in the same sequence**.

In **cross-attention**, queries and key/values come from *different* sources:

- `Q` from a **target** sequence (e.g., decoder hidden states)
- `K` and `V` from a **source** sequence or another modality (e.g., encoder outputs, image features)

The mechanics (QKᵀ, softmax, AV) are identical; only the origin of Q vs. K/V differs.

## Minimal Self-Attention Implementation: A Single-Head PyTorch Sketch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, d_head: int, causal: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.causal = causal

        # Linear projections for Q, K, V.
        # Input: (..., d_model) → Output: (..., d_head)
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)

        # Optional output projection back to d_model
        self.W_O = nn.Linear(d_head, d_model, bias=False)

    def forward(self, x, attn_mask: torch.Tensor | None = None):
        """
        x: (batch_size, seq_len, d_model)
        attn_mask: optional mask broadcastable to (batch_size, 1, seq_len, seq_len)
                   where 0 = keep, -inf (or very negative) = mask out.
        """
        B, T, _ = x.shape

        # Project to Q, K, V: (B, T, d_head)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Compute attention scores.
        # Step 1: add a "heads" dim (here =1) so shapes are:
        # Q: (B, 1, T, d_head), K: (B, 1, T, d_head)
        Qh = Q.unsqueeze(1)
        Kh = K.unsqueeze(1)

        # Step 2: scores = Q * K^T over d_head:
        # scores: (B, 1, T_query, T_key) = (B, 1, T, T)
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Optional causal mask: prevent attending to future positions.
        if self.causal:
            # causal_mask: (T, T) with 0 on allowed, -inf on disallowed.
            causal_mask = torch.full((T, T), float("-inf"), device=x.device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            # Broadcast to (1, 1, T, T) → (B, 1, T, T)
            scores = scores + causal_mask

        # Optional external attention mask (e.g., padding).
        if attn_mask is not None:
            # attn_mask should already be broadcastable to scores.shape
            # Typical shape: (B, 1, 1, T) or (B, 1, T, T)
            scores = scores + attn_mask

        # Softmax over key dimension (last dim) → weights sum to 1 across keys.
        # attn_weights: (B, 1, T_query, T_key)
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values.
        # Vh: (B, 1, T_value, d_head)
        Vh = V.unsqueeze(1)
        # out_head: (B, 1, T_query, d_head)
        out_head = torch.matmul(attn_weights, Vh)
        # Remove heads dim: (B, T, d_head)
        out = out_head.squeeze(1)

        # Final projection back to model dim: (B, T, d_model)
        out = self.W_O(out)

        return out, attn_weights  # return weights for inspection


if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, d_model, d_head = 2, 4, 8, 4
    x = torch.randn(B, T, d_model)

    # Example padding mask: suppose last token of each sequence is padding.
    # mask_base: (B, T) with 0 for real tokens, -inf for padding.
    mask_base = torch.zeros(B, T)
    mask_base[:, -1] = float("-inf")
    # Expand to (B, 1, 1, T) so each query token in a sequence
    # can't attend to padded key positions.
    attn_mask = mask_base.view(B, 1, 1, T)

    attn = SingleHeadSelfAttention(d_model=d_model, d_head=d_head, causal=False)

    out, weights = attn(x, attn_mask=attn_mask)

    print("x.shape:", x.shape)               # (2, 4, 8)
    print("out.shape:", out.shape)           # (2, 4, 8)
    print("weights.shape:", weights.shape)   # (2, 1, 4, 4)

    # Sanity check 1: attention weights sum to 1 over keys (last dim).
    # sums: (B, 1, T_query)
    sums = weights.sum(dim=-1)
    print("weights row sums:", sums)

    # Sanity check 2: compare attention on last (padded) position vs others.
    # For each batch, query token 0's weights to all keys:
    # weights[:, :, query_idx, :]
    print("query 0 weights (batch 0):", weights[0, 0, 0, :])
    print("query 0 weights (batch 1):", weights[1, 0, 0, :])

    # The last key position (index -1) should have near-zero probability
    # because we added -inf in the mask for that column.
    print("weights to padded key (index -1), batch 0:",
          weights[0, 0, :, -1])
    print("weights to padded key (index -1), batch 1:",
          weights[1, 0, :, -1])
```

## Multi-Head Self-Attention: Projecting, Splitting, and Recombining

Multi-head self-attention extends single-head attention by letting the model look at the sequence through multiple “views” in parallel. Each head can specialize in different relationships: some may focus on short-range patterns (e.g., local word order), others on long-range dependencies (e.g., subject–verb agreement), or different types of information (syntactic vs. semantic cues). Technically, this is done by having each head operate in its own learned subspace of the model’s feature space.

Assume an input tensor:

- `X` with shape `(batch_size, seq_len, d_model)`

First, we **linearly project** `X` into a higher-dimensional space that packs all heads at once. Instead of making separate projections per head in code, we usually share one set of big projection matrices:

- `W_Q ∈ ℝ^{d_model × (H·d_head)}`
- `W_K ∈ ℝ^{d_model × (H·d_head)}`
- `W_V ∈ ℝ^{d_model × (H·d_head)}`

We compute:

- `Q = X W_Q`
- `K = X W_K`
- `V = X W_V`

Each of `Q, K, V` has shape `(batch_size, seq_len, H·d_head)`. We then **reshape and split** into heads:

- Reshape to `(batch_size, seq_len, H, d_head)`
- Often transpose to `(batch_size, H, seq_len, d_head)` for convenience

By design, `d_model = H × d_head`, so concatenating all heads later gives us back `d_model` features per token.

For **parallel computation** of self-attention:

1. For each head `h`:
   - Take `Q_h, K_h, V_h` with shape `(batch_size, seq_len, d_head)`
   - Compute scaled dot-product attention:

     \[
     \text{Attention}(Q_h, K_h, V_h) = \text{softmax}\left(\frac{Q_h K_h^\top}{\sqrt{d_\text{head}}}\right)V_h
     \]

2. Stack all heads’ outputs:
   - Each head output: `(batch_size, seq_len, d_head)`
   - Concatenate along the last dimension to get `(batch_size, seq_len, H·d_head) = (batch_size, seq_len, d_model)`

A minimal sketch in PyTorch-like pseudocode:

```python
import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        def split_heads(t):
            t = t.view(B, T, self.h, self.d_head)
            return t.transpose(1, 2)  # (B, H, T, d_head)

        Q = split_heads(self.W_q(x))
        K = split_heads(self.W_k(x))
        V = split_heads(self.W_v(x))

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, H, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)

        heads = attn @ V  # (B, H, T, d_head)
        heads = heads.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.W_o(heads)
        return out
```

The **final output projection** `W_O ∈ ℝ^{d_model × d_model}` mixes information from all heads and returns to the model dimension. In a full Transformer block:

- You add a **residual connection**: `x + MultiHeadSelfAttention(x)`
- Then apply **layer normalization**, often as: `LayerNorm(x + MHA(x))`

This preserves gradient flow, stabilizes training, and lets subsequent layers re-weight or reinterpret what each head produced.

Choosing **number of heads `H` and head dimension `d_head`** is a trade-off:

- Larger `H`:
  - Pros: more subspaces, potentially richer patterns.
  - Cons: attention score tensor scales as `O(B · H · T²)` in memory and compute; too many heads can be redundant and slower.
- Larger `d_head` (with fixed `H`):
  - Pros: more capacity per head.
  - Cons: more parameters in `W_Q, W_K, W_V, W_O` and higher FLOPs per attention operation.

In practice, `d_model` is usually fixed by the model size, and you choose `H` such that `d_head = d_model / H` is not too small (to keep each head expressive) and `H` is not too large (to keep memory and compute manageable, especially for long sequences).

## Masks in Self-Attention: Padding, Causality, and Custom Patterns

Masking controls *who can see whom* inside self-attention.

- **Padding masks**: hide fake tokens added to make sequences the same length.
- **Causal masks**: hide *future* tokens to enforce autoregressive behavior.

Consider token indices `[0, 1, 2, 3]`.

- Padding example: real tokens at positions `[0, 1]`, padding at `[2, 3]`.  
  Padding mask (1 = keep, 0 = mask) for one sequence:
  - `[1, 1, 0, 0]`  
  Any attention to positions `2` or `3` should be blocked.
- Causal example (no future looking): token at `t` can only attend to `≤ t`.  
  Valid attention pairs:
  - `0 → 0`
  - `1 → {0, 1}`
  - `2 → {0, 1, 2}`  
  So row 2 cannot attend to column 3 (future).

### Constructing a causal mask matrix

For sequence length `L`, a standard causal mask is a lower-triangular matrix:

For `L = 4`, allowed positions (1 = allowed, 0 = masked):

```text
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

In practice we often store a *boolean or 0/−inf* mask and add it to the attention logits.

Minimal PyTorch-style sketch:

```python
import torch

batch_size = 2
num_heads = 4
seq_len = 4

# [L, L] boolean: True = mask (block), False = keep
causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

# Broadcast to [batch, heads, L, L]
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)          # [1, 1, L, L]
causal_mask = causal_mask.expand(batch_size, num_heads, -1, -1)

# Example attention logits: [batch, heads, L, L]
attn_logits = torch.randn(batch_size, num_heads, seq_len, seq_len)

# Large negative for masked positions
NEG_INF = -1e9
attn_logits = attn_logits.masked_fill(causal_mask, NEG_INF)

attn_weights = torch.softmax(attn_logits, dim=-1)            # along "key" axis
```

Broadcasting via `unsqueeze` and `expand` lets one base mask work across all batches and heads.

### Why add −∞ *before* softmax?

We mask by adding a large negative number (`-1e9` or similar) to disallowed logits *before* softmax. Intuition:

- `softmax([…, x, -1e9]) ≈ […, p, 0]`  
  The masked position’s probability is numerically ~0.
- The remaining positions are renormalized over only the allowed tokens.

This is preferred over:

- **Zeroing after softmax**:  
  - You break the probability simplex (rows no longer sum to 1).
  - Gradients back through the softmax don’t reflect the true constrained distribution.
- **Setting logits to zero**: still gives them non-negligible probability relative to real negatives.

Adding a huge negative ensures masked entries are effectively absent from both the forward pass and gradient flow.

### Common masking bugs

Frequent pitfalls:

- **Wrong shape**:  
  - Mask `[batch, L]` mistakenly used where attention is `[batch, heads, L, L]`.  
  - Broadcasting on the wrong dimensions silently mis-masks.
- **Wrong axis**:  
  - Softmax over the wrong dimension (e.g., over queries instead of keys).  
  - Mask compared against the wrong dimension order (`[L, L]` vs `[L]`).
- **Wrong dtype**:  
  - Using `float` mask with values `0/1` but `masked_fill` expects boolean.  
  - Or using integer mask directly in arithmetic without casting.
- **Mask applied after softmax**:  
  - `attn = softmax(logits); attn *= (1 - mask)`  
  - Probabilities no longer sum to 1; gradients are distorted.
- **Mixing padding and causal semantics**:  
  - Using a padding mask as if it were causal leaks future information in language models.
  - Using only a causal mask when sequences contain padding lets tokens attend to pad positions.

These bugs often don’t crash; they just degrade model performance or leak information.

### Debugging and sanity checks

Low-friction tactics:

- **Visualize attention for a toy example**:
  - Use a tiny batch, `L = 4`, `heads = 1`.
  - Plot `attn_weights[0, 0]` as a heatmap.
  - For causal masks, verify upper-right triangle is ~zero.
- **Numerical assertions**:
  - After softmax, check:
    ```python
    masked_probs = attn_weights[causal_mask]
    assert (masked_probs < 1e-5).all()
    ```
  - Check each row sums to ~1:
    ```python
    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    ```
- **Unit tests with known patterns**:
  - Construct simple logits:
    - Identity: large values on diagonal, small elsewhere.
    - With a mask that allows only self-attention, test that:
      - Each token’s highest attention is to itself.
      - Off-diagonal attention is near zero.
  - For causal tests:
    - Give future tokens very high logits.
    - Verify the model *still* does not attend to them.

Combining small visualizations, row-sum checks, and hand-crafted logits catches nearly all padding/causal masking mistakes before they turn into mysterious training failures.

## Edge Cases and Failure Modes of Self-Attention

Self-attention is powerful, but it has sharp edges in practice. Understanding where it breaks helps you design more robust models and training setups.

### Very Long Sequences

For a sequence of length `L`, standard self-attention builds an `L × L` attention matrix per head. This has two major consequences:

- **Quadratic memory and compute**  
  - Memory: `O(L^2 * H)` for `H` heads (plus activations for backprop).  
  - Compute: matrix multiplications also scale as `O(L^2 * d)` where `d` is head dimension.  
  - Practically, doubling the context length can ~quadruple attention cost, making very long contexts (e.g., tens of thousands of tokens) unstable or impossible on a single GPU.

- **Vanishingly small attention weights**  
  With long sequences, each query attends over many keys. Even if softmax distributes probability mass non-uniformly, many tokens get extremely small weights (close to numerical underflow), so:
  - Gradients through those positions become tiny.
  - The model may effectively “ignore” a large portion of the context, undermining the point of long-range modeling.

- **Numerical instability in softmax**  
  Attention logits (`QKᵀ / √d`) can cover a wide range. For large `L` and large `d`:
  - Some logits can be very positive/negative, causing `exp(logit)` to overflow or underflow in float32.
  - This can manifest as `NaN` attention matrices, exploding losses, or gradients that suddenly become `inf`.

Mitigations include:
- **Truncation / windowing**: restrict attention to a sliding window around each token.
- **Chunking**: process sequences in chunks and optionally pass summaries between chunks.
- **Stable softmax implementations**: always subtract the per-row max before exponentiation.

### Head Collapse and Degenerate Patterns

In theory, multiple heads should learn diverse patterns. In practice, you often see:

- **Uniform attention**: a head assigns nearly equal weights to all tokens. This behaves like an average-pooling layer and carries little structure-specific information.
- **Highly peaked attention**: a head always locks onto:
  - The current token (self-attention degenerates to an MLP-like behavior), or
  - Special tokens (e.g., BOS/CLS), regardless of content.
- **Redundant heads**: multiple heads learn nearly identical maps, reducing effective model capacity.

Symptoms:
- Lower utilization of model size: large models behaving like smaller ones.
- Poor generalization on tasks requiring multiple distinct relationships (e.g., syntax vs. coreference vs. long-range dependencies).

Mitigations:
- **Initialization** that avoids extremely large or tiny logits early in training.
- **Regularization** encouraging head diversity (e.g., penalties on similarity between attention maps, or dropout on entire heads).
- **Monitoring** head-wise patterns and pruning persistently redundant heads.

### Sensitivity to Tokenization and Rare Tokens

Self-attention is only as good as the representations it operates on:

- **Suboptimal tokenization**:
  - Over-fragmentation (e.g., rare words split into many subwords) scatters information across multiple tokens.
  - Self-attention must re-aggregate these pieces; if it fails, meaning is diluted.

- **Rare tokens and OOV-like behavior**:
  - Embeddings for infrequent tokens are poorly trained.
  - Their keys/queries may be noisy, leading to:
    - Low attention weights when they should matter.
    - Erratic spikes of attention in irrelevant contexts.

Downstream effects:
- Worse performance on domain-specific terminology, names, or code identifiers.
- Unstable behavior for sequences dominated by rare or out-of-domain tokens.

Mitigations:
- **Domain-adapted tokenizers** (e.g., additional merges for frequent domain terms).
- **More training on domain data** so rare tokens become less rare.
- **Embedding regularization** and tying (e.g., sharing input/output embeddings) to stabilize representations.

### Positional and Counting Failures

Self-attention is permutation-invariant without positional information. Even with positional encodings, certain tasks are tricky:

- **Off-by-one and order-related errors**:
  - Counting tasks (“the 5th item in a list”), matching parentheses, or strict sequence labeling can fail if:
    - Positional encodings are weak or not expressive enough.
    - The network is too shallow to compose position-dependent patterns.

- **Pattern confusion**:
  Models may:
  - Mix up repeated structures (e.g., confusing the first `if` with the second).
  - Mis-handle long-distance order constraints where many similar tokens appear.

Mitigations:
- **Rich positional encodings** (sinusoidal, learned, relative position bias).
- **Sufficient depth** to build hierarchical patterns.
- **Architectural bias** for ordering (e.g., relative attention, local windows, or combining with convolutions/RNNs for specific tasks).

### Mitigation Tactics in Practice

A few practical levers you can pull:

- **Truncation and chunking**
  - Hard truncate inputs to a maximum length.
  - Use chunk-level processing with overlap to preserve local context.

- **Sparse or approximate attention**
  - Restrict attention to:
    - Local windows (banded attention).
    - Top-k keys per query.
    - Predefined patterns (block sparse, strided).
  - Use approximate algorithms (e.g., kernel approximations, clustering) to reduce `O(L^2)` to something closer to linear.

- **Initialization and regularization**
  - Initialize `Q`, `K`, `V` weights so that early logits are in a moderate range (e.g., scaled Xavier/He).
  - Apply:
    - Dropout on attention weights.
    - Head dropout.
    - Weight decay or norm constraints to stabilize training.

- **Monitoring attention diversity**

  Simple metric sketch (pseudocode) to track head collapse during training:

  ```python
  import torch

  def attention_diversity(attn):  # attn: [batch, heads, seq, seq]
      # Compute average attention map per head
      # Then measure pairwise cosine similarity between heads
      B, H, S, _ = attn.shape
      attn_mean = attn.mean(dim=0)        # [H, S, S]
      flat = attn_mean.reshape(H, -1)     # [H, S*S]
      norm = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
      sim = norm @ norm.t()              # [H, H]
      # We want off-diagonal similarity to be not too close to 1
      off_diag = sim[~torch.eye(H, dtype=bool)]
      return off_diag.mean().item()
  ```

  You can log this metric and watch for:
  - Values approaching 1.0 → heads very similar (collapse).
  - Reasonable mid-range values → diverse head behavior.

Combining these tactics yields models that handle longer sequences more gracefully, avoid degeneracy in attention patterns, and remain robust to real-world token distributions and ordering constraints.

## Performance and Memory Considerations for Self-Attention

Self-attention is powerful but expensive. Understanding where the cost comes from helps you size models and hardware realistically and avoid nasty surprises in production.

### Computational complexity

Consider a single self-attention layer with:

- Batch size: `B`
- Sequence length: `L`
- Model dimension: `d_model`
- Number of heads: `H`
- Per-head dimension: `d_head = d_model / H`

For one head, you project inputs `X ∈ ℝ^{B×L×d_model}` to:

- `Q, K, V ∈ ℝ^{B×L×d_head}`

The heavy step is the score matrix:

- `S = Q Kᵀ`, where per batch, `Q ∈ ℝ^{L×d_head}`, `Kᵀ ∈ ℝ^{d_head×L}`  
- Multiplication cost per head, per batch: `O(L² · d_head)`
- Across `H` heads and `B` batches: `O(B · H · L² · d_head)`

This quickly dominates:

- Doubling sequence length `L` roughly *quadruples* attention FLOPs.
- Doubling number of heads `H` doubles cost (for fixed `d_head`).
- Increasing `d_model` typically increases `H` or `d_head`, increasing cost linearly in `d_head`.

### Memory: where it actually goes

During training, major consumers (per layer) are:

- **Input / hidden states**: `X ∈ ℝ^{B×L×d_model}`  
  - Scales linearly with `B` and `L`.
- **Q, K, V activations**: `B×L×H×d_head` each  
  - Total ~ `3 · B · L · H · d_head`
- **Attention weights**: `A ∈ ℝ^{B×H×L×L}`  
  - Quadratic in `L`: `B · H · L²`  
  - This is usually the main memory bottleneck for long sequences.
- **Gradients**: roughly another copy of activations, especially for Q/K/V and attention weights.

Key scaling:

- Doubling `L` → ~4× memory for attention weights.
- Increasing `H` → linear growth in both compute and memory for weights and Q/K/V.
- Larger `d_model` → linear growth in hidden states and Q/K/V.

### Practical knobs to reduce cost

Typical levers, from least to most painful:

- **Mixed-precision (FP16/BF16)**  
  - Halves memory for activations; often increases throughput.
  - Ensure numerically stable implementations (e.g., scaled dot-product attention with safe softmax).

- **Gradient checkpointing**  
  - Recompute activations during backward instead of storing them all.
  - Trade extra compute for reduced peak memory (often 30–50% savings).
  - Frameworks usually offer this layer-wise.

- **Reduce sequence length `L`**  
  - The single most impactful knob because cost is O(L²).
  - Truncate, window, or chunk inputs when full context isn’t needed.
  - For classification, consider using only prefix tokens or pooled representations.

- **Reduce head count `H` or `d_model` / `d_head`**  
  - Fewer / smaller heads shrink Q/K/V and attention weights.
  - May hurt modeling capacity; consider pruning or distillation to find redundancies.

- **Reduce batch size `B`**  
  - Linear reduction in memory and compute per step.
  - Compensate via gradient accumulation if you need a large effective batch size.

A minimal sketch of how some of this appears in code:

```python
import torch
import torch.nn.functional as F

def self_attn(x, W_q, W_k, W_v, W_o, n_heads):
    B, L, d_model = x.shape
    d_head = d_model // n_heads

    q = x @ W_q      # (B, L, d_model)
    k = x @ W_k
    v = x @ W_v

    # reshape to (B, n_heads, L, d_head)
    def split_heads(t):
        return t.view(B, L, n_heads, d_head).transpose(1, 2)

    q, k, v = map(split_heads, (q, k, v))

    # attention scores: (B, n_heads, L, L)
    scores = q @ k.transpose(-2, -1) / (d_head ** 0.5)
    attn = F.softmax(scores, dim=-1)
    out = attn @ v  # (B, n_heads, L, d_head)

    out = out.transpose(1, 2).contiguous().view(B, L, d_model)
    return out @ W_o  # (B, L, d_model)
```

Even in this tiny example, you can see where `B`, `L`, `n_heads`, and `d_head` appear in the tensor shapes and thus govern cost.

### Inference-time trade-offs

At inference, no gradients are stored, but attention is still O(L²):

- **Batching short sequences**  
  - Many short sequences in one batch maximize GPU utilization and throughput.
  - Good for online services with variable-length requests; pad to a reasonable max length but avoid huge outliers.

- **Few long sequences**  
  - Long `L` increases latency and can blow memory because of `L²` attention weights.
  - Consider chunked decoding or specialized long-context variants if long inputs are common.

- **Caching K/V in causal models**  
  - Autoregressive decoding can reuse past `K` and `V`:
    - At step `t`, compute attention against all past tokens with cached K/V.
    - Per-step cost becomes O(t · d_head · H) instead of recomputing from scratch.
  - Memory holds cached K/V per layer: `O(L · H · d_head)` instead of `L²`.
  - Trade-off:
    - Better throughput for long generations.
    - Higher per-request memory footprint, especially with many concurrent requests.

- **Latency vs. throughput**  
  - Larger batches → higher throughput, but higher per-request latency.
  - For real-time applications, cap batch size or use dynamic batching with tight timeouts.

### Observability and detecting pathologies

You won’t manage what you don’t measure:

- **Instrument GPU memory usage**
  - Log peak and per-layer memory for typical and worst-case inputs.
  - Track OOM events with associated `L`, `B`, and model config.

- **Track per-layer FLOPs / runtime**
  - Use profiling tools to measure:
    - Time spent in attention vs. MLP.
    - Which layers dominate runtime (often early or middle layers with largest `L`).
  - Helps prioritize optimizations (e.g., apply more aggressive tricks to bottleneck layers).

- **Log sequence-length distribution**
  - Record histograms of `L` in both training and production.
  - Watch for:
    - Rare but extremely long inputs causing spikes in latency/memory.
    - Drift over time (e.g., users pasting full documents instead of short prompts).
  - Implement guards:
    - Hard caps on `L`.
    - Automatic truncation or rejection of pathological requests.

Closing the loop between theory (O(B · H · L² · d_head)) and observability data lets you choose model sizes and deployment settings that are sustainable rather than just barely working.

## Interpreting and Using Attention Weights Safely

Inspecting attention starts with getting the weights out of your model. In most Transformer implementations you can enable an option like `output_attentions=True` and receive a tensor shaped roughly like:

`attn: (num_layers, batch_size, num_heads, seq_len, seq_len)`

Each `attn[l, b, h]` is a token–token matrix: how much head `h` in layer `l` attends from each query token (rows) to each key token (columns).

On a short example sequence:

```python
import torch

def inspect_attention(model, tokenizer, text):
    encoded = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)

    # attentions: list[length = num_layers] of (batch, num_heads, seq, seq)
    attentions = torch.stack(outputs.attentions)  # (L, B, H, S, S)
    L, B, H, S, _ = attentions.shape

    # Aggregate across heads (simple average)
    avg_over_heads = attentions.mean(dim=2)       # (L, B, S, S)

    # Aggregate across layers (simple average)
    avg_over_layers = avg_over_heads.mean(dim=0)  # (B, S, S)

    # Get single example
    attn_matrix = avg_over_layers[0]              # (S, S)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    return attn_matrix, tokens
```

You can now visualize `attn_matrix` as a heatmap over a token–token grid:

- Rows: query tokens (where information is read from).
- Columns: key tokens (where information is attended to).
- Cell color: attention weight.

Common visualization approaches:

- **Global averaged heatmap:**  
  Average over heads and/or layers, then show a single `seq_len × seq_len` heatmap. This gives a coarse sense of “which tokens talk to which.”
- **Head-specific views:**  
  Plot one heatmap per head (possibly only for one layer). This reveals specialized heads (e.g., punctuation, long-distance dependencies).
- **Head selection/aggregation:**
  - Rank heads by some metric (e.g., average attention to certain token types) and inspect the top few.
  - Cluster heads based on their attention patterns, then visualize one representative per cluster.
  - For debugging, compare the same head across different inputs to see if it behaves consistently.

When interpreting these patterns, several caveats are critical:

- Attention weights are **not guaranteed causal importance**. High attention from token A to B does *not* mean B is the main reason for the model’s prediction.
- Scaling (e.g., `1/sqrt(d_k)` in dot-product attention), softmax temperature, and normalization within each row can make weights look sharper or flatter without changing true influence.
- Gradient-based attributions (e.g., input gradients, integrated gradients) often **disagree** with attention maps. That disagreement is a hint that attention is at best a partial view.
- Multi-head attention can use some heads mostly for routing, position, or other internal signals that are not semantically meaningful to humans.

Because of these limitations, using raw attention scores directly in business logic is risky. Examples of brittle patterns:

- Treating any attention weight above a threshold as “ground truth” explanation.
- Implementing rules like “if token X gets the highest attention, then classify as Y.”
- Using attention maps as the sole signal for compliance, safety, or critical decisions.

Safer patterns include:

- Using attention as a **soft hint** for UX (e.g., highlighting tokens that might be relevant) while still relying on the model’s overall output probabilities.
- Employing attention maps for **exploratory analysis** and debugging, not as formal explanations.
- Combining attention with other signals (gradients, perturbation tests, counterfactual inputs) when you want stronger interpretability.

To improve your confidence, build small unit-test-style probes:

- Construct synthetic inputs with **known relational structure**:
  - Simple coreference: “Alice gave Bob a book. She smiled.”  
    Check if some heads in later layers strongly connect “She” to “Alice.”
  - Simple arithmetic or matching: “A: 7, B: 3, answer: A.”  
    Check if the token “answer” attends to “A.”
- Verify that **at least some heads** reflect these relationships in a plausible way, while accepting that:
  - Many heads will be opaque or mixed.
  - Different layers might specialize differently (lower layers: local patterns; higher: semantic relations).

Treat attention weights as one debugging and interpretability lens among several, not as a faithful map of “what the model is really thinking.”
