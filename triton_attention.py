"""
Triton Attention Kernels for benchmarking against native and SDPA attention.

Phase 1: Naive Triton attention (this file)
Phase 2: Flash Attention style (future)
Phase 3: 2-Pass Cascade (future)
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math
from typing import Optional, Tuple


# ============================================================================
# Naive Triton Attention Kernel
# ============================================================================

@triton.jit
def _naive_attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,  # Q strides: batch, head, seq, dim
    stride_kb, stride_kh, stride_kn, stride_kk,  # K strides
    stride_vb, stride_vh, stride_vn, stride_vk,  # V strides
    stride_ob, stride_oh, stride_om, stride_ok,  # Output strides
    batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,  # Block size for queries
    BLOCK_N: tl.constexpr,  # Block size for keys/values
    BLOCK_K: tl.constexpr,  # Block size for head dimension
):
    """
    Naive Triton attention kernel.

    Each program handles one (batch, head, query_block).
    This is NOT memory-efficient - it's meant to be simple and correct.
    """
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)  # Query block index

    # Compute offsets for this query block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to Q for this batch, head, and query block
    q_ptrs = Q + (pid_batch * stride_qb + pid_head * stride_qh +
                  offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

    # Load Q block with bounds checking
    q_mask = (offs_m[:, None] < seq_len_q) & (offs_k[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize accumulator for output and softmax statistics
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # running sum

    # Iterate over all K/V blocks
    num_blocks_n = tl.cdiv(seq_len_k, BLOCK_N)

    for block_n in range(num_blocks_n):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K block
        k_ptrs = K + (pid_batch * stride_kb + pid_head * stride_kh +
                      offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        k_mask = (offs_n[:, None] < seq_len_k) & (offs_k[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute QK^T for this block: [BLOCK_M, BLOCK_N]
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        qk *= scale

        # Apply causal mask if needed
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))

        # Mask out-of-bounds positions
        qk = tl.where(offs_n[None, :] < seq_len_k, qk, float('-inf'))

        # Online softmax: update running max and sum
        m_ij = tl.max(qk, axis=1)  # max of this block
        m_new = tl.maximum(m_i, m_ij)  # new running max

        # Correction factor for previous accumulator
        alpha = tl.exp(m_i - m_new)

        # Compute exp(qk - m_new) for this block
        p = tl.exp(qk - m_new[:, None])

        # Update running sum
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Update accumulator with correction
        acc = acc * alpha[:, None]

        # Load V block and accumulate
        v_ptrs = V + (pid_batch * stride_vb + pid_head * stride_vh +
                      offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        v_mask = (offs_n[:, None] < seq_len_k) & (offs_k[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Accumulate: p @ V
        acc += tl.dot(p.to(v.dtype), v)

        # Update running max
        m_i = m_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output
    out_ptrs = Out + (pid_batch * stride_ob + pid_head * stride_oh +
                      offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    out_mask = (offs_m[:, None] < seq_len_q) & (offs_k[None, :] < head_dim)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=out_mask)


def triton_naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = True,
) -> torch.Tensor:
    """
    PyTorch wrapper for naive Triton attention.

    Args:
        q: Query tensor [batch, heads, seq_q, dim]
        k: Key tensor [batch, heads, seq_k, dim]
        v: Value tensor [batch, heads, seq_k, dim]
        scale: Scaling factor (default: 1/sqrt(dim))
        causal: Whether to apply causal masking

    Returns:
        Attention output [batch, heads, seq_q, dim]
    """
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Allocate output
    out = torch.empty_like(q)

    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = triton.next_power_of_2(head_dim)

    # Grid dimensions: (batch, heads, num_query_blocks)
    grid = (batch_size, num_heads, triton.cdiv(seq_len_q, BLOCK_M))

    # Launch kernel
    _naive_attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
        scale,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out


# ============================================================================
# Model Injection Interface
# ============================================================================

def triton_attention_for_model(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Wrapper that matches HuggingFace's attention function signature.

    This allows injecting our Triton kernel into HuggingFace models.

    Args:
        module: The attention module (unused, for API compatibility)
        query: [batch, heads, seq, dim]
        key: [batch, heads, seq, dim]
        value: [batch, heads, seq, dim]
        attention_mask: Attention mask (unused - we use causal masking)
        scaling: Scaling factor
        dropout: Dropout probability (unused in inference)
        **kwargs: Additional arguments (ignored)

    Returns:
        Tuple of (output, attention_weights)
        - output: [batch, seq, heads * dim] (transposed and reshaped)
        - attention_weights: None (not computed for efficiency)
    """
    # Apply Triton attention
    output = triton_naive_attention(query, key, value, scale=scaling, causal=True)

    # Transpose to [batch, seq, heads, dim] then reshape to [batch, seq, heads * dim]
    output = output.transpose(1, 2).contiguous()

    return output, None


def inject_triton_attention(model) -> str:
    """
    Inject Triton attention into a HuggingFace model.

    Args:
        model: A HuggingFace model (e.g., OPTForCausalLM, LlamaForCausalLM)

    Returns:
        The original attention implementation name
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # Store original implementation
    original_impl = getattr(model.config, '_attn_implementation', 'eager')

    # Register our custom attention function
    ALL_ATTENTION_FUNCTIONS["triton_naive"] = triton_attention_for_model

    # Set the model to use our implementation
    model.config._attn_implementation = "triton_naive"

    print(f"Injected Triton naive attention (original: {original_impl})")

    return original_impl


# ============================================================================
# Correctness Verification
# ============================================================================

def verify_triton_correctness(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 256,
    head_dim: int = 64,
    tolerance: float = 0.01,
) -> bool:
    """
    Verify that Triton attention matches PyTorch native attention.

    Returns:
        True if verification passes, False otherwise
    """
    print(f"Verifying Triton correctness: batch={batch_size}, heads={num_heads}, "
          f"seq={seq_len}, dim={head_dim}")

    # Create random inputs
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    device='cuda', dtype=torch.float16)

    scale = 1.0 / math.sqrt(head_dim)

    # PyTorch reference (native attention)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1)
    scores = scores.masked_fill(causal_mask.bool(), float('-inf'))

    # Softmax and output
    attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out_pytorch = torch.matmul(attn_weights, v)

    # Triton implementation
    out_triton = triton_naive_attention(q, k, v, scale=scale, causal=True)

    # Compare
    diff = (out_pytorch - out_triton).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    passed = max_diff < tolerance
    if passed:
        print(f"  PASSED: Triton matches PyTorch within {tolerance} tolerance")
    else:
        print(f"  FAILED: Differences exceed {tolerance} tolerance")

    return passed


def verify_model_correctness(
    model_id: str,
    seq_len: int = 256,
    tolerance: float = 0.05,
    token: Optional[str] = None,
) -> bool:
    """
    Verify Triton attention produces same outputs as native attention
    when integrated into a full model.

    Args:
        model_id: HuggingFace model ID
        seq_len: Sequence length to test
        tolerance: Maximum allowed difference in logits
        token: HuggingFace token for gated models

    Returns:
        True if verification passes
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import gc

    print(f"\nVerifying model correctness: {model_id}, seq_len={seq_len}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create input
    text = "The quick brown fox jumps over the lazy dog. " * (seq_len * 2)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Test with native attention
    print("  Loading model with native attention...")
    model_native = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        token=token,
    )

    with torch.no_grad():
        out_native = model_native(**inputs)
    logits_native = out_native.logits.clone()

    del model_native
    gc.collect()
    torch.cuda.empty_cache()

    # Test with Triton attention
    print("  Loading model with Triton attention...")
    model_triton = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",  # Start with eager, then inject
        token=token,
    )
    inject_triton_attention(model_triton)

    with torch.no_grad():
        out_triton = model_triton(**inputs)
    logits_triton = out_triton.logits.clone()

    del model_triton
    gc.collect()
    torch.cuda.empty_cache()

    # Compare logits
    diff = (logits_native - logits_triton).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max logit difference: {max_diff:.6f}")
    print(f"  Mean logit difference: {mean_diff:.6f}")

    passed = max_diff < tolerance
    if passed:
        print(f"  PASSED: Model outputs match within {tolerance} tolerance")
    else:
        print(f"  FAILED: Model outputs differ by more than {tolerance}")

    return passed


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    print("="*60)
    print("TRITON ATTENTION VERIFICATION")
    print("="*60)

    # Test kernel correctness with different configurations
    configs = [
        (2, 8, 256, 64),   # OPT-like
        (2, 8, 256, 128),  # LLaMA-like
        (1, 32, 512, 64),  # Larger model
    ]

    all_passed = True
    for batch, heads, seq, dim in configs:
        passed = verify_triton_correctness(batch, heads, seq, dim)
        all_passed = all_passed and passed
        print()

    if all_passed:
        print("All kernel tests PASSED!")
    else:
        print("Some kernel tests FAILED!")
