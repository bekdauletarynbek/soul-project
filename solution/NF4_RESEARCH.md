# NF4 kernel — research notes, not acted on yet

Written while deciding whether to write a Triton NF4 dequantization kernel (idea suggested
from https://github.com/Griffith-7/nf4-triton-kernel, code not copied — see below for why
nothing was built from it yet). Source: `bitsandbytes-foundation/bitsandbytes` tag `0.50.2`
(the pinned version for this task), fetched directly from GitHub, not from memory.

## What is confirmed, exactly (safe to reuse later)

- **NF4 codebook (16 values)**, `bitsandbytes/functional.py::get_4bit_type("nf4")`:
  `[-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
  -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
  0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
  0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]` — hardcoded in bnb
  itself (comment: "Values are hardcoded to avoid a scipy dependency at runtime"), so this is
  not at risk of silently drifting between bnb builds.
- **Nibble packing order**, `csrc/kernels.cu::kDequantizeBlockwise` (NF4 branch):
  `vals[j*2] = dDequantizeNF4(qvals[j] >> 4) * local_abs_max; vals[j*2+1] =
  dDequantizeNF4(qvals[j] & 0x0F) * local_abs_max;` — high nibble is the **even** index,
  low nibble is the **odd** index, scaled by one absmax per `blocksize` (64 by default).
- **Double quantization** (`compress_statistics=True`, which is what "double quant" in our
  README means): `offset = absmax.mean()`, then `absmax - offset` is itself quantized with
  **8-bit blockwise** (not NF4) at **blocksize 256** (`quantize_4bit` in `functional.py`).
  Nested blocksize/dtype/offset all travel in `QuantState` (`nested_absmax`,
  `nested_blocksize=256`, `nested_offset`, `nested_quant_map`).

## Why nothing was wired in (the actual blocker, found only after this research)

`bitsandbytes/nn/modules.py::Linear4bit.forward` does **not** call a standalone dequantize
step we could drop a Triton kernel next to. It calls one fused op:

```python
return bnb.matmul_4bit(x, self.weight, bias=bias, quant_state=quant_state).to(inp_dtype)
```

`matmul_4bit` dispatches to bitsandbytes' own hand-written, SM-version-specific fused
dequant+GEMM/GEMV CUDA kernels (`csrc/gemm_4bit_sm75.cu`, `gemm_4bit_sm80.cu`,
`gemm_4bit_simt.cu` — three different kernels picked by compute capability). A "NF4
dequantization kernel" that actually runs during training is therefore not a small pointwise
op we can fuse in next to RMSNorm/cross-entropy — it means **replacing bitsandbytes' own
tuned fused GEMM** with our own Triton GEMM that dequantizes on the fly, for every `q_proj`/
`v_proj`/`k_proj`/`o_proj`/MLP matmul in the model, forward **and** backward (`x` needs a
gradient through the frozen dequantized weight for the LoRA adapters upstream to receive one
at all).

That is a different, much bigger task than the original "dequantize 4-bit weights" framing:
writing a Triton GEMM that is both correct and not slower than a vendor's SM-specific hand
tuned kernel, with zero ability to benchmark or numerically verify it without a GPU. A
mismatch here would not just cost a few % of speed — any gradient-to-weight-dtype mistake
changes every matmul in the model, which is exactly the kind of "quiet but total" failure
mode `harness/check_rules.py` check 6/7 (loss curve, eval loss) is designed to catch, and
check 8 (not >10% slower than the base) directly punishes a GEMM that loses to bnb's own.

## Decision

Not pursuing this without real Kaggle/GPU access to iterate and benchmark against
`bnb.matmul_4bit` directly. RMSNorm (`solution/rmsnorm.py`) and cross-entropy
(`solution/fused_cross_entropy.py`) stay the two kernels in `apply()`. If time and GPU access
allow revisiting this later, the codebook and packing order above are verified and ready to
use — the gap is the GEMM itself, not the dequant math.
