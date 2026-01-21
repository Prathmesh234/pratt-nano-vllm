# Nano-vLLM Build Plan

Rebuild `nano-vllm` step-by-step to understand the internals of high-performance LLM serving.

## Phase 1: Configuration & Data Structures
1. **`config.py`**: Learn how to define model hyperparameters (hidden size, heads, etc.) and vLLM-specific settings like block size.
2. **`sampling_params.py`**: Understand how to encapsulate generation settings (temperature, top-p) into a reusable state.
3. **`engine/sequence.py`**: Learn to represent a single request as a "Sequence" that tracks prompt tokens, generated tokens, and KV cache slot mapping.

## Phase 2: Primitive Layers
4. **`layers/rotary_embedding.py`**: Implement RoPE for position encoding—the backbone of modern LLMs.
5. **`layers/layernorm.py` & `layers/activation.py`**: Build RMSNorm and SwiGLU components for numerical stability and non-linearity.
6. **`layers/linear.py`**: Implement specialized Linear layers that handle weight loading and potential quantization/parallelism structures.
7. **`layers/embed_head.py`**: Handle the input embedding and the final LM head for token prediction.

## Phase 3: Paged Attention & Model
8. **`layers/attention.py`**: **Critical.** Learn Grouped Query Attention (GQA) and how to interface with PagedAttention kernels to fetch KV blocks from non-contiguous memory.
9. **`models/qwen3.py`**: Assemble the building blocks into a full Transformer architecture, specifically the Qwen/Llama style.

## Phase 4: The Engine Core
10. **`engine/block_manager.py`**: Implement the "Memory OS." Learn how to manage a physical pool of KV cache blocks and map logical sequence tokens to physical memory.
11. **`engine/scheduler.py`**: Build the logic that decides which requests to run, which to preempt, and how to maximize throughput.
12. **`engine/model_runner.py`**: Create the bridge between the engine and the model. Learn how to prepare input tensors and execute the forward pass for a whole batch.

## Phase 5: Public API
13. **`engine/llm_engine.py`**: Coordinate the Scheduler, Block Manager, and Model Runner into a unified execution loop.
14. **`llm.py`**: Create the user-facing class for offline inference (e.g., `llm.generate(...)`).

## Phase 6: Testing
15. **`example.py`**: Verify your hand-coded implementation with a real generation task.
