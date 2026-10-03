# Reuse and customization

This document describes how to reuse and customize TileFusion, including changing the NVIDIA GPU, adding a computation graph, setting tensor inputs, and registering a new operator.

## Platform

TileFusion is implemented on Triton 3.5. It runs on NVIDIA GPUs supported by that release (Ampere and newer), e.g. A100, H100, and H800. Switching among these GPUs needs no extra porting: the same scripts run on any NVIDIA device that can run Triton 3.5.

## Inputs

Input tensors are prepared in the workload scripts (e.g. `prepare()` in `triton/tilefusion/models/attn_model.py`). To customize them, change `--seqlen` or edit `prepare()` directly.

## Benchmarks

To add a new computation graph, refer to `triton/tilefusion/models/attn_model.py` as an example.

A graph has **inputs**, **operator nodes**, and **outputs**. Construct the graph first, then compile it.

Construct:

- `add_input(*tensor_names)`: required. Names of the graph inputs.
- `add_node(kernel_name, inputs=None, parents=None, output=None, **params)`: `kernel_name` and `parents` are required. `inputs`, `output`, and `params` depend on that kernel.
- `add_output(name, source, shape)`: required. Output name, producing-node index, and tensor shape. `dtype`, `device`, and `output_index` are optional.

Compile and run:

Call `graph.compile(...)`. It returns `(best_compiled, best_partition, best_time)`. Wrap those as a callable (in this example, `FusedAttentionOp`) and run it from `main()` on tensors from `prepare()`.

## Add an operator

Adding a new operator needs both an **implementation** and a **registration**, taking ReLU as an example.

1. **Implementation:** provide the Triton kernel under `triton/tilefusion/ops/` (e.g. `triton/tilefusion/ops/relu.py`).

2. **Registration:** provide `KernelMetadata` by calling `register_kernel` inside `register_standard_kernels` in `triton/tilefusion/core/kernel_registry.py`.

```python
register_kernel(KernelMetadata(
    name=...,            # key for add_node(kernel_name)
    family=...,          # search may swap variants in the same family
    ttir_symbol=...,     # must match the @triton.jit function name; used when fusing two kernels
    tensors=[            # TensorSpec for every tensor pointer (all inputs and outputs)
        TensorSpec(
            name=...,    # tensor name used when wiring add_node(inputs=...)
            role=...,    # "input" | "output" | "input_output"
            dims=...,    # dimension names; used if the runtime must allocate the tensor
            dtype=...,
            arg_index=...,       # argument slot of this pointer
            stride_indices=...,  # argument slots of this tensor’s strides
        ),
    ],
    required_params=..., # scalar parameters (exclude tiling), e.g., heads, scale, …
    block_params=...,    # tiling names, e.g. BM, BN
    dim_arg_indices=..., # where shape scalars sit in the kernel args, e.g., {"M": 2, "N": 3}
    other_arg_indices=..., # where other scalars sit, e.g., {"scale": 10}
    grid_template=...,   # three string formulas for tl.program_id(0), (1), (2). Names in the formula (M, BM, ...) are scalar/block params. Example: ("cdiv(M, BM)", "batch*heads", "1") → pid(0) tiles M, pid(1) is batch*heads, pid(2) unused.
    grid_direction=...,  # which scalar dim pid(0) splits: "0" = pid(0) splits M, "1" = pid(0) splits N. This information is already in grid_template; the field exists only to simplify the current implementation and will be removed. 
))
```
