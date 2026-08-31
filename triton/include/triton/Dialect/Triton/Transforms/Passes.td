#ifndef TRITON_PASSES
#define TRITON_PASSES

include "mlir/Pass/PassBase.td"

def TritonCombineOps : Pass</*cli-arg*/"triton-combine", /*Op*/"mlir::ModuleOp"> {
  let summary = "combine ops";
  let description = [{
    This pass aims to optimize the five following patterns:
    - `dot(a, b, 0) + c => dot(a, b, c)`

    - `addptr(addptr(ptr, idx0), idx1) => addptr(ptr, AddI(idx0, idx1))`

    - `select(cond, load(ptrs, broadcast(cond), ???), other) =>
         load(ptrs, broadcast(cond), other)`

    - `broadcast(constant) => reshaped_constant`
    - `torch.sum(x[:,:,None].expand(-1,-1,n) * y[None,:,:].expand(m,-1,-1),1)
       => dot(x,y,splat(0))`
  }];

  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def TritonReorderBroadcast : Pass</*cli-arg*/"triton-reorder-broadcast", /*Op*/"mlir::ModuleOp"> {
  let summary = "Moves broadcast and splat after elementwise operations";
  let description = [{
    The purpose of this pass is to transform:
      - `elementwise(broadcast(a)) => broadcast(elementwise(a))`
      - `elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))`
    In the event of a match, the broadcast (or splat) operation is delayed
    and performed after the ElementWise operation.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonRewriteTensorPointer : Pass</*cli-arg*/"triton-rewrite-tensor-pointer", /*Op*/"mlir::ModuleOp"> {
  let summary = "Rewrite load/stores with tensor pointers into legacy load/stores";
  let description = [{
    This pass rewrites all load/store semantics initiated by a `tt.make_tensor_ptr` and `tt.advance` into legacy
    semantics. After this pass, `tt.make_tensor_ptr` and `tt.advance` will disappear, and it generates logics to compute
    the pointer/mask/other for each load/store.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonRewriteTensorDescriptorToPointer : Pass</*cli-arg*/"triton-rewrite-tensor-descriptor-to-pointer", /*Op*/"mlir::ModuleOp"> {
  let summary = "Rewrite load/stores of tensor descriptors into pointer load/stores";
  let description = [{
    This pass rewrites all load/store semantics initiated by a `tt.make_tensor_descriptor` into pointer semantics. After
    this pass, `tt.make_tensor_descriptor`  will disappear, and it generates logics to compute the pointer/mask/other
    for each load/store.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopUnroll : Pass</*cli-arg*/"triton-loop-unroll", /*Op*/"mlir::ModuleOp"> {
  let summary = "Loop unroller";
  let description = [{
    The pass unrolls a scf loop with tt.loop_unroll_factor attribute. The attribute specialises how many iterations
    the loop should be unrolled.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopInvariantCodeMotion : Pass</*cli-arg*/"triton-licm", /*Op*/"mlir::ModuleOp"> {
  let summary = "MLIR's LICM plus hoist load ops out of loops with masks.";
  let description = [{
    This pass uses MLIR's LICM pass as base. Additionally, it hoists load ops
    out of loops that consists of pure/read-only ops. For scf.for loops, it
    generates a trip-count check. For scf.while loops, it clones the condition
    from the before body.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopAwareCSE : Pass<"triton-loop-aware-cse", "mlir::ModuleOp"> {
  let summary = "CSE within loop bodies";

  let description = [{
    The `triton-loop-aware-cse` pass performs recursive common subexpression
    elimination within loop bodies. Unlike regular CSE, which is a single-pass
    greedy algorithm, this pass can recursively eliminate loop iteration
    arguments and subcomputations that always have the same value.
  }];
}

// Fuse two Triton kernels (tt.func) by wiring the producer's output into the
// consumer's input, and materializing a single fused kernel function.
// The pass expects both functions to be present in the same ModuleOp.
def TritonFuseKernels : Pass</*cli-arg*/"triton-fuse-kernels", /*Op*/"mlir::ModuleOp"> {
  let summary = "Fuse two Triton kernels into one based on a simple dependency mapping";
  let description = [{
    Create a new fused kernel by composing a producer and a consumer kernel and
    connecting the producer's output to the consumer's input. This pass is
    designed for elementwise kernels that share the same launch geometry and
    indexing pattern (e.g., add followed by div). It keeps the original
    functions unchanged and adds a new fused function to the module.

    Options:
      - --producer=<symbol>   Name of the producer function (e.g., add_kernel)
      - --consumer=<symbol>   Name of the consumer function (e.g., div_kernel)
      - --mapping=<str>       Mapping in the form "P->C" that indicates wiring
                               the producer's output into the consumer's input
                               position. For typical pointer-output kernels,
                               this means the consumer will not load its first
                               operand from memory but will use the in-register
                               result from the producer. Default is "2->0"
                               (map producer arg index 2 to consumer arg index 0).
  }];

  let options = [
    Option<"producer", "producer", "std::string", /*default*/"\"\"",
           "Producer function name to fuse">,
    Option<"consumer", "consumer", "std::string", /*default*/"\"\"",
           "Consumer function name to fuse">,
    Option<"mapping", "mapping", "std::string", /*default*/"\"2->0\"",
           "Mapping from producer output arg index to consumer input arg index (e.g., 2->0)">
  ];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect"
  ];
}

def TritonOptimizeFusedLoops : Pass</*cli-arg*/"triton-optimize-fused-loops", /*Op*/"mlir::ModuleOp"> {
  let summary = "Optimizes fused kernels by merging loops and forwarding values.";
  let description = [{
    This pass identifies sequential, compatible scf.for loops within a fused
    Triton kernel and merges them. It also performs store-to-load forwarding
    to eliminate memory round-trips for values produced and consumed across
    the original loop boundaries. This is crucial for optimizing naively
    fused kernels like GEMM -> Softmax.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect"
  ];
}

def TritonPointerOpt : Pass</*cli-arg*/"pointeropt", /*Op*/"mlir::ModuleOp"> {
  let summary = "Optimizes pointer operations within Triton kernels.";
  let description = [{
    This pass identifies and optimizes pointer operations within Triton kernels.
    It aims to reduce pointer indirection and improve memory access patterns,
    ultimately enhancing the performance of the generated code.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect"
  ];
}

def TritonRedundantLoadElimination : Pass</*cli-arg*/"triton-redundant-load-elimination", /*Op*/"mlir::ModuleOp"> {
  let summary = "Eliminate redundant scalar load operations in Triton kernels";
  let description = [{
    This pass identifies and eliminates redundant tt.load operations where
    the same pointer is loaded more than once with no intervening store.
    This commonly arises after kernel fusion where each original kernel
    independently loaded shared invariants (e.g., positions, strides) in
    its prologue, resulting in duplicate loads in the fused kernel.

    The pass matches:
      %a = tt.load %ptr
      ... (no tt.store to %ptr)
      %b = tt.load %ptr   <- redundant; replaced by %a

    Only scalar (non-tensor) loads are considered to avoid mismatching
    vector loads that may differ in mask or other predicates.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];
}

// def TritonDeadStoreElimination : Pass</*cli-arg*/"triton-dead-store-elimination", /*Op*/"mlir::ModuleOp"> {
//   let summary = "Eliminates dead store operations in Triton kernels.";
//   let description = [{
//     This pass removes store operations that are never read (dead stores).
//     It is safe to delete stores to temporary buffers (marked with temp_buffer attribute)
//     if they are never loaded. This optimization reduces unnecessary memory traffic.
//   }];
//   let dependentDialects = [
//     "mlir::triton::TritonDialect",
//     "mlir::arith::ArithDialect"
//   ];
// }

def TritonOptimizeCausalMask : Pass</*cli-arg*/"triton-optimize-causal-mask", /*Op*/"mlir::ModuleOp"> {
  let summary = "Optimizes causal mask loading in Flash Attention by generating masks on-the-fly";
  let description = [{
    This pass identifies mask load operations in Flash Attention kernels and replaces
    them with dynamically generated causal masks. Instead of loading a pre-computed
    upper-triangular mask from memory (where elements above the diagonal are -inf),
    the pass generates the mask pattern directly in the kernel based on the current
    block's position in the global matrix.
    
    The optimization computes:
    - Global row indices: row_start = pid_m * BLOCK_M + local_row
    - Global column indices: col_start = loop_iter + local_col  
    - Causal condition: mask[i,j] = (row_i < col_j) ? -inf : 0
    
    This eliminates memory bandwidth consumption for mask loading and allows
    the compiler to optimize the mask generation together with other operations.
  }];
  let options = [
    Option<"producer", "producer", "std::string", /*default*/"\"\"",
           "Producer function name to fuse">,
    Option<"maskArgIndex", "mask-arg-index", "int", /*default*/"17",
           "The argument index of the mask tensor in the kernel">,
    Option<"targetAxis", "target-axis", "int", /*default*/"0",
           "The program id axis to use for row start calculation (0 or 1)">
  ];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::math::MathDialect"
  ];
}

// Fuse additive -inf/0 mask into the C operand of tt.dot
def TritonFuseMaskIntoDot : Pass</*cli-arg*/"triton-fuse-mask-into-dot", /*Op*/"mlir::ModuleOp"> {
  let summary = "Fuse additive causal mask into tt.dot's C operand";
  let description = [{
    This pass identifies patterns of the form:

      - addf(truncf(tt.dot(a,b,0)), mask_f16) (optionally followed by extf)
      - addf(tt.dot(a,b,0), mask_f32)

    where mask is a tensor consisting of -inf on masked positions and 0
    elsewhere (typically produced by select(cmp(rows, cols), -inf, 0)).
    It rewrites them into a single tt.dot(a, b, mask) with the mask cast to
    the accumulator dtype as needed. This eliminates extra cast/add and aligns
    with the optimized TL style that feeds the bias into dot directly.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect"
  ];
}

def TritonFuseSoftmaxGemm : Pass</*cli-arg*/"triton-fuse-softmax-gemm", /*Op*/"mlir::ModuleOp"> {
  let summary = "Fuse softmax and GEMM by moving division after dot product";
  let description = [{
    This pass implements the Flash Attention optimization strategy for softmax + GEMM fusion.
    
    Pattern to optimize:
    ```
    // First loop: compute max and sum for softmax
    %max, %sum = scf.for ... {
      %qk = dot(Q, K)  // M x N
      %max_val = reduce_max(%qk)
      %exp_val = exp(%qk - %max_val)
      %sum_val = reduce_sum(%exp_val)
      yield %max_val, %sum_val
    }
    
    // Second loop: compute softmax then dot with V
    %result = scf.for ... {
      %softmax = (load - %max) / %sum  // element-wise operations
      %out = dot(%softmax, V)          // M x K (K << N)
      yield %out
    }
    ```
    
    Optimized to:
    ```
    %result = scf.for ... {
      %unnormalized = exp(load - %max)
      %out = dot(%unnormalized, V)     // M x K matrix
      yield %out
    }
    %final = %result / broadcast(%sum)  // Single division on M x K matrix
    ```
    
    This reduces computation by moving the division outside the loop,
    operating on a smaller M x K matrix instead of M x N per iteration.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::math::MathDialect"
  ];
}

def TritonFlashAttentionFusion : Pass</*cli-arg*/"triton-flash-attention-fusion", /*Op*/"mlir::ModuleOp"> {
  let summary = "Fuse two loops into one for Flash Attention pattern";
  let description = [{
    This pass merges two consecutive loops:
    1. First loop: computes QK, max, sum (online softmax)
    2. Second loop: loads QK, normalizes, does dot with V
    
    Into a single loop:
    1. Merged loop: computes QK, max, sum, dot with V (accumulates unnormalized)
    2. Final normalization outside loop: result / sum
    
    This eliminates redundant loads and stores of intermediate QK scores.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::math::MathDialect"
  ];
}

// Post-LICM micro-optimizations for softmax math
def TritonSoftmaxMicroOpts : Pass</*cli-arg*/"triton-softmax-micro-opts", /*Op*/"mlir::ModuleOp"> {
  let summary = "Fold exp2(log2(e)*x)->exp(x) and divf(X,broadcast(S))->mulf(X,broadcast(recip(S)))";
  let description = [{
    This pass performs local math simplifications commonly appearing in
    fused softmax/attention: it folds exp2(log2(e) * x) into exp(x), and
    rewrites division by a broadcasted tensor into multiplication by a
    broadcasted reciprocal. These changes reduce arithmetic intensity and
    enable further CSE/canonicalization without changing numerical semantics.
  }];
}

// Hoist loop-invariant load operations out of scf.for loops
def TritonHoistInvariantLoads : Pass</*cli-arg*/"triton-hoist-invariant-loads", /*Op*/"mlir::ModuleOp"> {
  let summary = "Hoist loop-invariant tt.load operations outside scf.for loops";
  let description = [{
    This pass identifies and hoists load operations that are loop-invariant
    (i.e., their pointer and mask don't depend on loop iteration variables)
    out of scf.for loops. This is particularly beneficial for attention
    mechanisms where Q matrix can be loaded once before iterating over K/V.
    
    The pass performs the following transformations:
    1. Identifies tt.load operations inside scf.for loops
    2. Checks if the load is loop-invariant (pointer, mask, other don't change)
    3. Hoists the load and all its dependent operations before the loop
    4. Replaces uses inside the loop with the hoisted value
    
    Benefits:
    - Reduces memory traffic (avoid redundant loads)
    - Improves cache locality
    - Enables better register allocation
    - Can lead to 2-3x performance improvements in attention kernels
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect"
  ];
}

// Remove float cast round-trips like truncf(extf(x)) and extf(truncf(x)), and
// collapse nested same-direction chains extf(extf(x)) / truncf(truncf(x)).
def TritonEliminateCastRoundtrip : Pass</*cli-arg*/"triton-eliminate-cast-roundtrip", /*Op*/"mlir::ModuleOp"> {
  let summary = "Eliminate float cast round-trips and collapse cast chains";
  let description = [{
    This pass removes redundant float cast round-trips:
      - truncf(extf(x)) -> x (lossless)
      - extf(truncf(x)) -> x (semantics-changing; enabled on request)
      - collapse nested extf/extf and truncf/truncf chains to the direct cast.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect"
  ];
}

// Move scale before loop by pre-scaling Q once: rewrite mulf(dot(Q,K), scale)
// to dot(Q*scale, K) when Q is loop-invariant and scale is a scalar splat.
def TritonPreScaleQuery : Pass</*cli-arg*/"triton-pre-scale-query", /*Op*/"mlir::ModuleOp"> {
  let summary = "Hoist scale and pre-scale Query to avoid per-iter mul";
  let description = [{
    This pass identifies mulf(dot(Q,K), scale) inside the softmax loop and
    rewrites it to dot(Q*scale, K) by hoisting a one-time pre-scaling of Q
    outside the loop. The pre-scale uses f16 to match Q precision and avoid
    extra conversions, which slightly changes rounding.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::TritonDialect"
  ];
}

// Remove redundant dense zero constants and generate them on-demand
def TritonRemoveRedundantZeroConstants : Pass</*cli-arg*/"triton-remove-redundant-zero-constants", /*Op*/"mlir::ModuleOp"> {
  let summary = "Remove redundant dense zero constant tensors";
  let description = [{
    This pass identifies dense zero constants (arith.constant dense<0.0>)
    that occupy large register space and replaces their uses with on-demand
    generation using scalar constants and splat operations. This reduces
    global register pressure significantly, especially for large tensor
    dimensions (e.g., 128x128 or 128x64 tensors).

    The pass:
    1. Finds all dense zero constant tensors
    2. For each use, generates a local zero tensor (scalar 0 + splat)
    3. Removes now-unused dense constants
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect"
  ];
}

// Move V loads earlier within the softmax loop to hide latency
def TritonEarlyLoadV : Pass</*cli-arg*/"triton-early-load-v", /*Op*/"mlir::ModuleOp"> {
  let summary = "Move tt.load of V earlier in scf.for to overlap latency";
  let description = [{
    This pass identifies tt.load(V) inside the Flash Attention loop whose
    result feeds the dot with V, and moves the load earlier in the loop body
    as soon as its address/mask dependencies are available and before heavy
    compute (exp/reduce), to better hide global memory latency.

    It conservatively avoids reordering across store operations and only
    moves loads inside the same loop body.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

// Add tt.divisibility = 16 to all tt.func arguments if missing
def TritonAddDivisibilityAttr : Pass</*cli-arg*/"triton-add-divisibility-attr", /*Op*/"mlir::ModuleOp"> {
  let summary = "Add tt.divisibility=16 to all function arguments";
  let description = [{
    This pass walks all Triton kernel functions (tt.func) and ensures that
    each argument carries the attribute `tt.divisibility = 16 : i32`.
    Existing `tt.divisibility` attributes are preserved. This is useful for
    pipelines that rely on these alignment hints but where the upstream
    generator omitted them.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];
}

def TritonCombineTensorScalarMults : Pass</*cli-arg*/"triton-combine-scalar-mults", /*Op*/"mlir::ModuleOp"> {
  let summary = "Combine tensor scalar multiplications before and inside loops";
  let description = [{
    This pass combines two scalar multiplication operations:
    1. Before loop: load(query) * scalar_arg  (e.g., %21 * %23)
    2. Inside loop: dot(...) * constant       (e.g., %67 * %cst_2)
    
    It merges both scalar factors into a single combined constant inside the loop,
    and removes the redundant pre-loop multiplication. This reduces memory bandwidth
    and computational overhead.
    
    Pattern: 
      Before: %query_scaled = load(query) * %arg16
              %dot_result = tt.dot(%query_scaled, %key, ...)
              %result = %dot_result * %cst_2
      
      After:  %query = load(query)
              %dot_result = tt.dot(%query, %key, ...)
              %combined_cst = %arg16 * %cst_2
              %result = %dot_result * %combined_cst
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

def TritonDeadStoreElimination : Pass</*cli-arg*/"triton-dead-store-elimination", /*Op*/"mlir::ModuleOp"> {
  let summary = "Eliminate dead store operations";
  let description = [{
    This pass eliminates store operations that are never subsequently loaded.
    It provides safety checks:
    - Only stores to temporary buffers (marked with argN.temp_buffer) can be eliminated
    - Function arguments without the temp_buffer attribute are protected
    - Specific argument indices can be preserved from elimination
  }];
  
  let options = [
    ListOption<"preserveArgs", "preserve-args", "int32_t",
      "Argument indices whose stores should not be eliminated (comma-separated)",
      "llvm::cl::ZeroOrMore">
  ];
  
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif
