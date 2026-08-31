"""
Kernel注册表系统

提供统一的kernel元数据管理，支持：
1. Kernel注册和查询
2. 自动参数映射推导
3. Grid兼容性检查
4. 计算图自动融合
"""

from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import uuid


class KernelType(Enum):
    """Kernel类型"""
    GEMM = "gemm"
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    CUSTOM = "custom"


@dataclass
class TensorSpec:
    """张量规格"""
    name: str
    role: str  # "input", "output", "input_output"
    dims: List[str]  # ["batch", "heads", "M", "N"]
    dtype: str  # "f16", "f32", etc.
    arg_index: int  # 在kernel参数列表中的索引
    stride_indices: List[int] = field(default_factory=list)  # stride参数的索引


@dataclass
class KernelMetadata:
    """Kernel元数据"""
    name: str
    kernel_type: KernelType
    ttir_generator: Callable  # 生成TTIR的函数
    ttir_symbol: str  # 该kernel在TTIR中的函数名（symbol）
    tensors: List[TensorSpec]
    
    # Grid配置 (使用字符串模板)
    grid_template: Tuple[str, str, str]  # 例如: ("cdiv(M, BM)", "batch*heads", "1")
    
    # Block配置
    block_params: List[str]  # ["BM", "BN"]
    
    # 唯一标识符（自动生成，必须放在有默认值的字段最后）
    id: str = field(default_factory=lambda: str(uuid.uuid4().hex[:8]))
    
    # 逻辑算子家族，用于同一算子的不同实现（variant）统一选择，例如: "gemm", "softmax", "add_mask", "scale"
    family: str = ""
    # 其他必需参数
    required_params: List[str] = field(default_factory=list)  # ["M", "N", "K", "scale", etc.]
    # 形状参数在TTIR中的位置索引，例如 {"M": 3, "N": 4, "K": 5}
    dim_arg_indices: Dict[str, int] = field(default_factory=dict)
    # 其他标量参数的索引，例如 {"scale": 1} (对于scale kernel)
    other_arg_indices: Dict[str, int] = field(default_factory=dict)
    
    # 性能参数
    default_num_warps: int = 4
    default_num_stages: int = 2
    
    # 并行维度方向 (用于优化pass)
    # "0" = M-grid (grid沿M维度划分)
    # "1" = N-grid (grid沿N维度划分)
    grid_direction: str = "0"
    
    # 备注信息
    description: str = ""

    def __post_init__(self):
        # 缺省时用 name 的前缀作为 family（如 gemm_* -> gemm）
        if not self.family:
            if "_" in self.name:
                self.family = self.name.split("_")[0]
            else:
                self.family = self.name
    
    def get_input_tensors(self) -> List[TensorSpec]:
        """获取输入张量"""
        return [t for t in self.tensors if t.role in ["input", "input_output"]]
    
    def get_output_tensors(self) -> List[TensorSpec]:
        """获取输出张量"""
        return [t for t in self.tensors if t.role in ["output", "input_output"]]
    
    def get_total_param_count(self) -> int:
        """
        计算kernel的总参数数量
        
        公式：
        - 每个tensor: 1个指针参数
        - 每个tensor的stride: len(stride_indices)个参数
        - dim_arg_indices: 形状参数数量
        - other_arg_indices: 其他标量参数数量
        """
        count = 0
        
        # 统计tensor和stride
        for tensor in self.tensors:
            count += 1  # tensor指针
            count += len(tensor.stride_indices)  # stride参数
        
        # 统计形状参数
        count += len(self.dim_arg_indices)
        
        # 统计其他参数
        count += len(self.other_arg_indices)
        
        return count
    
    def get_grid(self, **params) -> Tuple[Any, Any, Any]:
        """根据参数计算实际grid"""
        grid = []
        # 为了避免诸如先替换 "M" 导致 "BM" 变成 "B4096" 的问题，
        # 我们按变量名长度从长到短进行正则替换（使用词边界）。
        keys_sorted = sorted(params.keys(), key=len, reverse=True)
        for template in self.grid_template:
            expr = template
            for key in keys_sorted:
                value = params[key]
                # 使用词边界，保证只替换完整的标识符
                expr = re.sub(rf"\b{re.escape(key)}\b", str(value), expr)
            # 计算表达式
            grid.append(eval(expr, {"cdiv": lambda a, b: (a + b - 1) // b}))
        return tuple(grid)
    
    def create_ttir(self, **params) -> str:
        """创建TTIR"""
        # 仅向生成器传递所需参数，避免出现多余关键字参数
        allowed = set(self.required_params) | set(self.block_params) | {"scale"}
        filtered = {k: v for k, v in params.items() if k in allowed}
        return self.ttir_generator(**filtered)
    
    def get_tensor_by_name(self, name: str) -> Optional[TensorSpec]:
        """根据名称获取tensor规格
        
        Args:
            name: tensor名称
            
        Returns:
            匹配的TensorSpec，如果未找到返回None
        """
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        return None

class _KernelRegistry:
    """Kernel注册表（内部类，使用全局单例）"""
    
    def __init__(self):
        self._registry: Dict[str, KernelMetadata] = {}
        # family -> variants
        self._families: Dict[str, List[KernelMetadata]] = {}
    
    def register(self, metadata: KernelMetadata):
        """注册kernel"""
        if metadata.name in self._registry:
            print(f"Warning: Overwriting existing kernel: {metadata.name}")
        self._registry[metadata.name] = metadata
        # 维护家族索引
        fam = metadata.family
        self._families.setdefault(fam, []).append(metadata)
        # print(f"Registered kernel: {metadata.name}")
    
    def get(self, name: str) -> Optional[KernelMetadata]:
        """获取kernel元数据"""
        return self._registry.get(name)
    
    def list_all(self) -> List[str]:
        """列出所有已注册的kernel"""
        return list(self._registry.keys())
    
    def find_by_type(self, kernel_type: KernelType) -> List[KernelMetadata]:
        """按类型查找kernel"""
        return [m for m in self._registry.values() if m.kernel_type == kernel_type]

    def find_variants(self, family: str) -> List[KernelMetadata]:
        """按family查找所有变体（variants）"""
        # 保持稳定顺序：先注册的在前；也可按名称排序保证可重复性
        variants = list(self._families.get(family, []))
        # 次序规则：名称升序，作为稳定回退（避免不同运行注册顺序差异）
        variants.sort(key=lambda m: m.name)
        return variants

    def list_families(self) -> List[str]:
        return sorted(self._families.keys())

    # -------- Variant helpers for grid-based selection --------
    def get_family_default(self, family: str) -> Optional[KernelMetadata]:
        """返回某 family 的默认变体（按名称排序后的第一个）。"""
        vs = self.find_variants(family)
        return vs[0] if vs else None

    def find_variants_matching_grid(self, family: str, params: Dict[str, Any], target_grid: Tuple[Any, Any, Any]) -> List[KernelMetadata]:
        """返回在给定参数下 grid 等于 target_grid 的所有变体。"""
        matches: List[KernelMetadata] = []
        for vm in self.find_variants(family):
            try:
                g = vm.get_grid(**params)
                if g == target_grid:
                    matches.append(vm)
            except Exception:
                continue
        return matches
    
    def check_compatibility(self, kernel1_name: str, kernel2_name: str, **params) -> Tuple[bool, str]:
        """检查两个kernel是否可以融合"""
        k1 = self.get(kernel1_name)
        k2 = self.get(kernel2_name)
        
        if not k1 or not k2:
            return False, "Kernel not found in registry"
        
        # 检查grid是否兼容
        try:
            grid1 = k1.get_grid(**params)
            grid2 = k2.get_grid(**params)
            
            if grid1 == grid2:
                return True, "Grids match exactly"
            else:
                return False, f"Grid mismatch: {grid1} vs {grid2}"
        except Exception as e:
            return False, f"Error computing grid: {str(e)}"


# 全局注册表实例
_global_registry = _KernelRegistry()


def register_kernel(metadata: KernelMetadata):
    """注册kernel到全局注册表"""
    _global_registry.register(metadata)


def get_kernel(name: str) -> Optional[KernelMetadata]:
    """从全局注册表获取kernel"""
    return _global_registry.get(name)


def get_registry() -> _KernelRegistry:
    """获取全局注册表（仅供内部使用）"""
    return _global_registry


# ============================================================================
# 预定义的Kernel注册
# ============================================================================

def register_standard_kernels():
    """注册标准的kernel"""
    from tilefusion.ops.gemm_mgrid_kernel import _ttir_of_gemm
    from tilefusion.ops.gemm_mgrid_loopk_kernel import _ttir_of_gemm_loopk, _ttir_of_gemm_grid_n
    from tilefusion.ops.elementwise import _ttir_of_scale, _ttir_of_addmask, _ttir_of_tanh, _ttir_of_scale_c, _ttir_of_tanh_ngrid
    from tilefusion.ops.softmax_blockM import _ttir_of_softmax
    
    # H2O专用kernel
    from tilefusion.ops.softmax_store_sum import _ttir_of_softmax_store, _ttir_of_softmax_recompute_ngrid, _ttir_of_softmax_recompute, _ttir_of_softmax_reduce_kernel
    from tilefusion.ops.gemm_ngrid_loopm_kernel import _ttir_of_gemm_ngrid
    from tilefusion.ops.elementwize_ngrid_loopm import _ttir_of_addmask_ngrid, _ttir_of_scale_ngrid
    from tilefusion.ops.sum import _ttir_of_sum, _ttir_of_sum_dim1, _ttir_of_broadcast
    from tilefusion.ops.topk import _ttir_of_topk_bhmn
    from tilefusion.ops.scatter import _ttir_of_scatter
    
    # DSA专用kernel
    from tilefusion.ops.relu import _ttir_of_relu
    from tilefusion.ops.mul import _ttir_of_mul
    
    # Corm专用kernel
    from tilefusion.ops.mask_any import _ttir_of_mask_any
    
    # 1. GEMM (Q @ K^T)
    register_kernel(KernelMetadata(
        name="gemm_qk",
        kernel_type=KernelType.GEMM,
        family="gemm_loopn",
        ttir_generator=lambda M, N, K, BM, BN, batch, heads: _ttir_of_gemm(M, N, K, BM, BN, batch, heads),
        ttir_symbol="gemm_mgrid_kernel",
        tensors=[
            TensorSpec("Q", "input", ["batch", "heads", "M", "K"], "f16", 
                      arg_index=0, stride_indices=[6, 7, 8]),
            TensorSpec("K", "input", ["batch", "heads", "N", "K"], "f16", 
                      arg_index=1, stride_indices=[9, 10, 11]),
            TensorSpec("scores", "output", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=2, stride_indices=[12, 13, 14]),
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "K", "batch", "heads"],
        dim_arg_indices={"M": 3, "N": 4, "K": 5},
        description="Matrix multiplication: Q @ K^T"
    ))
    
    # 2. Scale
    register_kernel(KernelMetadata(
        name="scale",
        kernel_type=KernelType.ELEMENTWISE,
        family="scalex",
        ttir_generator=lambda M, N, BM, BN, scale, batch, heads: _ttir_of_scale(M, N, BM, BN, scale, batch, heads),
        ttir_symbol="scale_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=1, stride_indices=[7, 8, 9]),
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "scale", "batch", "heads"],
        dim_arg_indices={"M": 2, "N": 3},
        other_arg_indices={"scale": 10}, 
        description="Element-wise scale: input * scale"
    ))
    
    # 2b. Scale_c (constant scale, for Gemma2)
    register_kernel(KernelMetadata(
        name="scale_c",
        kernel_type=KernelType.ELEMENTWISE,
        family="scale",
        ttir_generator=lambda M, N, BM, BN, scale, batch, heads: _ttir_of_scale_c(M, N, BM, BN, scale, batch, heads),
        ttir_symbol="scale_kernel_c",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=1, stride_indices=[7, 8, 9]),
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "scale", "batch", "heads"],
        dim_arg_indices={"M": 2, "N": 3},
        # other_arg_indices={"scale": 10}, 
        description="Element-wise scale: input * scale"
    ))
    
    # 3. AddMask
    register_kernel(KernelMetadata(
        name="add_mask",
        kernel_type=KernelType.ELEMENTWISE,
        family="add_mask",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_addmask(M, N, BM, BN, batch, heads),
        ttir_symbol="add_mask_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=0, stride_indices=[5, 6, 7]),
            TensorSpec("mask", "input", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=1, stride_indices=[8, 9, 10]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=2, stride_indices=[11, 12, 13]),
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 3, "N": 4},
        description="Element-wise add: input + mask"
    ))
    
    # 4. Softmax
    register_kernel(KernelMetadata(
        name="softmax",
        kernel_type=KernelType.REDUCTION,
        family="softmax",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_softmax(M, N, BM, BN, batch, heads),
        ttir_symbol="softmax_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=1, stride_indices=[7, 8, 9]),
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 2, "N": 3},
        description="Softmax along last dimension"
    ))
    
    # 4b. Tanh (for Gemma2 logit softcapping)
    register_kernel(KernelMetadata(
        name="tanh",
        kernel_type=KernelType.ELEMENTWISE,
        family="tanh",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_tanh(M, N, BM, BN, batch, heads),
        ttir_symbol="tanh_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=1, stride_indices=[7, 8, 9]),
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 2, "N": 3},
        description="Element-wise tanh: tanh(x)"
    ))
    
    # 4b. Tanh (for Gemma2 logit softcapping)
    register_kernel(KernelMetadata(
        name="tanh_ngrid",
        kernel_type=KernelType.ELEMENTWISE,
        family="tanh",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_tanh_ngrid(M, N, BM, BN, batch, heads),
        ttir_symbol="tanh_ngrid_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=1, stride_indices=[7, 8, 9]),
        ],
        grid_template=("cdiv(N, BN)", "batch*heads", "1"),  
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 2, "N": 3},
        grid_direction="1",  # N-grid并行
        description="Element-wise tanh: tanh(x)"
    ))

    # 4c. Log_Neg (M-grid for Keyformer)
    from tilefusion.ops.elementwise import _ttir_of_log_neg
    register_kernel(KernelMetadata(
        name="log_neg",
        kernel_type=KernelType.ELEMENTWISE,
        family="log_neg",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_log_neg(M, N, BM, BN, batch, heads),
        ttir_symbol="log_neg_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16",
                      arg_index=1, stride_indices=[7, 8, 9,]),
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 2, "N": 3},
        description="M-grid Log-Neg for Keyformer Gumbel noise: y = -log(x)"
    ))
    
    # 5. GEMM_loopk (Probs @ V)
    register_kernel(KernelMetadata(
        name="gemm_pv",
        kernel_type=KernelType.GEMM,
        family="gemm_loopk",
        ttir_generator=lambda M, N, K, BM, BN, batch, heads: _ttir_of_gemm_loopk(M, N, K, BM, BN, batch, heads),
        ttir_symbol="gemm_mgrid_loopk_kernel",
        tensors=[
            TensorSpec("probs", "input", ["batch", "heads", "M", "K"], "f16", 
                      arg_index=0, stride_indices=[6, 7, 8]),
            TensorSpec("V", "input", ["batch", "heads", "N", "K"], "f16", 
                      arg_index=1, stride_indices=[9, 10, 11]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=2, stride_indices=[12, 13, 14]),
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "K", "batch", "heads"],
        dim_arg_indices={"M": 3, "N": 4, "K": 5},
        description="Matrix multiplication: Probs @ V"
    ))
    
    # ========================================================================
    # H2O专用kernels
    # ========================================================================
    
    # 6. Softmax with store (保存sum用于后续计算)
    register_kernel(KernelMetadata(
        name="softmax_store",
        kernel_type=KernelType.REDUCTION,
        family="softmax",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_softmax_store(M, N, BM, BN, batch, heads),
        ttir_symbol="softmax_fwd_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=0, stride_indices=[5, 6, 7]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=1, stride_indices=[8, 9, 10]),
            TensorSpec("sum_out", "output", ["batch", "heads", "M", "1"], "f32",
                      arg_index=2, stride_indices=[11, 12]),
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 3, "N": 4},
        description="Softmax with sum output for H2O"
    ))
    
    # 6b. Softmax reduce (只计算第一个loop，输出sum用于recompute)
    register_kernel(KernelMetadata(
        name="softmax_reduce",
        kernel_type=KernelType.REDUCTION,
        family="softmax_reduce",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_softmax_reduce_kernel(M, N, BM, BN, batch, heads),
        ttir_symbol="softmax_reduce_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("sum_out", "output", ["batch", "heads", "M", "1"], "f32",
                      arg_index=1, stride_indices=[7, 8]),
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 2, "N": 3},
        description="Softmax reduce (first pass only, outputs sum for recompute)"
    ))
    

    
    # 7. GEMM_ngrid (N维度并行的GEMM)
    register_kernel(KernelMetadata(
        name="gemm_ngrid",
        kernel_type=KernelType.GEMM,
        family="gemm_loopn",
        ttir_generator=lambda M, N, K, BM, BN, batch, heads: _ttir_of_gemm_ngrid(M, N, K, BM, BN, batch, heads),
        ttir_symbol="gemm_ngrid_kernel",
        tensors=[
            TensorSpec("Q", "input", ["batch", "heads", "M", "K"], "f16",
                      arg_index=0, stride_indices=[6, 7, 8]),
            TensorSpec("K", "input", ["batch", "heads", "N", "K"], "f16",
                      arg_index=1, stride_indices=[9, 10, 11]),
            TensorSpec("scores", "output", ["batch", "heads", "M", "N"], "f16",
                      arg_index=2, stride_indices=[12, 13, 14]),
        ],
        grid_template=("cdiv(N, BN)", "batch*heads", "1"),  # 注意：N维度并行！
        block_params=["BM", "BN"],
        required_params=["M", "N", "K", "batch", "heads"],
        dim_arg_indices={"M": 3, "N": 4, "K": 5},
        grid_direction="1",  # N-grid并行
        description="Matrix multiplication with N-grid parallelism"
    ))
    
        # 7. GEMM_ngrid (N维度并行的GEMM)
    register_kernel(KernelMetadata(
        name="gemm_pv_ngrid",
        kernel_type=KernelType.GEMM,
        family="gemm_loopk",
        ttir_generator=lambda M, N, K, BM, BN, batch, heads: _ttir_of_gemm_grid_n(M, N, K, BM, BN, batch, heads),
        ttir_symbol="gemm_ngrid_kernel",
        tensors=[
            TensorSpec("probs", "input", ["batch", "heads", "M", "K"], "f16", 
                      arg_index=0, stride_indices=[6, 7, 8]),
            TensorSpec("V", "input", ["batch", "heads", "N", "K"], "f16", 
                      arg_index=1, stride_indices=[9, 10, 11]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16", 
                      arg_index=2, stride_indices=[12, 13, 14]),
        ],
        grid_template=("cdiv(N, BN)", "batch*heads", "1"),  # 注意：N维度并行！
        block_params=["BM", "BN"],
        required_params=["M", "N", "K", "batch", "heads"],
        dim_arg_indices={"M": 3, "N": 4, "K": 5},
        grid_direction="1",  # N-grid并行
        description="Matrix multiplication with N-grid parallelism"
    ))
    
    # 8. Scale_ngrid (N维度并行)
    register_kernel(KernelMetadata(
        name="scale_ngrid",
        kernel_type=KernelType.ELEMENTWISE,
        family="scalex",
        ttir_generator=lambda M, N, BM, BN, scale, batch, heads: _ttir_of_scale_ngrid(M, N, BM, BN, scale, batch, heads),
        ttir_symbol="scale_ngrid_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16",
                      arg_index=1, stride_indices=[7, 8, 9]),
        ],
        grid_template=("cdiv(N, BN)", "batch*heads", "1"),  
        block_params=["BM", "BN"],
        required_params=["M", "N", "scale", "batch", "heads"],
        dim_arg_indices={"M": 2, "N": 3},
        other_arg_indices={"scale": 10},
        grid_direction="1",  # N-grid并行
        description="Element-wise scale with N-grid parallelism"
    ))
    
    # 9. AddMask_ngrid (N维度并行)
    register_kernel(KernelMetadata(
        name="add_mask_ngrid",
        kernel_type=KernelType.ELEMENTWISE,
        family="add_mask",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_addmask_ngrid(M, N, BM, BN, batch, heads),
        ttir_symbol="add_mask_ngrid_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[5, 6, 7]),
            TensorSpec("mask", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=1, stride_indices=[8, 9, 10]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16",
                      arg_index=2, stride_indices=[11, 12, 13]),
        ],
        grid_template=("cdiv(N, BN)", "batch*heads", "1"),  
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 3, "N": 4},
        grid_direction="1",  # N-grid并行
        description="Element-wise add with N-grid parallelism"
    ))
    
    # 10. Softmax_recompute_
    register_kernel(KernelMetadata(
        name="softmax_recompute_mgrid",
        kernel_type=KernelType.REDUCTION,
        family="softmax_recompute",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_softmax_recompute(M, N, BM, BN, batch, heads),
        ttir_symbol="softmax_recompute_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[5, 6, 7]),  
            TensorSpec("sum_out", "input", ["batch", "heads", "M", "1"], "f32",
                      arg_index=2, stride_indices=[11, 12]),  
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16",
                      arg_index=1, stride_indices=[8, 9, 10]),  
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),  
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 3, "N": 4},
        grid_direction="0",  # N-grid并行
        description="Softmax recompute with M-grid parallelism"
    ))
    
    # 10. Softmax_recompute_ngrid (N维度并行，重计算版本)
    register_kernel(KernelMetadata(
        name="softmax_recompute_ngrid",
        kernel_type=KernelType.REDUCTION,
        family="softmax_recompute",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_softmax_recompute_ngrid(M, N, BM, BN, batch, heads),
        ttir_symbol="softmax_recompute_ngrid_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[5, 6, 7]),  
            TensorSpec("sum_out", "input", ["batch", "heads", "M", "1"], "f32",
                      arg_index=2, stride_indices=[11, 12]),  
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16",
                      arg_index=1, stride_indices=[8, 9, 10]),  
        ],
        grid_template=("cdiv(N, BN)", "batch*heads", "1"),  
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 3, "N": 4},
        grid_direction="1",  # N-grid并行
        description="Softmax recompute with N-grid parallelism"
    ))
    
    # 11. Sum_h2o (计算H2O score: sum over M dimension)
    register_kernel(KernelMetadata(
        name="sum_h2o",
        kernel_type=KernelType.REDUCTION,
        family="sum",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_sum(M, N, BM, BN, batch, heads),
        ttir_symbol="sum_h2o_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("output", "output", ["batch", "heads", "N"], "f16",
                      arg_index=1, stride_indices=[7, 8]),  # 输出是3D，只有2个stride
        ],
        grid_template=("cdiv(N, BN)", "batch*heads", "1"),  
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 2, "N": 3},
        grid_direction="1",  # N-grid并行
        description="Sum over M dimension for H2O scoring"
    ))
    
    # 11b. Mask_any (计算Corm score: any(probs >= mask) over M dimension)
    register_kernel(KernelMetadata(
        name="mask_any",
        kernel_type=KernelType.REDUCTION,
        family="mask_any",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_mask_any(M, N, BM, BN, batch, heads),
        ttir_symbol="mask_any_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[5, 6, 7,]),  # x_ptr + 4 strides
            TensorSpec("mask", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=1, stride_indices=[8, 9, 10]),  # y_ptr + 4 strides
            TensorSpec("output", "output", ["batch", "heads", "N"], "i8",
                      arg_index=2, stride_indices=[11, 12]),  # z_ptr + 3 strides
        ],
        grid_template=("cdiv(N, BN)", "batch*heads", "1"),  
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 3, "N": 4},
        grid_direction="1",  # N-grid并行
        description="Compare x >= y and any over M dimension for Corm scoring"
    ))
    
    # 12. ReLU (Element-wise activation)
    register_kernel(KernelMetadata(
        name="relu",
        kernel_type=KernelType.ELEMENTWISE,
        family="relu",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_relu(M, N, BM, BN, batch, heads),
        ttir_symbol="relu_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16",
                      arg_index=1, stride_indices=[7, 8, 9,]),
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 2, "N": 3},
        grid_direction="0",  # M-grid并行
        description="ReLU activation: max(0, x)"
    ))
    
    # 13. Mul (Element-wise multiplication)
    register_kernel(KernelMetadata(
        name="mul",
        kernel_type=KernelType.ELEMENTWISE,
        family="mul",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_mul(M, N, BM, BN, batch, heads),
        ttir_symbol="mul_kernel",
        tensors=[
            TensorSpec("x", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[5, 6, 7]),
            TensorSpec("y", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=1, stride_indices=[8, 9, 10]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16",
                      arg_index=2, stride_indices=[11, 12, 13]),
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 3, "N": 4},
        grid_direction="0",  # M-grid并行
        description="Element-wise multiplication: x * y"
    ))

    # 14. Sum over heads (Indexer)
    register_kernel(KernelMetadata(
        name="sum_dim1",
        kernel_type=KernelType.REDUCTION,
        family="sum_dim1",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_sum_dim1(M, N, BM, BN, batch, heads),
        ttir_symbol="sum_dim1_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("sum_out", "output", ["batch", "M", "N"], "f16",
                      arg_index=1, stride_indices=[7, 8,]),
        ],
        grid_template=("cdiv(M, BM)", "batch", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 2, "N": 3},
        description="Sum over heads dimension"
    ))

    # 15. Broadcast [B,M,N] -> [B,H,M,N]
    register_kernel(KernelMetadata(
        name="broadcast_bmn_to_bhmn",
        kernel_type=KernelType.ELEMENTWISE,
        family="broadcast",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_broadcast(M, N, BM, BN, batch, heads),
        ttir_symbol="bmh_broadcast_kernel",
        tensors=[
            TensorSpec("src", "input", ["batch", "M", "N"], "f16",
                      arg_index=0, stride_indices=[5, 6]),
            TensorSpec("dst", "output", ["batch", "heads", "M", "N"], "f16",
                      arg_index=1, stride_indices=[7, 8, 9]),
        ],
        grid_template=("cdiv(M, BM)", "batch", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"heads": 2, "M": 3, "N": 4},
        description="Broadcast [B,M,N] tensor to [B,H,M,N]"
    ))

    # 16. TopK over last dim
    register_kernel(KernelMetadata(
        name="topk_bhmn",
        kernel_type=KernelType.CUSTOM,
        family="topk",
        ttir_generator=lambda M, N, K, BM, BN, batch, heads: _ttir_of_topk_bhmn(M, N, K, BM, BN, batch, heads),
        ttir_symbol="topk_forward_bhmn",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[6, 7, 8]),
            TensorSpec("topk_vals", "output", ["batch", "heads", "M", "K"], "f16",
                      arg_index=1, stride_indices=[9, 10, 11]),
            TensorSpec("topk_idx", "output", ["batch", "heads", "M", "K"], "i32",
                      arg_index=2, stride_indices=[]),
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "K", "BM", "BN", "batch", "heads"],
        dim_arg_indices={"heads": 3, "M": 4, "N": 5},
        description="TopK selection along N dimension"
    ))

    # 17. Scatter mask update
    register_kernel(KernelMetadata(
        name="scatter_mask",
        kernel_type=KernelType.CUSTOM,
        family="scatter",
        ttir_generator=lambda M, N, K, BM, BN, batch, heads: _ttir_of_scatter(M, N, K, BM, BN, batch, heads),
        ttir_symbol="scatter4d_kernel_scalar",
        tensors=[
            TensorSpec("mask_tensor", "input_output", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[6, 7, 8]),
            TensorSpec("indices", "input", ["batch", "heads", "M", "K"], "i32",
                      arg_index=1, stride_indices=[9, 10, 11]),
        ],
        grid_template=("cdiv(M, BM)", "batch*heads", "1"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "K", "batch", "heads"],
        dim_arg_indices={"heads":2, "M": 3, "N": 4, "K": 5},
        description="Scatter scalar into mask tensor along last dim"
    ))

    # ========================================================================
    # DSA 专用 kernels (Sparse MLA / Dynamic Sparse Attention)
    # Grid: (Batch, M, Head_blocks) - 即 (batch, M, cdiv(heads, BLOCK_H))
    # ========================================================================
    from tilefusion.ops.gather_matmul_qk import _ttir_of_gather_mm_qk
    from tilefusion.ops.gather_matmul_sv import _ttir_of_gather_mm_sv
    from tilefusion.ops.scale_softmax_mgrid import _ttir_of_scale_mgrid, _ttir_of_add_mask, _ttir_of_softmax_mgrid, _ttir_of_softmax_reduce_mgrid, _ttir_of_softmax_recompute_mgrid

    # 18. Gather GEMM QK (Q @ K[indices] with gather)
    register_kernel(KernelMetadata(
        name="gather_gemm_qk_mgrid",
        kernel_type=KernelType.GEMM,
        family="gather_gemm_loopn",
        ttir_generator=lambda M, N, TopK, K, BM, BN, BLOCK_D1, BLOCK_D2, batch, heads: _ttir_of_gather_mm_qk(M, N, TopK, K, BM, BN, BLOCK_D1, BLOCK_D2, batch, heads),
        ttir_symbol="gather_gemm_dot_kernel",
        tensors=[
            TensorSpec("Q", "input", ["batch", "heads", "M", "K"], "f16",
                      arg_index=0, stride_indices=[6, 7, 8,]),
            TensorSpec("K", "input", ["batch", "1", "N", "K"], "f16",
                      arg_index=1, stride_indices=[9, 10, 11,]),
            TensorSpec("Indices", "input", ["batch", "1", "M", "TopK"], "i32",
                      arg_index=2, stride_indices=[12, 13, 14]),
            TensorSpec("scores", "output", ["batch", "heads", "M", "TopK"], "f16",
                      arg_index=3, stride_indices=[15, 16, 17]),
        ],
        grid_template=("batch", "M", "cdiv(heads, BM)"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "TopK", "K", "BLOCK_D1", "BLOCK_D2", "batch", "heads"],
        dim_arg_indices={"heads": 4, "TopK": 5},
        grid_direction="0",
        description="Gather GEMM: Q @ K[indices] for sparse attention"
    ))

    # 19. Scale kernel with mgrid layout (Batch, M, Head_blocks)
    register_kernel(KernelMetadata(
        name="scale_mgrid",
        kernel_type=KernelType.ELEMENTWISE,
        family="scale",
        ttir_generator=lambda M, TopK, BM, BN, batch, heads, scale: _ttir_of_scale_mgrid(M, TopK, BM, BN, batch, heads, scale),
        ttir_symbol="scale_kernel_mgrid",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "TopK"], "f16",
                      arg_index=0, stride_indices=[4, 5, 6,]),
            TensorSpec("output", "output", ["batch", "heads", "M", "TopK"], "f16",
                      arg_index=1, stride_indices=[7, 8, 9,]),
        ],
        grid_template=("batch", "M", "cdiv(heads, BM)"),
        block_params=["BM", "BN"],
        required_params=["M", "TopK", "batch", "heads", "scale"],
        dim_arg_indices={"heads": 2, "TopK": 3},
        grid_direction="0",
        description="Scale kernel with mgrid (B, M, H_blocks) layout"
    ))

    # 20. Add mask kernel with mgrid layout
    register_kernel(KernelMetadata(
        name="add_mask_mgrid",
        kernel_type=KernelType.ELEMENTWISE,
        family="add_mask",
        ttir_generator=lambda M, TopK, BM, BN, batch, heads: _ttir_of_add_mask(M, TopK, BM, BN, batch, heads),
        ttir_symbol="add_mask_kernel_mgrid",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "TopK"], "f16",
                      arg_index=0, stride_indices=[5, 6, 7]),
            TensorSpec("mask", "input", ["batch", "heads", "M", "TopK"], "f16",
                      arg_index=1, stride_indices=[8, 9, 10]),
            TensorSpec("output", "output", ["batch", "heads", "M", "TopK"], "f16",
                      arg_index=2, stride_indices=[11, 12, 13]),
        ],
        grid_template=("batch", "M", "cdiv(heads, BM)"),
        block_params=["BM", "BN"],
        required_params=["M", "TopK", "batch", "heads"],
        dim_arg_indices={"heads": 3, "TopK": 4},
        grid_direction="0",
        description="Add mask kernel with mgrid (B, M, H_blocks) layout"
    ))

    # 21. Softmax kernel with mgrid layout
    register_kernel(KernelMetadata(
        name="softmax_mgrid",
        kernel_type=KernelType.REDUCTION,
        family="softmax",
        ttir_generator=lambda M, TopK, BM, BN, batch, heads: _ttir_of_softmax_mgrid(M, TopK, BM, BN, batch, heads),
        ttir_symbol="softmax_kernel_mgrid",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "TopK"], "f16",
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("output", "output", ["batch", "heads", "M", "TopK"], "f16",
                      arg_index=1, stride_indices=[7, 8, 9]),
        ],
        grid_template=("batch", "M", "cdiv(heads, BM)"),
        block_params=["BM", "BN"],
        required_params=["M", "TopK", "batch", "heads"],
        dim_arg_indices={"heads": 2, "TopK": 3},
        grid_direction="0",
        description="Softmax kernel with mgrid (B, M, H_blocks) layout"
    ))
    
    # 6b. Softmax reduce (只计算第一个loop，输出sum用于recompute)
    register_kernel(KernelMetadata(
        name="softmax_reduce_topk",
        kernel_type=KernelType.REDUCTION,
        family="softmax_reduce_topk",
        ttir_generator=lambda M, TopK, BM, BN, batch, heads: _ttir_of_softmax_reduce_mgrid(M, TopK, BM, BN, batch, heads),
        ttir_symbol="softmax_reduce_mgrid_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "TopK"], "f16", 
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("sum_out", "output", ["batch", "heads", "M", "1"], "f32",
                      arg_index=1, stride_indices=[7, 8]),
        ],
        grid_template=("batch", "M", "cdiv(heads, BM)"),
        block_params=["BM", "BN"],
        required_params=["M", "TopK", "batch", "heads"],
        dim_arg_indices={"heads": 2, "TopK": 3},
        grid_direction="0",
        description="Softmax reduce (first pass only, outputs sum for recompute)"
    ))
    
    # 6b. Softmax reduce (只计算第一个loop，输出sum用于recompute)
    register_kernel(KernelMetadata(
        name="softmax_recompute_mgrid_kernel",
        kernel_type=KernelType.REDUCTION,
        family="softmax_recompute_mgrid_kernel",
        ttir_generator=lambda M, TopK, BM, BN, batch, heads: _ttir_of_softmax_recompute_mgrid(M, TopK, BM, BN, batch, heads),
        ttir_symbol="softmax_recompute_mgrid_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "TopK"], "f16",
                      arg_index=0, stride_indices=[5, 6, 7]),  
            TensorSpec("sum_out", "input", ["batch", "heads", "M", "1"], "f32",
                      arg_index=2, stride_indices=[11, 12]),  
            TensorSpec("output", "output", ["batch", "heads", "M", "TopK"], "f16",
                      arg_index=1, stride_indices=[8, 9, 10]),  
        ],
        grid_template=("batch", "M", "cdiv(heads, BM)"), 
        block_params=["BM", "BN"],
        required_params=["M", "TopK", "batch", "heads"],
        dim_arg_indices={"M": 3, "TopK": 4},
        grid_direction="0",  # N-grid并行
        description="Softmax recompute with N-grid parallelism"
    ))

    # 22. Gather GEMM SV (Scores @ V[indices] with gather)
    register_kernel(KernelMetadata(
        name="gather_gemm_sv_mgrid",
        kernel_type=KernelType.GEMM,
        family="gather_gemm_loopk",
        ttir_generator=lambda M, N, TopK, D, BM, BN, BLOCK_D1, batch, heads: _ttir_of_gather_mm_sv(M, N, TopK, D, BM, BN, BLOCK_D1, batch, heads),
        ttir_symbol="gather_gemm_score_v_stacked_kernel",
        tensors=[
            TensorSpec("Scores", "input", ["batch", "heads", "M", "TopK"], "f16",
                      arg_index=0, stride_indices=[6, 7, 8]),
            TensorSpec("V", "input", ["batch", "1", "N", "D"], "f16",
                      arg_index=1, stride_indices=[9, 10, 11]),
            TensorSpec("Indices", "input", ["batch", "1", "M", "TopK"], "i32",
                      arg_index=2, stride_indices=[12, 13, 14]),
            TensorSpec("output", "output", ["batch", "heads", "M", "D"], "f16",
                      arg_index=3, stride_indices=[15, 16, 17]),
        ],
        grid_template=("batch", "M", "cdiv(heads, BM)"),
        block_params=["BM", "BN"],
        required_params=["M", "N", "TopK", "D", "BLOCK_D1", "batch", "heads"],
        dim_arg_indices={"heads": 4, "TopK": 5},
        grid_direction="0",
        description="Gather GEMM: Scores @ V[indices] for sparse attention"
    ))

    # 23. Sum_and_Square (计算ROCO score: sum and sum_sq over M dimension)
    from tilefusion.ops.sum import _ttir_of_sum_and_square
    register_kernel(KernelMetadata(
        name="sum_and_square",
        kernel_type=KernelType.REDUCTION,
        family="sum_and_square",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_sum_and_square(M, N, BM, BN, batch, heads),
        ttir_symbol="sum_and_square_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[5, 6, 7]),
            TensorSpec("output_sum", "output", ["batch", "heads", "N"], "f16",
                      arg_index=1, stride_indices=[8, 9]),
            TensorSpec("output_sq", "output", ["batch", "heads", "N"], "f16",
                      arg_index=2, stride_indices=[10, 11]),
        ],
        grid_template=("cdiv(N, BN)", "batch*heads", "1"),  
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 3, "N": 4},
        grid_direction="1",  # N-grid并行
        description="Sum and Sum-Square over M dimension for ROCO scoring"
    ))

    # 24. Avg_Pool1d (1D average pooling for SnapKV)
    from tilefusion.ops.avg_pool1d import _ttir_of_avg_pool1d
    register_kernel(KernelMetadata(
        name="avg_pool1d",
        kernel_type=KernelType.REDUCTION,
        family="avg_pool1d",
        ttir_generator=lambda N, BN, kernel_size, padding, batch, heads: _ttir_of_avg_pool1d(N, BN, kernel_size, padding, batch, heads),
        ttir_symbol="avg_pool1d_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "N"], "f16",
                      arg_index=0, stride_indices=[3, 4]),
            TensorSpec("output", "output", ["batch", "heads", "N"], "f16",
                      arg_index=1, stride_indices=[5, 6]),
        ],
        grid_template=("cdiv(N, BN)", "batch*heads", "1"),
        block_params=["BN"],
        required_params=["N", "batch", "heads", "kernel_size", "padding"],
        dim_arg_indices={"N": 2},
        grid_direction="1",
        description="1D Average Pooling for SnapKV score"
    ))

    # 25. Log_Neg_Ngrid (N-grid log-neg for Keyformer)
    from tilefusion.ops.elementwize_ngrid_loopm import _ttir_of_log_neg_ngrid
    register_kernel(KernelMetadata(
        name="log_neg_ngrid",
        kernel_type=KernelType.ELEMENTWISE,
        family="log_neg",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_log_neg_ngrid(M, N, BM, BN, batch, heads),
        ttir_symbol="log_neg_ngrid_kernel",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16",
                      arg_index=1, stride_indices=[7, 8, 9]),
        ],
        grid_template=("cdiv(N, BN)", "batch*heads", "1"),  # N-grid
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 2, "N": 3},
        grid_direction="1",
        description="N-grid Log-Neg for Keyformer Gumbel noise"
    ))

    # 26. Add_Ngrid (N-grid element-wise add for Keyformer)
    from tilefusion.ops.elementwize_ngrid_loopm import _ttir_of_add_ngrid
    register_kernel(KernelMetadata(
        name="add_ngrid",
        kernel_type=KernelType.ELEMENTWISE,
        family="add",
        ttir_generator=lambda M, N, BM, BN, batch, heads: _ttir_of_add_ngrid(M, N, BM, BN, batch, heads),
        ttir_symbol="add_ngrid_kernel",
        tensors=[
            TensorSpec("input1", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[3, 4, 5]),
            TensorSpec("input2", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=1, stride_indices=[6, 7, 8]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16",
                      arg_index=2, stride_indices=[9, 10, 11]),
        ],
        grid_template=("cdiv(N, BN)", "batch*heads", "1"),  # N-grid
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads"],
        dim_arg_indices={"M": 12, "N": 13},
        grid_direction="1",
        description="N-grid element-wise addition"
    ))

    # 27. Scale_C_Ngrid (N-grid scale with constant for Keyformer)
    from tilefusion.ops.elementwize_ngrid_loopm import _ttir_of_scale_ngrid_c
    register_kernel(KernelMetadata(
        name="scale_c_ngrid",
        kernel_type=KernelType.ELEMENTWISE,
        family="scale",
        ttir_generator=lambda M, N, BM, BN, scale, batch, heads: _ttir_of_scale_ngrid_c(M, N, BM, BN, scale, batch, heads),
        ttir_symbol="scale_ngrid_kernel_c",
        tensors=[
            TensorSpec("input", "input", ["batch", "heads", "M", "N"], "f16",
                      arg_index=0, stride_indices=[4, 5, 6]),
            TensorSpec("output", "output", ["batch", "heads", "M", "N"], "f16",
                      arg_index=1, stride_indices=[7, 8, 9]),
        ],
        grid_template=("cdiv(N, BN)", "batch*heads", "1"),  # N-grid
        block_params=["BM", "BN"],
        required_params=["M", "N", "batch", "heads", "scale"],
        dim_arg_indices={"M": 2, "N": 3},
        grid_direction="1",
        description="N-grid scale with constant (e.g., 1/tau)"
    ))


# 自动注册
register_standard_kernels()
