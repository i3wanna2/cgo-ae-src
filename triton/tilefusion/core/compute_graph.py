import threading
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
import tempfile
import torch
import triton
from triton.compiler import compile as triton_compile
import time
import copy
import itertools
import gc
from tilefusion.core.kernel_registry import get_registry, KernelMetadata
from tilefusion.core.compiler import fuse_kernels_in_ttir, opt_kernels_in_ttir
from tilefusion.utils.utils import build_combined_module
from tilefusion.core.auto_mapper import AutoMapper
from tilefusion.core.autotuner import AutoTuner

# Dtype mapping for automatic tensor creation
dtype_map = {
    'f16': torch.float16,
    'f32': torch.float32,
    'f64': torch.float64,
    'i32': torch.int32,
    'i64': torch.int64,
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
    'int32': torch.int32,
    'int64': torch.int64,
}

_node_id_counter = 0

def _generate_node_id(prefix: str = "node") -> str:
    global _node_id_counter
    _node_id_counter += 1
    return f"{prefix}_{_node_id_counter}"


@dataclass
class InputNode:
    name: str 
    
    @property
    def id(self) -> str:
        return f"input_{self.name}"
    
    def __repr__(self):
        return f"InputNode({self.name})"


@dataclass
class OutputNode:
    """输出节点，用于标记图的输出并预分配 tensor"""
    name: str  # 输出名称
    source_idx: int  # 源计算节点的索引
    shape: Optional[Tuple[int, ...]] = None  # tensor 形状
    dtype: torch.dtype = torch.float16  # tensor 数据类型
    tensor: Optional[torch.Tensor] = None  # 预分配的输出 tensor
    output_index: int = 0  # 源节点的输出索引（用于多输出 kernel，0 表示第一个输出）
    
    @property
    def id(self) -> str:
        return f"output_{self.name}"
    
    def __repr__(self):
        return f"OutputNode({self.name}, source={self.source_idx}, output_index={self.output_index})"


@dataclass
class ComputeNode:
    kernel_name: str 
    params: Dict[str, Any]  
    metadata: Optional[KernelMetadata] = None 
    input_mappings: Optional[Dict[str, str]] = None  
    parent_ids: Optional[List[str]] = None  
    depth: int = 0  
    id: Optional[str] = None 
    
    def __post_init__(self):
        if self.id is None:
            self.id = _generate_node_id(self.kernel_name)
    
    def __repr__(self):
        return f"ComputeNode({self.kernel_name}, depth={self.depth}, id={self.id})"


class ComputeGraph:
    def __init__(self, name: str = "compute_graph"):
        self.name = name
        self.nodes: List[Any] = [] 
        self.registry = get_registry()
        self.auto_mapper = AutoMapper()  
        self.enable_causal_opt: bool = True
        self._id_to_node: Dict[str, Any] = {} 
        self._id_to_index: Dict[str, int] = {} 
        self._output_node_ids: Dict[str, str] = {}  # output_name -> node_id 的映射 (旧方式，兼容)
        self._outputs: Dict[str, OutputNode] = {}  # output_name -> OutputNode 的映射
        # 进程内缓存：MLIR pipeline 对相同 params 只跑一次，triton_compile 在锁外并发
        self._ttir_cache: Dict[tuple, str] = {}          # params_key -> fused_path
        self._compiled_cache: Dict[tuple, Any] = {}      # (params_key, warps, stages) -> (compiled, grid)
        self._compile_lock = threading.Lock()
    
    def _get_node_by_id(self, node_id: str) -> Optional[Any]:
        if node_id in self._id_to_node:
            return self._id_to_node[node_id]
        for node in self.nodes:
            if node.id == node_id:
                self._id_to_node[node_id] = node
                return node
        return None
    
    def _get_index_by_id(self, node_id: str) -> int:
        if node_id in self._id_to_index:
            return self._id_to_index[node_id]
        for idx, node in enumerate(self.nodes):
            if node.id == node_id:
                self._id_to_index[node_id] = idx
                return idx
        return -1
    
    def _update_id_cache(self, node: Any, idx: int):
        self._id_to_node[node.id] = node
        self._id_to_index[node.id] = idx
    
    def add_input(self, *tensor_names: str) -> 'ComputeGraph':
        for name in tensor_names:
            node_idx = len(self.nodes)
            node = InputNode(name)
            self.nodes.append(node)
            self._update_id_cache(node, node_idx)
            print(f"  Added InputNode[{node_idx}]: {name}")
        return self
    
    def add_node(self, kernel_name: str, inputs: Optional[Dict[str, str]] = None,
                 parents: Optional[List[int]] = None, output: Optional[str] = None, **params) -> 'ComputeGraph':
        metadata = self.registry.get(kernel_name)
        if metadata is None:
            available = self.registry.list_all()
            raise ValueError(
                f"Kernel '{kernel_name}' not found in registry.\n"
                f"Available kernels: {available}"
            )
        
        # 验证父节点索引
        if parents is None:
            raise ValueError(
                f"Must specify 'parents' for node '{kernel_name}'.\n"
                f"Current nodes: {[self._node_repr(i) for i in range(len(self.nodes))]}\n"
                f"Example: parents=[0, 1] to depend on nodes 0 and 1"
            )
        
        for parent_idx in parents:
            if parent_idx < 0 or parent_idx >= len(self.nodes):
                raise ValueError(
                    f"Invalid parent index {parent_idx} for node '{kernel_name}'. "
                    f"Valid range: [0, {len(self.nodes)-1}]"
                )
  
        depth = 0
        parent_ids: List[str] = []
        if parents:
            for parent_idx in parents:
                parent = self.nodes[parent_idx]
                parent_ids.append(parent.id)
                if isinstance(parent, ComputeNode):
                    depth = max(depth, parent.depth + 1)
                else: 
                    depth = max(depth, 1)
        
        node_idx = len(self.nodes)
        node = ComputeNode(
            kernel_name=kernel_name,
            params=params,
            metadata=metadata,
            input_mappings=inputs,
            parent_ids=parent_ids,
            depth=depth,
        )
        self.nodes.append(node)
        self._update_id_cache(node, node_idx)
        
        if output:
            self._output_node_ids[output] = node.id
        
        parent_repr = ", ".join([self._node_repr(p) for p in parents])
        output_marker = f", output='{output}'" if output else ""
        print(f"  Added ComputeNode[{node_idx}]: {kernel_name}, parents=[{parent_repr}]{output_marker}")
        return self
    
    def _node_repr(self, idx: int) -> str:
        """返回节点的简短表示"""
        if idx < 0 or idx >= len(self.nodes):
            return f"Invalid({idx})"
        node = self.nodes[idx]
        if isinstance(node, InputNode):
            return f"{idx}:{node.name}"
        else:
            return f"{idx}:{node.kernel_name}"
    
    def add_output(self, name: str, source: int, shape: Tuple[int, ...], 
                   dtype: torch.dtype = torch.float16, device: str = 'cuda',
                   output_index: int = 0) -> 'ComputeGraph':
        """
        添加输出节点，预分配输出 tensor
        
        Args:
            name: 输出名称
            source: 源计算节点的索引
            shape: 输出 tensor 的形状
            dtype: 数据类型，默认 float16
            device: 设备，默认 'cuda'
            output_index: 源节点的输出索引（用于多输出 kernel，默认 0 表示第一个输出）
                        例如 sum_and_square 有两个输出，output_index=0 是 output_sum，output_index=1 是 output_sq
        
        Returns:
            self (支持链式调用)
        
        示例:
            graph.add_output("attn_output", source=8, shape=(batch, heads, M, K))
            graph.add_output("h2o_score", source=9, shape=(batch, heads, N))
            # 对于多输出 kernel:
            graph.add_output("sum", source=5, shape=(batch, N), output_index=0)  # 第一个输出
            graph.add_output("sq", source=5, shape=(batch, N), output_index=1)   # 第二个输出
        """
        if source < 0 or source >= len(self.nodes):
            raise ValueError(f"Invalid source index {source}. Valid range: [0, {len(self.nodes)-1}]")
        
        source_node = self.nodes[source]
        if not isinstance(source_node, ComputeNode):
            raise ValueError(f"Source node at index {source} must be a ComputeNode, got {type(source_node).__name__}")
        
        # 预分配 tensor
        tensor = torch.empty(shape, dtype=dtype, device=device)
        
        output_node = OutputNode(
            name=name,
            source_idx=source,
            shape=shape,
            dtype=dtype,
            tensor=tensor,
            output_index=output_index  # 保存输出索引
        )
        
        self._outputs[name] = output_node
        # 同时记录 source 节点的 kernel_name 用于 tensor key 匹配
        self._output_node_ids[name] = source_node.kernel_name
        
        # 获取该输出索引对应的 tensor 名称用于打印
        output_tensors = [t for t in source_node.metadata.tensors if t.role == 'output']
        tensor_name = output_tensors[output_index].name if output_index < len(output_tensors) else f"output[{output_index}]"
        print(f"  Added OutputNode: '{name}' <- [{source}:{source_node.kernel_name}].{tensor_name}, shape={shape}")
        return self
    
    @property
    def outputs(self) -> Dict[str, torch.Tensor]:
        """获取所有预分配的输出 tensor"""
        return {name: node.tensor for name, node in self._outputs.items() if node.tensor is not None}
    
    def get_output_tensors(self, intermediate_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        从 intermediate_tensors 中提取已标记的输出 tensor
        优先返回预分配的 tensor（如果存在）
        """
        # 优先返回预分配的输出
        if self._outputs:
            return self.outputs
        
        # 回退到旧的搜索方式
        outputs = {}
        for output_name, kernel_name in self._output_node_ids.items():
            for key, tensor in intermediate_tensors.items():
                if f"_{kernel_name}_" in f"_{key}" or key.startswith(f"{kernel_name}_"):
                    if '_output' in key:
                        outputs[output_name] = tensor
                        break
            if output_name not in outputs:
                for key, tensor in intermediate_tensors.items():
                    if f"_{kernel_name}_" in f"_{key}" or key.startswith(f"{kernel_name}_"):
                        outputs[output_name] = tensor
                        break
        return outputs
    
    def get_fusion_plan(self, nodes: Optional[List[ComputeNode]] = None, verbose: bool = True) -> List[Tuple[str, Any, List[int], List[int]]]:
        """
        生成融合计划 - 按顺序线性融合
        
        通用函数：支持对整个图或任意节点切片生成融合计划。
        每个节点（除了第一个）都与前一个节点融合，不管它的 parent_ids 是什么。
        
        Args:
            nodes: 要生成融合计划的节点列表。如果为None，使用整个图的所有计算节点。
            verbose: 是否打印调试信息
        
        Returns:
            List of (consumer_kernel_name, ttir_generator, prod_out_idx, cons_in_idx)
        """
        # 确定要处理的节点
        if nodes is None:
            raise ValueError("Must provide 'nodes' parameter to get_fusion_plan")
        else:
            # 使用指定的节点切片，按顺序建立索引
            compute_nodes = [(i, node) for i, node in enumerate(nodes) if isinstance(node, ComputeNode)]
        
        if len(compute_nodes) < 2:
            return []
        
        fusion_stages = []
        
        # 重置mapper
        self.auto_mapper.reset()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🚀 Auto-inferring Fusion Mappings (Linear)")
            print(f"{'='*70}")
        
        # 初始化：第一个计算节点的参数总数
        _, first_compute_node = compute_nodes[0]
        self.auto_mapper.accumulated_param_count = first_compute_node.metadata.get_total_param_count()
        
        # 用于缓存已融合节点的元数据
        fused_metadata_cache = {}  # {node_index: fused_metadata}
        fused_metadata_cache[compute_nodes[0][0]] = first_compute_node.metadata
        
        # 映射 InputNode 名称 -> (tensor arg_index in fused meta)
        # 用于在后续节点引用相同 InputNode 时建立 stride 映射
        input_node_arg_mapping: Dict[str, int] = {}

        # 获取InputNode的名称集合（从原始nodes或self.nodes）
        source_nodes = nodes
        input_node_names = set()
        for node in source_nodes:
            if isinstance(node, InputNode):
                input_node_names.add(node.name)
        
        # 初始化第一个节点的 InputNode -> arg_index 映射
        # 遍历第一个节点的 parent_ids，按顺序对应其输入 tensor
        first_inputs = sorted(
            [t for t in first_compute_node.metadata.tensors if t.role == 'input'],
            key=lambda t: t.arg_index
        )
        first_parent_ids = first_compute_node.parent_ids or []
        for i, parent_id in enumerate(first_parent_ids):
            if parent_id.startswith("input_") and i < len(first_inputs):
                input_name = parent_id[6:]  # 去掉 "input_" 前缀
                input_node_arg_mapping[input_name] = first_inputs[i].arg_index
        
        if verbose:
            print(f"\n📍 Node[{compute_nodes[0][0]}] '{first_compute_node.kernel_name}': First node (base)")
        
        # 为每个节点索引记录其在融合后 metadata 中的输出 tensor arg_index
        # 这样当后续节点需要引用某个前驱节点的输出时，可以快速查找
        node_output_arg_indices: Dict[int, int] = {}
        
        # 第一个节点的输出 tensor arg_index
        first_outputs = first_compute_node.metadata.get_output_tensors()
        if first_outputs:
            node_output_arg_indices[compute_nodes[0][0]] = first_outputs[0].arg_index
        
        # 按顺序处理每个计算节点（从第二个开始，每个都与前一个融合）
        for i in range(1, len(compute_nodes)):
            consumer_idx, consumer_node = compute_nodes[i]
            producer_idx, producer_node = compute_nodes[i - 1]  # 前一个节点
            
            # 获取producer的元数据（可能已经是融合后的）
            producer_meta = fused_metadata_cache.get(producer_idx, producer_node.metadata)
            consumer_meta = consumer_node.metadata
            
            # 获取consumer的输入tensor（按 arg_index 排序）
            cons_inputs = sorted(
                [t for t in consumer_meta.tensors if t.role == 'input'],
                key=lambda t: t.arg_index
            )
            
            # 基于 parent_ids 顺序构建 tensor_correspondence
            # parent_ids 的顺序与 cons_inputs 的顺序一一对应
            tensor_correspondence = {}
            
            consumer_parent_ids = consumer_node.parent_ids or []
            
            # 构建当前 partition 中的 ID -> index 映射
            local_id_to_idx = {node.id: idx for idx, node in compute_nodes}
            
            # 用于记录本次融合新引入的 InputNode（需要在融合后添加到参数列表）
            new_input_nodes_for_fusion: List[Tuple[str, int]] = []  # [(input_name, cons_arg_idx)]
            
            for input_idx, parent_id in enumerate(consumer_parent_ids):
                if input_idx >= len(cons_inputs):
                    break
                
                cons_arg_idx = cons_inputs[input_idx].arg_index
                
                # 检查是否是 InputNode
                if parent_id.startswith("input_"):
                    input_name = parent_id[6:]  # 去掉 "input_" 前缀
                    # 检查这个 InputNode 是否在之前的节点中被使用过
                    if input_name in input_node_arg_mapping:
                        prod_arg_idx = input_node_arg_mapping[input_name]
                        tensor_correspondence[prod_arg_idx] = cons_arg_idx
                    else:
                        # 这是一个新的 InputNode，需要添加到融合后的参数列表
                        # 暂时记录下来，稍后处理
                        new_input_nodes_for_fusion.append((input_name, cons_arg_idx))
                    continue
                
                # 尝试找到这个 parent 在本地 partition 中的索引
                parent_idx = local_id_to_idx.get(parent_id)
                
                if parent_idx is not None and parent_idx in node_output_arg_indices:
                    # 找到这个 parent 节点的输出 tensor 在融合后 metadata 中的 arg_index
                    prod_arg_idx = node_output_arg_indices[parent_idx]
                    tensor_correspondence[prod_arg_idx] = cons_arg_idx
                elif parent_idx is not None:
                    # 可能是更早的节点，尝试从 fused_metadata_cache 查找
                    if parent_idx in fused_metadata_cache:
                        parent_meta = fused_metadata_cache[parent_idx]
                    else:
                        # 查找原始节点（通过 ID）
                        parent_node = self._get_node_by_id(parent_id)
                        if parent_node and isinstance(parent_node, ComputeNode):
                            parent_meta = parent_node.metadata
                        else:
                            continue
                    
                    parent_outputs = parent_meta.get_output_tensors()
                    if parent_outputs:
                        prod_arg_idx = parent_outputs[0].arg_index
                        tensor_correspondence[prod_arg_idx] = cons_arg_idx

            # 处理新引入的 InputNode
            # 这些 InputNode 需要作为新参数添加到融合后的 kernel 中
            # 为它们分配新的 arg_index（基于当前 accumulated_param_count）
            for input_name, cons_arg_idx in new_input_nodes_for_fusion:
                # 为这个新 InputNode 分配一个融合后的 arg_index
                # 注意：这里我们不直接修改 tensor_correspondence，而是将其标记为"新参数"
                # infer_mapping 会处理这种情况
                # 但我们需要更新 input_node_arg_mapping 以便后续节点可以引用
                new_arg_idx = self.auto_mapper.accumulated_param_count
                # 暂不添加到 tensor_correspondence，因为这不是从 producer 映射过来的
                # 而是需要作为新参数添加
                # 更新 input_node_arg_mapping 为后续节点准备
                # 注意：这里的 new_arg_idx 会在 update_for_next_fusion 后变化
                # 所以我们需要在融合完成后再更新
                pass  # 稍后在融合完成后更新
            
            # 自动推断映射
            prod_idx, cons_idx = self.auto_mapper.infer_mapping(
                producer_meta,
                consumer_meta,
                tensor_correspondence=tensor_correspondence
            )
            # 构建融合后的元数据并缓存
            fused_meta = self.auto_mapper.build_fused_metadata(
                producer_meta,
                consumer_meta,
                tensor_correspondence
            )
            fused_metadata_cache[consumer_idx] = fused_meta
            
            # 更新新引入的 InputNode 的 arg_index 映射
            # 这些 InputNode 对应的参数位置 = 融合前的 accumulated_param_count + consumer 中的相对位置
            for input_name, cons_arg_idx in new_input_nodes_for_fusion:
                # consumer 中该 InputNode 对应的 tensor
                cons_tensor = cons_inputs[consumer_parent_ids.index(f"input_{input_name}")] if f"input_{input_name}" in consumer_parent_ids else None
                if cons_tensor:
                    # 计算融合后的 arg_index
                    # = accumulated_param_count (融合前) + cons_arg_idx - len(tensor_correspondence)
                    fused_arg_idx = self.auto_mapper.accumulated_param_count - (consumer_meta.get_total_param_count() - len(cons_idx)) + cons_arg_idx
                    input_node_arg_mapping[input_name] = fused_arg_idx
            
            # 更新当前节点的输出 arg_index（在融合后 metadata 中）
            fused_outputs = fused_meta.get_output_tensors()
            if fused_outputs:
                # 最后一个输出是当前 consumer 的输出
                node_output_arg_indices[consumer_idx] = fused_outputs[-1].arg_index
            
            # 更新mapper状态
            self.auto_mapper.update_for_next_fusion(consumer_node.metadata, cons_idx)
            
            fusion_stages.append((
                consumer_meta.name,
                lambda meta=consumer_meta, params=consumer_node.params: meta.create_ttir(**params),
                prod_idx,
                cons_idx
            ))
        
        return fusion_stages

    def _detect_flash_attention_pass(self, subgraphs: List[List[ComputeNode]]) -> List[bool]:
        patterns = []
        
        for i, subgraph in enumerate(subgraphs):
            kernel_names = [node.kernel_name for node in subgraph]
            kernel_set = set(kernel_names)
            
            # Flash Attention必需的kernel序列（可以有额外的kernel，但这些是核心）
            has_gemm_qk = any('gemm' in k and ('qk' in k or 'mgrid' in k) for k in kernel_names)
            has_scale = any('scale' in k for k in kernel_names)
            has_mask = any('mask' in k for k in kernel_names)
            has_softmax = any('softmax' in k for k in kernel_names)
            has_gemm_pv = any('gemm' in k and ('pv' in k or 'loopk' in k) for k in kernel_names)
            is_gemm_last = "gemm" in kernel_names[-1]
            # 检查是否符合Flash Attention的kernel序列
            is_flash_attn = (has_gemm_qk and has_scale and has_mask and 
                            has_softmax and has_gemm_pv and is_gemm_last)
            
            patterns.append(is_flash_attn)
     
        return patterns

    def _find_softmax_in_subgraph(self, subgraph: List[ComputeNode]) -> Tuple[Optional[ComputeNode], int]:
        """在子图中查找 softmax 节点，返回 (softmax_node, index)"""
        for idx, node in enumerate(subgraph):
            if 'softmax' in node.kernel_name and 'recompute' not in node.kernel_name:
                return node, idx
        return None, -1
    
    def _check_dependency_on_subgraph(
        self, 
        subgraph: List[ComputeNode], 
        target_subgraph: List[ComputeNode],
        target_node_filter: Optional[str] = None
    ) -> Tuple[bool, List[str], Optional[ComputeNode]]:
        """
        检查 subgraph 是否依赖于 target_subgraph 中的节点
        
        Args:
            subgraph: 当前子图
            target_subgraph: 目标子图
            target_node_filter: 可选，只检查包含此字符串的节点（如 'softmax'）
        
        Returns:
            (has_dependency, dependent_pairs, matched_target_node)
        """
        target_ids = {node.id: node for node in target_subgraph}
        dependent_pairs = []
        matched_target = None
        
        for node in subgraph:
            global_node = self._get_node_by_id(node.id) or node
            if global_node.parent_ids:
                for pid in global_node.parent_ids:
                    if pid in target_ids:
                        target_node = target_ids[pid]
                        if target_node_filter is None or target_node_filter in target_node.kernel_name:
                            dependent_pairs.append(f"{node.kernel_name} -> {target_node.kernel_name}")
                            matched_target = target_node
        
        return len(dependent_pairs) > 0, dependent_pairs, matched_target

    def check_recompute_dependency(
        self, 
        subgraph: List[ComputeNode], 
        prev_subgraph: List[ComputeNode],
        is_from_flash_attn: bool = True
    ) -> Tuple[bool, bool, int]:
        """
        统一的重计算依赖检查函数
        
        Args:
            subgraph: 当前子图
            prev_subgraph: 前驱子图
            is_from_flash_attn: 是否从 Flash Attention 子图重计算
        
        Returns:
            (needs_recompute, is_valid, softmax_idx)
            - needs_recompute: 是否需要重计算
            - is_valid: 是否有效（仅 Flash Attention 场景检查）
            - softmax_idx: softmax 在前驱子图中的索引
        """
        # scenario = "Flash Attention" if is_from_flash_attn else "non-Flash Attention"
        # print(f"\n{'='*70}")
        # print(f"🔍 Checking recomputation dependency ({scenario})")
        # print(f"{'='*70}")
        
        # 查找前驱子图中的 softmax
        softmax_node, softmax_idx = self._find_softmax_in_subgraph(prev_subgraph)
        if softmax_node is None:
            print(f"  ✓ No softmax found in previous subgraph")
            print(f"{'='*70}\n")
            return False, True, -1
        
        print(f"  ✓ Found softmax '{softmax_node.kernel_name}' at position {softmax_idx}")
        
        if is_from_flash_attn:
            # Flash Attention 场景：检查对整个前驱子图的依赖
            is_softmax_store = 'store' in softmax_node.kernel_name.lower()
            has_dep, dep_pairs, matched = self._check_dependency_on_subgraph(subgraph, prev_subgraph)
            
            if has_dep:
                for pair in dep_pairs:
                    print(f"  ⚡ Dependency: {pair}")
                
                # 如果依赖 softmax 但不是 softmax_store，则无效
                if matched and 'softmax' in matched.kernel_name and not is_softmax_store:
                    print(f"\n  ✗ Invalid: requires softmax output but softmax is not softmax_store")
                    print(f"{'='*70}\n")
                    return True, False, softmax_idx
                
                print(f"\n  ✅ Recomputation NEEDED and VALID")
            else:
                print(f"\n  ✅ Recomputation NOT needed")
            
            print(f"{'='*70}\n")
            return has_dep, True, softmax_idx
        else:
            # 非 Flash Attention 场景：只检查对 softmax 的依赖
            has_dep, dep_pairs, _ = self._check_dependency_on_subgraph(
                subgraph, [softmax_node], target_node_filter='softmax'
            )
            
            if has_dep:
                print(f"  ⚡ Need recomputation: {dep_pairs[0]}")
            else:
                print(f"  ✓ No dependency on previous subgraph's softmax")
            
            print(f"{'='*70}\n")
            return has_dep, True, softmax_idx

    def _replace_softmax_with_reduce(
        self, 
        subgraph: List[ComputeNode], 
        softmax_idx: int
    ) -> List[ComputeNode]:
        """将子图中的 softmax 替换为 softmax_reduce"""
        softmax_node = subgraph[softmax_idx]
        new_kernel_name = 'softmax_reduce_ngrid' if 'ngrid' in softmax_node.kernel_name else 'softmax_reduce'
        new_kernel_name = 'softmax_reduce_topk' if softmax_node.kernel_name == 'softmax_mgrid' else new_kernel_name
        new_softmax_node = ComputeNode(
            kernel_name=new_kernel_name,
            params=softmax_node.params.copy(),
            metadata=self.registry.get(new_kernel_name),
            input_mappings=softmax_node.input_mappings.copy() if softmax_node.input_mappings else None,
            parent_ids=softmax_node.parent_ids.copy() if softmax_node.parent_ids else None,
            depth=softmax_node.depth
        )
        
        new_subgraph = subgraph.copy()
        new_subgraph[softmax_idx] = new_softmax_node
        return new_subgraph

    def _find_variant_by_grid_direction(
        self,
        node: ComputeNode,
        target_grid_direction: str
    ) -> Tuple[str, 'KernelMetadata']:
        """根据目标 grid_direction 查找同 family 的变体"""
        current_meta = node.metadata
        
        # 如果当前节点的 grid_direction 已经匹配，直接返回
        if current_meta.grid_direction == target_grid_direction:
            return node.kernel_name, current_meta
        
        # 在同 family 中查找匹配 grid_direction 的变体
        family = current_meta.family
        variants = self.registry.find_variants(family)
        
        for variant in variants:
            if variant.grid_direction == target_grid_direction:
                return variant.name, variant
        
        # 找不到匹配的变体
        grid_desc = "M-grid" if target_grid_direction == "0" else "N-grid"
        raise ValueError(
            f"Cannot find {grid_desc} variant for kernel '{node.kernel_name}' "
            f"(family='{family}'). Available variants: {[v.name for v in variants]}"
        )

    def _copy_nodes_with_mapping(
        self,
        nodes: List[ComputeNode],
        old_id_to_new_node: Dict[str, ComputeNode],
        target_grid_direction: str
    ) -> List[ComputeNode]:
        """
        复制节点列表，应用 kernel 名称映射，并更新 parent_ids
        
        Args:
            nodes: 要复制的节点列表
            old_id_to_new_node: 已有的 ID 映射（会被更新）
        
        Returns:
            复制后的新节点列表
        """
        new_nodes = []
        for node in nodes:
            if node.metadata.grid_direction == target_grid_direction:
                new_kernel_name = node.kernel_name
            else:
                new_kernel_name = self._find_variant_by_grid_direction(
                    node, target_grid_direction
                )[0]
            
            # 更新 parent_ids
            new_parent_ids = []
            if node.parent_ids:
                for pid in node.parent_ids:
                    if pid in old_id_to_new_node:
                        new_parent_ids.append(old_id_to_new_node[pid].id)
                    else:
                        new_parent_ids.append(pid)
            
            new_node = ComputeNode(
                kernel_name=new_kernel_name,
                params=node.params.copy(),
                metadata=self.registry.get(new_kernel_name),
                input_mappings=node.input_mappings.copy() if node.input_mappings else None,
                parent_ids=new_parent_ids if new_parent_ids else None,
                depth=node.depth
            )
            new_nodes.append(new_node)
            old_id_to_new_node[node.id] = new_node
        
        return new_nodes

    def _update_subgraph_parent_ids(
        self,
        subgraph: List[ComputeNode],
        old_id_to_new_node: Dict[str, ComputeNode],
        fallback_node: Optional[ComputeNode] = None
    ) -> None:
        if fallback_node:
            for node in subgraph:
                if node.parent_ids:
                    for pid in node.parent_ids:
                        if 'softmax' in pid and pid not in old_id_to_new_node:
                            old_id_to_new_node[pid] = fallback_node
        
        # 更新 parent_ids
        subgraph_ids = {n.id for n in subgraph}
        for node in subgraph:
            global_node = self._get_node_by_id(node.id) or node
            original_parent_ids = global_node.parent_ids or node.parent_ids or []
            
            new_parent_ids = []
            for pid in original_parent_ids:
                if pid.startswith("input_"):
                    new_parent_ids.append(pid)
                elif pid in old_id_to_new_node:
                    new_parent_ids.append(old_id_to_new_node[pid].id)
                elif pid in subgraph_ids:
                    new_parent_ids.append(pid)
                elif fallback_node:
                    new_parent_ids.append(fallback_node.id)
            
            node.parent_ids = new_parent_ids if new_parent_ids else None

    def _refactor_with_recomputation(
        self,
        subgraph: List[ComputeNode],
        recompute_chain: List[ComputeNode],
        softmax_node: ComputeNode,
        need_softmax_recompute: bool = True,
    ) -> List[ComputeNode]:
        old_id_to_new_node: Dict[str, ComputeNode] = {}
        
        # 复制 recompute_chain
        recompute_nodes = self._copy_nodes_with_mapping(
            recompute_chain, old_id_to_new_node, subgraph[0].metadata.grid_direction
        )
        
        if subgraph[0].metadata.grid_template == ("cdiv(M, BM)", "batch*heads", "1"):
            new_kernel_name = 'softmax_recompute_mgrid'
        elif subgraph[0].metadata.grid_template == ("cdiv(N, BN)", "batch*heads", "1"):
            new_kernel_name = 'softmax_recompute_ngrid'
        else:
            new_kernel_name = 'softmax_recompute_mgrid_kernel'
            
        # 创建 softmax_recompute_ngrid（如果需要）
        softmax_recompute_node = None
        if need_softmax_recompute:
            new_kernel_name = new_kernel_name
            last_node = recompute_nodes[-1] if recompute_nodes else None
            softmax_recompute_node = ComputeNode(
                kernel_name=new_kernel_name,
                params=softmax_node.params.copy(),
                metadata=self.registry.get(new_kernel_name),
                input_mappings={'output': 'input', 'sum_out': 'sum_out'},
                parent_ids=[last_node.id] if last_node else None,
                depth=softmax_node.depth
            )
            old_id_to_new_node[softmax_node.id] = softmax_recompute_node
        
        self._update_subgraph_parent_ids(
            subgraph, old_id_to_new_node, 
            fallback_node=softmax_recompute_node or (recompute_nodes[-1] if recompute_nodes else None)
        )

        if need_softmax_recompute:
            result = recompute_nodes + [softmax_recompute_node] + subgraph
        else:
            result = recompute_nodes + subgraph

        return result

    def _find_dependency_cutoff(
        self,
        subgraph: List[ComputeNode],
        prev_subgraph: List[ComputeNode]
    ) -> Tuple[int, Optional[ComputeNode]]:
        prev_id_to_info = {node.id: (node, idx) for idx, node in enumerate(prev_subgraph)}
        
        max_idx = -1
        dependent_node = None
        
        for node in subgraph:
            global_node = self._get_node_by_id(node.id) or node
            if global_node.parent_ids:
                for pid in global_node.parent_ids:
                    if pid in prev_id_to_info:
                        target_node, target_idx = prev_id_to_info[pid]
                        if target_idx > max_idx:
                            max_idx = target_idx
                            dependent_node = target_node
        
        return max_idx, dependent_node

    def refactor_subgraph_with_recomputation(
        self, 
        subgraph: List[ComputeNode], 
        prev_subgraph: List[ComputeNode],
        softmax_idx: int,
        is_from_flash_attn: bool = True
    ) -> Tuple[List[ComputeNode], Optional[List[ComputeNode]]]:
        """
        统一的子图重计算重构函数
        
        Args:
            subgraph: 当前子图
            prev_subgraph: 前驱子图
            softmax_idx: softmax 在前驱子图中的索引
            is_from_flash_attn: 是否从 Flash Attention 子图重计算
        
        Returns:
            (refactored_subgraph, updated_prev_subgraph)
            - refactored_subgraph: 重构后的当前子图
            - updated_prev_subgraph: 更新后的前驱子图（非 Flash Attention 时会替换 softmax），Flash Attention 时为 None
        """
        softmax_node = prev_subgraph[softmax_idx] if 0 <= softmax_idx < len(prev_subgraph) else None
        updated_prev = None
        
        if is_from_flash_attn:
            # Flash Attention 场景：根据实际依赖的节点确定 recompute_chain
            cutoff_idx, dep_node = self._find_dependency_cutoff(subgraph, prev_subgraph)
            
            if cutoff_idx < 0:
                print(f"  ⚠️ No dependency found, skipping refactoring")
                return subgraph, None
            
            # recompute_chain 包含依赖节点及之前的所有节点
            recompute_chain = prev_subgraph[:cutoff_idx + 1]
            
            # 判断 recompute_chain 是否包含 softmax
            need_softmax_recompute = any('softmax' in n.kernel_name for n in recompute_chain)
            if need_softmax_recompute:
                recompute_chain = prev_subgraph[:-2]
            
        else:
            # 非 Flash Attention 场景：需要替换 softmax 为 softmax_reduce
            if softmax_idx is not len(prev_subgraph) - 1:
                print(f"  ⚠️ Invalid softmax_idx={softmax_idx}, skipping refactoring")
                return subgraph, None
            
            recompute_chain = prev_subgraph[:softmax_idx]
            updated_prev = self._replace_softmax_with_reduce(prev_subgraph, softmax_idx)
            need_softmax_recompute = True
        
        refactored = self._refactor_with_recomputation(
            subgraph, recompute_chain, softmax_node, 
            need_softmax_recompute=need_softmax_recompute,
        )
        
        return refactored, updated_prev

    def _is_valid_partition(self, partition: List[List[ComputeNode]], verbose: bool = True) -> bool:
        """
        检查拆分方案是否有效（子图内grid一致性检查）
        
        Args:
            partition: 拆分方案（子图列表）
            verbose: 是否打印详细信息
        
        Returns:
            bool: 如果所有子图内的节点grid一致，返回True；否则返回False
        """
        partition_desc = " | ".join([
            f"[{','.join([n.kernel_name for n in sg])}]" 
            for sg in partition
        ])
        
        for subgraph_idx, subgraph in enumerate(partition):
            if len(subgraph) == 0:
                continue
            
            # 获取第一个节点的grid和grid_direction作为基准
            first_node = subgraph[0]
            try:
                reference_grid = first_node.metadata.get_grid(**first_node.params)
                reference_direction = first_node.metadata.grid_direction
            except Exception as e:
                if verbose:
                    print(f"  ✗ Partition {partition_desc}")
                    print(f"     Subgraph {subgraph_idx}: Failed to compute grid for {first_node.kernel_name}: {e}")
                return False
            
            # 检查子图内其他节点的grid和grid_direction是否与第一个节点相同
            for node in subgraph[1:]:
                try:
                    node_grid = node.metadata.get_grid(**node.params)
                    node_direction = node.metadata.grid_direction
                    
                    # 检查grid_direction是否一致
                    if node_direction != reference_direction:
                        if verbose:
                            print(f"  ✗ Partition {partition_desc}")
                            print(f"     Subgraph {subgraph_idx}: Grid direction mismatch - {first_node.kernel_name} has direction={reference_direction}, but {node.kernel_name} has direction={node_direction}")
                        return False
                    
                    # 检查grid数值是否相同
                    if node_grid != reference_grid:
                        if verbose:
                            print(f"  ✗ Partition {partition_desc}")
                            print(f"     Subgraph {subgraph_idx}: Grid mismatch - {first_node.kernel_name} has {reference_grid}, but {node.kernel_name} has {node_grid}")
                        return False
                except Exception as e:
                    if verbose:
                        print(f"  ✗ Partition {partition_desc}")
                        print(f"     Subgraph {subgraph_idx}: Failed to compute grid for {node.kernel_name}: {e}")
                    return False
        
        if verbose:
            print(f"  ✓ Partition {partition_desc} - All subgraphs have consistent grids")
        
        return True

    def _expand_partition_with_variants(self, partition: List[List[ComputeNode]], verbose: bool = False) -> Tuple[List[List[List[ComputeNode]]], List[Dict[int, ComputeNode]]]:
        """
        基于给定的 partition，按每个节点的 family 扩展变体 partition 候选。
        对每个节点尝试用 family 中其他变体替换，生成新的 partition 副本。
        仅返回通过 _is_valid_partition 校验的候选。
        
        Args:
            partition: 当前partition（子图列表）
            verbose: 是否打印详细信息
        
        Returns:
            有效的变体partition列表
        """
        id_mappings: List[Dict[int, ComputeNode]] = []
        variant_partitions: List[List[List[ComputeNode]]] = []
        for sg_idx, sg in enumerate(partition):
            for node_idx, node in enumerate(sg):
                meta = node.metadata
                if meta is None:
                    continue
                family = meta.family
                variants = self.registry.find_variants(family)
                for vm in variants:
                    if vm.name == meta.name:
                        continue
                    # 构建partition的浅拷贝并替换对应节点
                    new_partition = [list(s) for s in partition]
                    
                    # 同步vm和原node的tensor名称，保持input_mappings的一致性
                    # 创建vm的深拷贝，修改其tensor名称与原node保持一致
                    vm_copy = copy.deepcopy(vm)
                    # Preserve the original metadata id so the variant's metadata
                    # can be matched against global nodes (prevents mismatch when
                    # rebuilding parent indices / comparing metadata.id later).
                    try:
                        vm_copy.id = meta.id
                    except Exception:
                        # If KernelMetadata is frozen or doesn't allow setting id,
                        # fall back to leaving vm_copy as-is. This is a best-effort
                        # attempt to keep metadata identity consistent for variants.
                        pass
                    original_tensors = {t.name: t for t in meta.tensors}  # 原节点tensor的name映射
                    for i, tensor in enumerate(vm_copy.tensors):
                        # 根据tensor的角色（role）和位置查找原节点中的对应tensor
                        for orig_tensor in meta.tensors:
                            if orig_tensor.role == tensor.role and orig_tensor.arg_index == tensor.arg_index:
                                # 更新vm中的tensor名称为原节点的名称
                                tensor.name = orig_tensor.name
                                break
                    
                    new_node = ComputeNode(
                        kernel_name=vm.name,
                        params=node.params.copy(),
                        metadata=vm_copy,
                        input_mappings=node.input_mappings.copy() if node.input_mappings else None,
                        parent_ids=node.parent_ids.copy() if node.parent_ids else None,
                        depth=node.depth,  # 保持原节点的深度
                        id=node.id  # 保持原节点的唯一ID
                    )
                    new_partition[sg_idx][node_idx] = new_node
                    if self._is_valid_partition(new_partition, verbose=False):
                        variant_partitions.append(new_partition)
                        mapping = {id(node): new_node}  # 原节点 → 新节点
                        id_mappings.append(mapping)
        return variant_partitions, id_mappings

    def _print_partition_candidates_to_file(
        self,
        refactored_partitions: List[Tuple[List[List[ComputeNode]], List[bool]]],
        output_file: str = "search_space_partitions.log"
    ) -> None:
        """
        将所有搜索空间的partition方案打印到文件中
        
        Args:
            refactored_partitions: 重构后的partition列表
            output_file: 输出文件路径，默认为 search_space_partitions.log
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("📋 All Search Space Partitions\n")
            f.write("="*70 + "\n\n")
            
            for p_idx, (partition, subgraph_patterns) in enumerate(refactored_partitions):
                f.write(f"[Partition {p_idx}]\n")
                # 构建当前 partition 中所有节点的 ID -> 节点信息映射
                id_to_info: Dict[str, Tuple[str, str]] = {}  # {id: (node_type, name)}
                
                # 先添加全局 InputNode 的信息
                for node in self.nodes:
                    if isinstance(node, InputNode):
                        id_to_info[node.id] = ("InputNode", node.name)
                
                for sg_idx, subgraph in enumerate(partition):
                    f.write(f"  Subgraph {sg_idx}:\n")
                    for local_idx, node in enumerate(subgraph):
                        kernel_name = node.kernel_name
                        node_id = node.id
                        
                        # 构建 parent 信息
                        parent_strs = []
                        if node.parent_ids:
                            for pid in node.parent_ids:
                                if pid in id_to_info:
                                    ptype, pname = id_to_info[pid]
                                    parent_strs.append(f"{pid}:{pname}")
                                else:
                                    parent_strs.append(f"{pid}")
                        
                        parent_info = f", parents=[{', '.join(parent_strs)}]" if parent_strs else ""
                        f.write(f"    [{local_idx}] {node_id}: {kernel_name}{parent_info}\n")
                        
                        # 记录当前节点信息
                        id_to_info[node_id] = ("ComputeNode", kernel_name)
                
                f.write("\n")
            
            f.write("="*70 + "\n")
        
        # 同时打印到控制台
        print(f"\n{'='*70}")
        print(f"📋 All Search Space Partitions")
        print(f"{'='*70}")
        print(f"✓ Output saved to: {output_file}")
        print(f"  Total partitions: {len(refactored_partitions)}")
        print(f"{'='*70}\n")

    def _copy_partition_with_new_ids(self, partition: List[List[ComputeNode]]) -> List[List[ComputeNode]]:
        """复制 partition，为每个节点生成新的 ID，但保持 parent_ids 的引用关系"""
        new_partition = []
        old_id_to_new_id: Dict[str, str] = {}
        
        # 第一遍：复制节点并生成新 ID
        for subgraph in partition:
            new_subgraph = []
            for node in subgraph:
                new_node = copy.deepcopy(node)
                new_id = _generate_node_id(node.kernel_name)
                old_id_to_new_id[node.id] = new_id
                new_node.id = new_id
                new_subgraph.append(new_node)
            new_partition.append(new_subgraph)
        
        # 第二遍：更新 parent_ids
        for subgraph in new_partition:
            for node in subgraph:
                if node.parent_ids:
                    node.parent_ids = [
                        pid if pid.startswith("input_") else old_id_to_new_id.get(pid, pid)
                        for pid in node.parent_ids
                    ]
        
        return new_partition

    def _refactor_partition_for_recomputation(
        self,
        partition: List[List[ComputeNode]],
        subgraph_patterns: List[bool],
        verbose: bool = True
    ) -> List[Tuple[List[List[ComputeNode]], List[bool]]]:
        """
        对单个 partition 进行重计算重构
        
        Returns:
            List of (refactored_partition, subgraph_patterns)
            每次发生重构时都会把重构后的结果加入列表
        """
        results: List[Tuple[List[List[ComputeNode]], List[bool]]] = []
        
        # 找到 Flash Attention 子图索引
        flash_attn_idx = next((i for i, p in enumerate(subgraph_patterns) if p), -1)
        
        for i, subgraph in enumerate(partition):
            # 情况1: 从 Flash Attention 子图重计算
            if not subgraph_patterns[i] and flash_attn_idx >= 0:
                need_recomp, valid, softmax_idx = self.check_recompute_dependency(
                    subgraph, partition[flash_attn_idx], is_from_flash_attn=True
                )
                if not valid:
                    if verbose:
                        print(f"    ✗ Invalid: recomputation needed but softmax cannot output sum")
                    return []  # 无效，返回空列表
                if need_recomp:
                    partition[i], _ = self.refactor_subgraph_with_recomputation(
                        subgraph, partition[flash_attn_idx], softmax_idx, is_from_flash_attn=True
                    )
                    # 情况1：把重构后的 partition copy 一份放到结果里
                    results.append((self._copy_partition_with_new_ids(partition), subgraph_patterns.copy()))
            
            # 情况2: 从前一个非 Flash Attention 子图重计算
            if i > 0 and not subgraph_patterns[i-1]:
                need_recomp, _, softmax_idx = self.check_recompute_dependency(
                    partition[i], partition[i-1], is_from_flash_attn=False
                )
                if need_recomp:
                    partition[i], updated_prev = self.refactor_subgraph_with_recomputation(
                        partition[i], partition[i-1], softmax_idx, is_from_flash_attn=False
                    )
                    if updated_prev is not None:
                        partition[i-1] = updated_prev
                    # 情况2：把重构后的 partition copy 一份放到结果里
                    results.append((self._copy_partition_with_new_ids(partition), subgraph_patterns.copy()))
        
        # 如果没有发生任何重构，把原始的 partition 加入结果
        if len(results) == 0:
            results.append((partition, subgraph_patterns))
        
        return results

    def _enumerate_and_refactor_partition_candidates(
        self,
        max_splits: int,
        verbose: bool = True,
    ) -> List[Tuple[List[List[ComputeNode]], List[bool]]]:

        import itertools
        all_partitions: List[List[List[ComputeNode]]] = []
        compute_nodes = [n for n in self.nodes if isinstance(n, ComputeNode)]
        n = len(compute_nodes)
        
        def try_add_partition(subgraphs: List[List[ComputeNode]]):
            if self._is_valid_partition(subgraphs, verbose=False):
                all_partitions.append(self._copy_partition_with_new_ids(subgraphs))
                variant_ps, _ = self._expand_partition_with_variants(subgraphs, verbose=False)
                for vp in variant_ps:
                    if vp not in all_partitions:
                        all_partitions.append(self._copy_partition_with_new_ids(vp))

        # 枚举所有拆分方案
        try_add_partition([compute_nodes])  # k=0: 不拆分
        for k in range(max_splits, max_splits + 1):
            for split_points in itertools.combinations(range(1, n), k):
                subgraphs = []
                start = 0
                for split_idx in split_points:
                    subgraphs.append(compute_nodes[start:split_idx])
                    start = split_idx
                subgraphs.append(compute_nodes[start:])
                try_add_partition(subgraphs)

        # 对每个 partition 进行重构
        refactored_partitions = []
        for p_idx, partition in enumerate(all_partitions):
            if verbose:
                desc = " | ".join([f"[{','.join([n.kernel_name for n in sg])}]" for sg in partition])
                print(f"\n  Processing partition {p_idx+1}/{len(all_partitions)}: {desc}")
            
            try:
                subgraph_patterns = self._detect_flash_attention_pass(partition)
                refactor_results = self._refactor_partition_for_recomputation(
                    partition, subgraph_patterns, verbose
                )
                
                for refactored_partition, patterns in refactor_results:
                    # 过滤掉包含单个 sum_h2o 子图的 partition
                    # has_single_sum_h2o = any(
                    #     len(sg) == 1 and sg[0].kernel_name == 'sum_h2o' 
                    #     for sg in refactored_partition
                    # )
                    # if has_single_sum_h2o:
                    #     if verbose:
                    #         print(f"    ✗ Partition skipped (contains standalone sum_h2o subgraph)")
                    #     continue

                    # 重构后再次验证 partition 是否满足 grid 一致性
                    if self._is_valid_partition(refactored_partition, verbose=False):
                        refactored_partitions.append((refactored_partition, patterns))
                        if verbose:
                            print(f"    ✓ Partition added to search space")
                    else:
                        if verbose:
                            print(f"    ✗ Partition invalid (grid mismatch)")
            except Exception as e:
                if verbose:
                    print(f"    ✗ Failed to refactor: {e}")
        
        return refactored_partitions

    def _benchmark_partitions(
        self,
        refactored_partitions: List[Tuple[List[List[ComputeNode]], List[bool]]],
        global_inputs: Dict[str, torch.Tensor],
        params_dict: Dict[str, Any],
        num_warps: Union[int, List[int]],
        num_stages: Union[int, List[int]],
        num_warmup: int,
        num_repeat: int,
        device: str,
        enable_causal_opt: bool,
        configs: Optional[List[Dict[str, Any]]] = None,
        verbose: bool = True,
        tuning_log_callback: Optional[Callable[[str, float, float], None]] = None,
        tuning_start_time: Optional[float] = None,
        initial_subgraph_times: Optional[List[float]] = None,  # Initial subgraph times from Phase 1
        aggressive_gc: bool = True,
    ) -> Tuple[List[List[ComputeNode]], List[Tuple[Any, Tuple]], float, List[Tuple[str, float, int, bool, List[Dict[str, Any]], float]], List[Dict[str, Any]], List[float]]:

        best_time = float('inf')
        best_partition = None
        best_compiled = None
        best_configs = None
        best_subgraph_times = None  # Track subgraph times for best partition
        all_results = []
        
        # If no start time provided, use current time
        if tuning_start_time is None:
            tuning_start_time = time.perf_counter()
        
        autotuner = AutoTuner()
        
        for idx, (partition, subgraph_patterns) in enumerate(refactored_partitions):
            partition_desc = " | ".join([
                f"[{','.join([n.kernel_name for n in sg])}]" 
                for sg in partition
            ])
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"Testing Partition {idx+1}/{len(refactored_partitions)}")
                print(f"{'='*70}")
                print(f"Strategy: {partition_desc}")
                print(f"Subgraphs: {len(partition)}")
            
            partition_total_time = 0
            partition_compiled_list = []
            partition_configs = []
            partition_success = True
            
            # Track best times for each subgraph in this partition
            # Use initial_subgraph_times from Phase 1 if provided, otherwise start with 0
            if initial_subgraph_times is not None and len(initial_subgraph_times) == len(partition):
                subgraph_best_times = list(initial_subgraph_times)
            else:
                subgraph_best_times = [0.0] * len(partition)
            
            try:
                for i, subgraph in enumerate(partition):
                    is_flash_attn = subgraph_patterns[i]
                    
                    # remaining_best_times = times for subgraphs after the current one
                    remaining_best_times = subgraph_best_times[i + 1:]
                    
                    best_config, best_compiled_sub, best_sub_time = autotuner.tune_subgraph(
                        subgraph,
                        i,
                        is_flash_attn,
                        global_inputs,
                        params_dict,
                        num_warmup,
                        num_repeat,
                        device,
                        enable_causal_opt,
                        self._compile_nodes,
                        self._build_fused_args_from_partition,
                        partition,
                        configs=configs,
                        tuning_log_callback=tuning_log_callback,
                        partition_current_time=sum(subgraph_best_times[:i]),  # Sum of previous subgraphs' best times
                        remaining_subgraphs_best_times=remaining_best_times,
                        tuning_start_time=tuning_start_time,
                        aggressive_gc=aggressive_gc,
                    )
                    
                    if best_config is None:
                        partition_success = False
                        break
                    
                    # Update the best time for this subgraph
                    subgraph_best_times[i] = best_sub_time
                    partition_total_time += best_sub_time
                    partition_compiled_list.append(best_compiled_sub)
                    partition_configs.append(best_config)
                
                if partition_success:
                    if verbose:
                        print(f"\n  ⏱️  Total Partition Time: {partition_total_time:.4f} ms")
                    # Include timestamp in results
                    timestamp = time.perf_counter() - tuning_start_time
                    all_results.append((partition_desc, partition_total_time, len(partition), True, partition_configs, timestamp))
                    
                    if partition_total_time < best_time:
                        prev_best = best_time
                        best_time = partition_total_time
                        best_partition = copy.deepcopy(partition)
                        best_compiled = partition_compiled_list
                        best_configs = partition_configs
                        best_subgraph_times = list(subgraph_best_times)  # Save subgraph times for best partition
                        if verbose:
                            if prev_best != float('inf'):
                                improvement = ((prev_best / partition_total_time - 1) * 100)
                                print(f"  ✓ New best! (improved by {improvement:.2f}%)")
                            else:
                                print(f"  ✓ New best!")
                else:
                    if verbose:
                        print(f"  ✗ Partition failed due to subgraph tuning failure")
                    timestamp = time.perf_counter() - tuning_start_time
                    all_results.append((partition_desc, float('inf'), len(partition), False, [], timestamp))
                    
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed to benchmark partition: {e}")
                timestamp = time.perf_counter() - tuning_start_time
                all_results.append((partition_desc, float('inf'), len(partition), False, [], timestamp))
            
            # Clean up after each partition to prevent memory accumulation
            gc.collect()
            torch.cuda.empty_cache()
        
        return best_partition, best_compiled, best_time, all_results, best_configs, best_subgraph_times

    def _print_benchmark_results(
        self,
        all_results: List[Tuple[str, float, int, bool, List[Dict[str, Any]]]],
        best_partition: List[List[ComputeNode]],
        best_time: float,
        best_configs: List[Dict[str, Any]],
    ) -> None:
        sorted_results = sorted(all_results, key=lambda x: x[1])
        
        print(f"\n{'Rank':<6} {'Time (ms)':<12} {'Subgraphs':<12} {'Status':<10} {'Strategy'}")
        print(f"{'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*50}")
        
        for rank, result in enumerate(sorted_results, 1):
            # Handle both old format (5 elements) and new format (6 elements with timestamp)
            partition_desc, exec_time, num_subgraphs, success, configs = result[:5]
            status = "✓" if success else "✗ Failed"
            time_str = f"{exec_time:.4f}" if exec_time != float('inf') else "N/A"
            marker = "🏆" if rank == 1 and success else "  "
            print(f"{marker}{rank:<4} {time_str:<12} {num_subgraphs:<12} {status:<10} {partition_desc}")
            if success and configs:
                for i, cfg in enumerate(configs):
                    cfg_str = f"BM={cfg['BM']}, BN={cfg['BN']}, W={cfg['num_warps']}, S={cfg['num_stages']}"
                    print(f"      Subgraph {i}: {cfg_str}")
        
        # 统计信息
        successful_results = [r for r in all_results if r[3]]
        if len(successful_results) > 1:
            best_exec = sorted_results[0][1]
            worst_exec = max([r[1] for r in successful_results if r[1] != float('inf')])
            avg_exec = sum([r[1] for r in successful_results if r[1] != float('inf')]) / len([r for r in successful_results if r[1] != float('inf')])
            
            print(f"\n📈 Statistics:")
            print(f"   Best:    {best_exec:.4f} ms")
            print(f"   Worst:   {worst_exec:.4f} ms")
            print(f"   Average: {avg_exec:.4f} ms")
            print(f"   Range:   {worst_exec - best_exec:.4f} ms ({(worst_exec/best_exec - 1)*100:.1f}% slower)")
            print(f"   Tested:  {len(all_results)} partitions ({len(successful_results)} successful, {len(all_results) - len(successful_results)} failed)")
        
        print(f"\n{'='*70}")
        print(f"🏆 Optimal Partition Found!")
        print(f"{'='*70}")
        
        # 打印配置信息
        if best_configs:
            for i, cfg in enumerate(best_configs):
                cfg_str = f"BM={cfg['BM']}, BN={cfg['BN']}, W={cfg['num_warps']}, S={cfg['num_stages']}"
                print(f"Subgraph {i} Config: {cfg_str}")
        
        best_desc = " | ".join([
            f"[{' → '.join([n.kernel_name for n in sg])}]" 
            for sg in best_partition
        ])
        print(f"Strategy: {best_desc}")
        print(f"Subgraphs: {len(best_partition)}")
        print(f"Best time: {best_time:.4f} ms")
        print(f"{'='*70}\n")

    def search_optimal_partition(
        self,
        benchmark_fn=None,
        global_inputs: Optional[Dict[str, torch.Tensor]] = None,
        params_dict: Optional[Dict[str, Any]] = None,
        num_warps: Union[int, List[int]] = 4,
        num_stages: Union[int, List[int]] = 2,
        num_warmup: int = 10,
        num_repeat: int = 100,
        max_splits: Optional[int] = None,
        topk: int = 10,
        device: str = 'cuda',
        enable_causal_opt: bool = True,
        search_optimal: bool = True,
        configs: Optional[List[Dict[str, Any]]] = None,
        aggressive_gc: bool = True,
    ) -> Tuple[List[List[ComputeNode]], List[Tuple[Any, Tuple]], float]:
        # Record tuning start time for logging timestamps
        tuning_start_time = time.perf_counter()
        
        compute_nodes = [n for n in self.nodes if isinstance(n, ComputeNode)]
        n = len(compute_nodes)

        if max_splits is None:
            max_splits = n - 1
        else:
            max_splits = min(max_splits, n - 1) 
        
        refactored_partitions = self._enumerate_and_refactor_partition_candidates(max_splits, verbose=True)
        
        # self._print_partition_candidates_to_file(refactored_partitions, "search_space_partitions.log")
        # refactored_partitions = [refactored_partitions[15]]
        if len(refactored_partitions) == 0:
            raise ValueError(
                f"No valid partitions found!\n"
                f"Total compute nodes: {n}\n"
                f"Max splits: {max_splits}\n"
                f"Please check:\n"
                f"  1. Are all nodes in the graph compatible for fusion?\n"
                f"  2. Do all nodes within each potential subgraph have consistent grid dimensions?\n"
                f"  3. Try increasing max_splits or checking kernel compatibility."
            )

        # Step 1: Fast pass with default config
        if configs:
            default_config = configs
        else:
            # Use provided num_warps/num_stages if they are ints, else default to 4/2
            w = num_warps if isinstance(num_warps, int) else 4
            s = num_stages if isinstance(num_stages, int) else 2
            bm = params_dict.get('BM', 64) if params_dict else 64
            bn = params_dict.get('BN', 64) if params_dict else 64
            default_config = [{'BM': bm, 'BN': bn, 'num_warps': w, 'num_stages': s}]

        print(f"\n{'='*70}")
        print(f"🚀 Phase 1: Fast Partition Search (Top-{topk})")
        print(f"{'='*70}")
        
        best_partition, best_compiled, best_time, fast_results, best_configs, phase1_subgraph_times = self._benchmark_partitions(
            refactored_partitions,
            global_inputs,
            params_dict,
            num_warps,
            num_stages,
            num_warmup=5,
            num_repeat=20,
            device=device,
            enable_causal_opt=enable_causal_opt,
            configs=default_config,
            verbose=False,
            tuning_log_callback=None,  # Phase 1 doesn't need config-level logging
            tuning_start_time=tuning_start_time,
            aggressive_gc=aggressive_gc,
        )
        
        # Print Phase 1 tuning log: global best after each partition
        print(f"\n📈 Phase 1 Tuning Log:")
        print(f"{'='*70}")
        current_best_phase1 = float('inf')
        for idx, res in enumerate(fast_results):
            partition_desc, partition_time, num_subgraphs, success, _, timestamp = res
            if success and partition_time < current_best_phase1:
                current_best_phase1 = partition_time
            if current_best_phase1 != float('inf'):
                print(f"  [{timestamp:.2f}s] {current_best_phase1:.4f} ms")
        
        if not search_optimal:
            print(f"\n✅ search_optimal=False, skipping Phase 2.")
            return best_partition, best_compiled, best_time
        
        # Step 2: Sort and pick top K
        successful_indices = [i for i, res in enumerate(fast_results) if res[3]]
        if not successful_indices:
             raise ValueError("All partitions failed in fast pass!")
             
        sorted_indices = sorted(successful_indices, key=lambda i: fast_results[i][1])
        
        print(f"\n📊 Phase 1 Ranking (Fast Pass):")
        for rank, idx in enumerate(sorted_indices, 1):
            res = fast_results[idx]
            print(f"  Rank {rank}: {res[1]:.4f} ms | {res[0]}")

        topk_indices = sorted_indices[:topk]
        topk_partitions = [refactored_partitions[i] for i in topk_indices]
        
        print(f"\nSelected Top-{len(topk_partitions)} partitions for full profiling.")
        
        # Clean up memory between Phase 1 and Phase 2
        gc.collect()
        torch.cuda.empty_cache()
        
        # Step 3: Full profiling on top K
        print(f"\n{'='*70}")
        print(f"🔥 Phase 2: Full Profiling")
        print(f"{'='*70}")
        
        # Phase 2 logging: track best performance across all config changes
        # Start from the best partition's performance in Phase 1
        best_partition_phase1_time = fast_results[sorted_indices[0]][1]
        phase2_tuning_log = []
        phase2_best_perf = best_partition_phase1_time
        
        def phase2_log_callback(event: str, perf: float, timestamp: float):
            nonlocal phase2_best_perf
            is_new_best = perf < phase2_best_perf
            if is_new_best:
                phase2_best_perf = perf
            phase2_tuning_log.append((event, perf, is_new_best, timestamp, phase2_best_perf))
        
        print(f"\n📈 Phase 2 Starting Point: Best Partition from Phase 1 = {best_partition_phase1_time:.4f} ms")
        if phase1_subgraph_times:
            print(f"   Subgraph times: {[f'{t:.4f}' for t in phase1_subgraph_times]}")
        
        best_partition, best_compiled, best_time, all_results, best_configs, _ = self._benchmark_partitions(
            topk_partitions,
            global_inputs,
            params_dict,
            num_warps,
            num_stages,
            num_warmup=5,
            num_repeat=20,
            device=device,
            enable_causal_opt=enable_causal_opt,
            configs=None,
            verbose=True,
            tuning_log_callback=phase2_log_callback,
            tuning_start_time=tuning_start_time,
            initial_subgraph_times=phase1_subgraph_times,  # Pass Phase 1 subgraph times
            aggressive_gc=aggressive_gc,
        )
        
        # Print Phase 2 tuning log: global best after each config
        print(f"\n📈 Phase 2 Tuning Log:")
        print(f"{'='*70}")
        print(f"  Starting: {best_partition_phase1_time:.4f} ms")
        for event, perf, is_new_best, timestamp, current_best in phase2_tuning_log:
            print(f"  [{timestamp:.2f}s] {current_best:.4f} ms")
        print(f"  Final: {phase2_best_perf:.4f} ms")
        if best_partition_phase1_time > 0 and phase2_best_perf > 0:
            speedup = (best_partition_phase1_time / phase2_best_perf - 1) * 100
            print(f"  Phase 2 speedup: {speedup:.2f}%")
        
        if best_partition is None:
            raise ValueError("All partition benchmarks failed!")

        self._print_benchmark_results(all_results, best_partition, best_time, best_configs)
        
        # Final cleanup after autotuning
        gc.collect()
        torch.cuda.empty_cache()
        
        return best_partition, best_compiled, best_time

    def _build_args_for_kernel(self, meta: KernelMetadata, tensors: Dict[str, torch.Tensor], 
                              params: Dict[str, Any], kernel_id: str = None) -> Tuple[List[Any], List[int]]:
        slots: Dict[int, Any] = {}
        for t in meta.tensors:
            tensor_key = t.name
            if kernel_id:
                alias_key = f"{kernel_id}_{t.name}"
                if alias_key in tensors:
                    tensor_key = alias_key
                elif t.name not in tensors:
                    tensor_key = alias_key
            
            if tensor_key not in tensors:
                capitalized_key = t.name.capitalize()
                if capitalized_key in tensors:
                    tensor_key = capitalized_key
                else:
                    # allocate if not provided
                    shape = []
                    for d in t.dims:
                        shape.append(params.get(d, 1))
                    new_tensor = torch.zeros(tuple(shape), device=tensors['__device__'], 
                                           dtype=dtype_map.get(t.dtype, torch.float16))
                    tensors[tensor_key] = new_tensor
            slots[t.arg_index] = tensors[tensor_key]
            # strides
            for si, sidx in enumerate(t.stride_indices):
                if sidx is None:
                    continue
                slots[sidx] = tensors[tensor_key].stride(si)
        # shapes
        for dim_name, aidx in meta.dim_arg_indices.items():
            slots[aidx] = int(params.get(dim_name, 0))
        # scalars
        for sname, sidx in getattr(meta, 'other_arg_indices', {}).items():
            slots[sidx] = params.get(sname, 0)
        # pack ordered
        ordered = sorted(slots.keys())
        return [slots[i] for i in ordered], ordered

    def _build_fused_args_from_partition(self, partition: List[List[ComputeNode]], 
                                        global_inputs: Dict[str, torch.Tensor],
                                        params_dict: Dict[str, Any],
                                        device: str = 'cuda',
                                        external_outputs: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[List[List[Any]], int, torch.Tensor, Dict[str, torch.Tensor]]:
        tensors: Dict[str, torch.Tensor] = {**global_inputs}
        tensors['__device__'] = device
        
        fused_args_per_subgraph: List[List[Any]] = []
        subgraph_count = 0
        output_tensor = None
        latest_tensor_keys: Dict[str, str] = {}
        if global_inputs:
            for name in global_inputs.keys():
                latest_tensor_keys[name] = name
        
        # 预分配的输出 tensor 映射：kernel_name -> {output_index -> (tensor_name, output_node_name, tensor)}
        # 用于在 _build_args_for_kernel 时直接使用预分配的 tensor
        # 支持多输出 kernel（如 sum_and_square 有两个输出，index 0 和 1）
        preallocated_outputs: Dict[str, Dict[int, Tuple[str, str, torch.Tensor]]] = {}
        for output_name, output_node in self._outputs.items():
            source_node = self.nodes[output_node.source_idx]
            if isinstance(source_node, ComputeNode):
                kernel_name = source_node.kernel_name
                if kernel_name not in preallocated_outputs:
                    preallocated_outputs[kernel_name] = {}
                
                # 通过 output_index 获取对应的 tensor 名称
                output_tensors = [t for t in source_node.metadata.tensors if t.role == 'output']
                output_idx = output_node.output_index
                tensor_name = output_tensors[output_idx].name if output_idx < len(output_tensors) else 'output'
                
                # 优先使用外部提供的输出 tensor
                tensor = output_node.tensor
                if external_outputs and output_name in external_outputs:
                    tensor = external_outputs[output_name]
                
                preallocated_outputs[kernel_name][output_idx] = (tensor_name, output_name, tensor)
        
        # 用于找到最大深度节点的输出
        max_depth = -1
        deepest_node_output_key = None
        
        for sg_idx, subgraph in enumerate(partition):
            # 从subgraph中提取所有ComputeNode
            compute_nodes: List[ComputeNode] = [n for n in subgraph if isinstance(n, ComputeNode)]
            
            if len(compute_nodes) == 0:
                continue
            
            subgraph_count += 1
            subgraph_args: List[Any] = []
            
            # 处理跨子图的tensor映射（不仅是第一个节点，所有节点都要处理）
            if sg_idx > 0:
                # 获取前一个子图的所有ComputeNode
                prev_subgraph = partition[sg_idx - 1]
                prev_compute_nodes = [n for n in prev_subgraph if isinstance(n, ComputeNode)]
                
                if prev_compute_nodes:
                    # 遍历当前子图的所有节点，处理它们的跨子图依赖
                    for curr_node in compute_nodes:
                        if not curr_node.input_mappings:
                            continue
                        # 对于跨子图的输入：可能存在多个同名输出（如早期relu输出 vs 后续mul输出 vs 下游add_mask输出）
                        # 选择“最后出现的”匹配作为最深/最新的生产者，避免使用影子或早期中间结果。
                        for source_name, target_name in curr_node.input_mappings.items():
                            candidate_keys = []
                            for prev_node in prev_compute_nodes:
                                prev_outputs = [t.name for t in prev_node.metadata.tensors if t.role == 'output']
                                if source_name in prev_outputs:
                                    source_key = f"{prev_node.metadata.id}_{source_name}"
                                    if source_key in tensors:
                                        candidate_keys.append(source_key)

                            selected_key = None
                            if candidate_keys:
                                # 取最后一个（最深的）匹配
                                selected_key = candidate_keys[-1]
                            else:
                                # 回退使用 latest 或原始名称
                                fallback_key = latest_tensor_keys.get(source_name, source_name)
                                if fallback_key in tensors:
                                    selected_key = fallback_key

                            if selected_key:
                                target_key = f"{curr_node.metadata.id}_{target_name}"
                                tensors[target_key] = tensors[selected_key]
                                latest_tensor_keys[target_name] = target_key
            
            # 第一个节点
            node0 = compute_nodes[0]
            
            # 如果该节点有预分配的输出 tensor，提前放入 tensors 字典
            if node0.kernel_name in preallocated_outputs:
                # 支持多输出：遍历该 kernel 的所有预分配输出
                for output_idx, (tensor_name, output_name, preallocated_tensor) in preallocated_outputs[node0.kernel_name].items():
                    # 输出 tensor 的 key 格式: {kernel_id}_{tensor_name}
                    output_key = f"{node0.metadata.id}_{tensor_name}"
                    tensors[output_key] = preallocated_tensor

            # 处理第一个节点的输入映射（包括 InputNode 名称与 kernel 元数据 tensor 名称不一致的情况）
            # 例如：indexer路径 gemm_qk 使用 Q_idx/K_idx 输入，但 kernel 期望 Q/K；需要在构建参数前建立别名。
            if node0.input_mappings:
                for source_name, target_name in node0.input_mappings.items():
                    # 查找 source tensor
                    source_key = None
                    if source_name in tensors:
                        source_key = source_name
                    elif source_name in latest_tensor_keys and latest_tensor_keys[source_name] in tensors:
                        source_key = latest_tensor_keys[source_name]
                    
                    if source_key:
                        # 创建以 kernel_id 命名的目标别名供 _build_args_for_kernel 使用
                        alias_key = f"{node0.metadata.id}_{target_name}"
                        tensors[alias_key] = tensors[source_key]
                        latest_tensor_keys[target_name] = alias_key
            
            args0, _ = self._build_args_for_kernel(node0.metadata, tensors, node0.params, 
                                                    kernel_id=node0.metadata.id)
            subgraph_args.extend(args0)
            
            # 记录此节点的输出tensor并更新最大深度
            for t in node0.metadata.tensors:
                if t.role == 'output':
                    tensor_key = f"{node0.metadata.id}_{t.name}"
                    if node0.depth > max_depth:
                        max_depth = node0.depth
                        deepest_node_output_key = tensor_key
                    latest_tensor_keys[t.name] = tensor_key
            
            # 后续节点（融合）- 使用 get_fusion_plan
            if len(compute_nodes) > 1:
                # 获取融合计划
                fusion_plan = self.get_fusion_plan(nodes=compute_nodes, verbose=False)
                
                # 使用融合计划构建参数
                for stage, (consumer_kernel_name, ttir_fn, prod_idx, cons_idx) in enumerate(fusion_plan, 1):
                    node = compute_nodes[stage]
                    prev_node = compute_nodes[stage - 1]
                    
                    # 如果该节点有预分配的输出 tensor，提前放入 tensors 字典
                    if node.kernel_name in preallocated_outputs:
                        # 支持多输出：遍历该 kernel 的所有预分配输出
                        for output_idx, (tensor_name, output_name, preallocated_tensor) in preallocated_outputs[node.kernel_name].items():
                            output_key = f"{node.metadata.id}_{tensor_name}"
                            tensors[output_key] = preallocated_tensor
                    
                    # 处理子图内部节点之间的tensor映射
                    if node.input_mappings:
                        for source_name, target_name in node.input_mappings.items():
                            # 查找source tensor（优先来自前一个节点）
                            source_key = f"{prev_node.metadata.id}_{source_name}"
                            if source_key not in tensors:
                                # 回退到最近一次同名tensor
                                fallback_key = latest_tensor_keys.get(source_name)
                                if fallback_key:
                                    source_key = fallback_key
                            
                            if source_key in tensors:
                                # 创建目标key并映射
                                target_key = f"{node.metadata.id}_{target_name}"
                                tensors[target_key] = tensors[source_key]
                                latest_tensor_keys[target_name] = target_key
                    
                    # 构造consumer的完整参数
                    cons_full, cons_order = self._build_args_for_kernel(node.metadata, tensors, 
                                                                        node.params, 
                                                                        kernel_id=node.metadata.id)
                    
                    # 仅追加非融合的参数
                    cons_idx_set = set(cons_idx)
                    for pos, val in enumerate(cons_full):
                        arg_index = cons_order[pos]
                        if arg_index not in cons_idx_set:
                            subgraph_args.append(val)
                    
                    # 记录此节点的输出tensor并更新最大深度
                    for t in node.metadata.tensors:
                        if t.role == 'output':
                            tensor_key = f"{node.metadata.id}_{t.name}"
                            if node.depth > max_depth:
                                max_depth = node.depth
                                deepest_node_output_key = tensor_key
                            latest_tensor_keys[t.name] = tensor_key

            fused_args_per_subgraph.append(subgraph_args)
        
        # 找到深度最大的节点的输出作为最终输出
        if deepest_node_output_key:
            output_tensor = tensors.get(deepest_node_output_key)
        
        # 返回tensors字典（不删除__device__，方便调试）
        result_tensors = {k: v for k, v in tensors.items() if k != '__device__'}
        return fused_args_per_subgraph, subgraph_count, output_tensor, result_tensors

    def _compile_nodes(
        self,
        nodes: List[ComputeNode],
        num_warps: int = 4,
        num_stages: int = 2,
        is_flash_attn: bool = False,
        enable_causal_opt: bool = True,
    ) -> Tuple[Any, Tuple]:
        """编译某个节点切片成单个融合 kernel。返回 (compiled, grid)。"""
        first_node = nodes[0]
        
        params_key = tuple((n.kernel_name, tuple(sorted(n.params.items()))) for n in nodes)
        compiled_key = (params_key, num_warps, num_stages)

        # 先查 compiled 缓存（完全命中直接返回）
        with self._compile_lock:
            if compiled_key in self._compiled_cache:
                return self._compiled_cache[compiled_key]

        # 单节点：create_ttir 是纯 Python，线程安全；triton_compile 在锁外并发
        if len(nodes) == 1:
            with self._compile_lock:
                if params_key not in self._ttir_cache:
                    current_ttir = first_node.metadata.create_ttir(**first_node.params)
                    with tempfile.NamedTemporaryFile("w", suffix=".ttir", delete=False) as f:
                        f.write(current_ttir)
                        self._ttir_cache[params_key] = f.name
            ttir_path = self._ttir_cache[params_key]
            options = {"num_warps": num_warps, "num_stages": num_stages}
            compiled = triton_compile(ttir_path, options=options)
            grid = first_node.metadata.get_grid(**first_node.params)
            with self._compile_lock:
                self._compiled_cache[compiled_key] = (compiled, grid)
            return compiled, grid

        # 多节点融合：MLIR pipeline（ir.context 非线程安全）在锁内串行执行
        with self._compile_lock:
            if params_key not in self._ttir_cache:
                current_ttir = first_node.metadata.create_ttir(**first_node.params)
                current_producer_name = first_node.metadata.ttir_symbol
                fusion_stages = self.get_fusion_plan(nodes=nodes, verbose=False)
                fused_path = None
                for consumer_kernel_name, ttir_fn, prod_out_idx, cons_in_idx in fusion_stages:
                    consumer_ttir = ttir_fn()
                    combined = build_combined_module(
                        current_ttir,
                        consumer_ttir,
                        current_producer_name,
                        self.registry.get(consumer_kernel_name).ttir_symbol,
                    )
                    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
                        f.write(combined)
                        combined_path = f.name
                    cons_name = self.registry.get(consumer_kernel_name).ttir_symbol
                    cons_name = cons_name + "_1" if cons_name == current_producer_name else cons_name
                    fused_path, fused_ttir_text = fuse_kernels_in_ttir(
                        combined_path,
                        producer_kernel_name=current_producer_name,
                        consumer_kernel_name=cons_name,
                        producer_output_arg_idx=prod_out_idx,
                        consumer_input_arg_idx=cons_in_idx,
                    )
                    current_ttir = fused_ttir_text
                    current_producer_name = f"{current_producer_name}_{cons_name}"

                use_causal_opt = enable_causal_opt and getattr(nodes[0].metadata, "grid_direction", "2") != "2"
                direction = nodes[0].metadata.grid_direction if use_causal_opt else "2"
                targetaxis = 0
                first_meta = nodes[0].metadata
                if hasattr(first_meta, 'grid_template'):
                    for axis, template in enumerate(first_meta.grid_template):
                        if 'M' in template and 'cdiv' in template:
                            targetaxis = axis; break
                        elif template == 'M':
                            targetaxis = axis; break

                mask_arg_index = 17
                for i, (consumer_kernel_name, ttir_fn, prod_out_idx, cons_in_idx) in enumerate(fusion_stages):
                    consumer_meta = self.registry.get(consumer_kernel_name)
                    if consumer_meta and consumer_meta.family == "add_mask":
                        if i + 1 < len(fusion_stages):
                            next_stage = fusion_stages[i + 1]
                            if next_stage[0] == "log_neg" or next_stage[0] == "log_neg_ngrid":
                                next_stage = fusion_stages[i + 2]
                            next_prod_out_idx = next_stage[2]
                            if len(next_prod_out_idx) > 0:
                                mask_arg_index = next_prod_out_idx[0] - 1
                        break

                fused_path, _ = opt_kernels_in_ttir(
                    fused_path, direction=direction,
                    targetaxis=targetaxis, mask_arg_index=mask_arg_index
                )
                self._ttir_cache[params_key] = fused_path

        # triton_compile 在锁外并发执行（调用独立 ptxas 子进程，线程安全）
        fused_path = self._ttir_cache[params_key]
        options = {"num_warps": num_warps, "num_stages": num_stages}
        compiled = triton_compile(fused_path, options=options)
        grid = first_node.metadata.get_grid(**first_node.params)
        with self._compile_lock:
            self._compiled_cache[compiled_key] = (compiled, grid)
        return compiled, grid

    def compile(
        self, 
        num_warps: int = 4, 
        num_stages: int = 2,
        search_optimal: bool = True,
        benchmark_fn: Optional[Callable] = None,
        global_inputs: Optional[Dict[str, torch.Tensor]] = None,
        params_dict: Optional[Dict[str, Any]] = None,
        num_warmup: int = 10,
        num_repeat: int = 100,
        max_splits: Optional[int] = 1,
        topk: int = 1,
        device: str = 'cuda',
        enable_causal_opt: Optional[bool] = None,
        configs: Optional[List[Dict[str, Any]]] = None,
        aggressive_gc: bool = True,
    ) -> Tuple[Any, ...]:

        if enable_causal_opt is None:
            enable_causal_opt = self.enable_causal_opt

        best_partition, best_compiled, best_time = self.search_optimal_partition(
            benchmark_fn=benchmark_fn,
            global_inputs=global_inputs,
            params_dict=params_dict,
            num_warps=num_warps,
            num_stages=num_stages,
            num_warmup=num_warmup,
            num_repeat=num_repeat,
            max_splits=max_splits,
            topk=topk,
            device=device,
            enable_causal_opt=enable_causal_opt,
            search_optimal=search_optimal,
            configs=configs,
            aggressive_gc=aggressive_gc,
        )

        # 释放 autotuning 期间缓存的中间编译产物，避免端到端场景 OOM
        self._compiled_cache.clear()
        self._ttir_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

        return best_compiled, best_partition, best_time