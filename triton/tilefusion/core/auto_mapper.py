from typing import List, Tuple, Dict, Optional
import copy
from tilefusion.core.kernel_registry import KernelMetadata, TensorSpec, KernelType


class AutoMapper:
    """自动映射推断器 - 支持多阶段融合
    
    该类负责在kernel融合过程中自动推断参数映射关系，包括：
    1. Tensor指针映射
    2. Stride参数映射（灵活匹配相同shape的tensors）
    3. Shape参数映射（如M, N, K等维度参数）
    
    支持多阶段融合，自动处理参数索引的累积和重编号。
    """
    
    def __init__(self):
        # 当前累积的参数数量（producer参数数）
        self.accumulated_param_count: int = 0
    
    def infer_mapping(
        self,
        producer_meta: KernelMetadata,
        consumer_meta: KernelMetadata,
        tensor_correspondence: Dict[int, int]  # producer_tensor_arg_idx -> consumer_tensor_arg_idx
    ) -> Tuple[List[int], List[int]]:
        
        producer_indices = []
        consumer_indices = []
        
        # Step 1: 映射对应的tensor指针
        # 直接使用 tensor_correspondence 中的索引
        for prod_arg_idx, cons_arg_idx in tensor_correspondence.items():
            producer_indices.append(prod_arg_idx)
            consumer_indices.append(cons_arg_idx)
        
        # Step 2: 基于tensor对应关系推断stride映射
        # 构建维度映射（dim -> dim）
        dim_mapping: Dict[str, str] = {}
        
        if tensor_correspondence:
            # 有 tensor 对应关系时，基于对应的 tensor 构建维度映射
            for prod_arg_idx, cons_arg_idx in tensor_correspondence.items():
                # 根据arg_index找到对应的tensor
                prod_tensor = None
                cons_tensor = None
                
                for t in producer_meta.tensors:
                    if t.arg_index == prod_arg_idx:
                        prod_tensor = t
                        break
                
                for t in consumer_meta.tensors:
                    if t.arg_index == cons_arg_idx:
                        cons_tensor = t
                        break
                
                if not prod_tensor or not cons_tensor:
                    continue
                
                # 检查维度数是否匹配，不匹配则raise error
                if len(prod_tensor.dims) != len(cons_tensor.dims):
                    raise ValueError(
                        f"Dimension mismatch in tensor mapping arg_idx[{prod_arg_idx}] -> arg_idx[{cons_arg_idx}]: "
                        f"producer has {len(prod_tensor.dims)} dims {prod_tensor.dims}, "
                        f"consumer has {len(cons_tensor.dims)} dims {cons_tensor.dims}"
                    )
                
                for i in range(len(prod_tensor.dims)):
                    dim_mapping[prod_tensor.dims[i]] = cons_tensor.dims[i]
        else:
            # 没有 tensor 对应关系（并行分支），假设同名维度对应
            # 收集 producer 和 consumer 的所有维度名称
            prod_dims = set()
            for t in producer_meta.tensors:
                prod_dims.update(t.dims)
            
            cons_dims = set()
            for t in consumer_meta.tensors:
                cons_dims.update(t.dims)
            
            # 同名维度直接映射
            for dim in prod_dims & cons_dims:
                dim_mapping[dim] = dim

        # 映射stride参数
        # 当有 tensor_correspondence 时，基于对应的 tensor 映射
        # 当没有时，基于同名维度匹配所有 tensor 的 stride
        
        def dims_match(mapped_from: List[str], target: List[str]) -> bool:
            if len(mapped_from) != len(target):
                return False
            for i, d in enumerate(mapped_from):
                mapped = dim_mapping.get(d)
                if mapped is None or mapped != target[i]:
                    return False
            return True
        
        if tensor_correspondence:
            # 有对应关系时，基于对应的 tensor 映射 stride
            for prod_arg_idx, cons_arg_idx in tensor_correspondence.items():
                prod_tensor = None
                for t in producer_meta.tensors:
                    if t.arg_index == prod_arg_idx:
                        prod_tensor = t
                        break
                
                if not prod_tensor:
                    continue

                target_cons_tensors = list(consumer_meta.tensors)

                # 映射该producer tensor的每个stride（按维度顺序）
                for dim_idx, prod_dim_name in enumerate(prod_tensor.dims):
                    if dim_idx >= len(prod_tensor.stride_indices):
                        continue
                    prod_stride_idx = prod_tensor.stride_indices[dim_idx]
                    if prod_stride_idx is None:
                        continue

                    cons_dim_name = dim_mapping.get(prod_dim_name)
                    if not cons_dim_name:
                        continue

                    for cons_tensor in target_cons_tensors:
                        # 如果producer和该consumer tensor的dims不一致，则跳过
                        if not dims_match(prod_tensor.dims, cons_tensor.dims):
                            continue
                        if cons_dim_name not in cons_tensor.dims:
                            continue
                        cons_dim_idx = cons_tensor.dims.index(cons_dim_name)
                        if cons_dim_idx >= len(cons_tensor.stride_indices):
                            continue
                        cons_stride_idx = cons_tensor.stride_indices[cons_dim_idx]
                        if cons_stride_idx is None:
                            continue
                        if cons_stride_idx in consumer_indices:
                            continue
                        producer_indices.append(prod_stride_idx)
                        consumer_indices.append(cons_stride_idx)
        else:
            # 找到 producer 的输出 tensor
            prod_output = None
            for t in producer_meta.tensors:
                if t.role == 'output':
                    prod_output = t
                    break
            
            # 映射 consumer 的所有 tensor（input 和 output）的 stride 到 producer 的 output stride
            # 因为它们共享相同的内存布局
            if prod_output:
                for cons_tensor in consumer_meta.tensors:
                    # 按维度顺序映射 stride
                    for dim_idx, cons_stride in enumerate(cons_tensor.stride_indices):
                        # if dim_idx >= len(prod_output.stride_indices):
                        #     break
                        prod_stride = prod_output.stride_indices[dim_idx]
                        if prod_stride is not None and cons_stride is not None:
                            if cons_stride not in consumer_indices:
                                producer_indices.append(prod_stride)
                                consumer_indices.append(cons_stride)
        
        # Step 3: 映射 shape 参数 (M, N, K, TopK 等)
        # 构建反向映射：consumer_dim -> producer_dim
        inv_dim_mapping: Dict[str, str] = {v: k for k, v in dim_mapping.items()}
        
        # 第一步：按 consumer 的参数顺序，映射已知的维度
        shape_pairs: List[Tuple[int, int]] = []  # (prod_idx, cons_idx)
        used_prod_dims = set()  # 已使用的 producer 维度名称
        
        for cons_dim_name, cons_idx in sorted(consumer_meta.dim_arg_indices.items(), key=lambda x: x[1]):
            # 检查 consumer 的这个维度是否能从 tensor 对应关系中找到映射
            prod_dim_name = inv_dim_mapping.get(cons_dim_name)
            
            if prod_dim_name and prod_dim_name in producer_meta.dim_arg_indices:
                # 有映射，添加参数映射
                prod_idx = producer_meta.dim_arg_indices[prod_dim_name]
                if prod_idx not in producer_indices:
                    shape_pairs.append((prod_idx, cons_idx))
                    used_prod_dims.add(prod_dim_name)
        
        # 按 consumer 参数索引排序，保证稳定的输出顺序
        shape_pairs.sort(key=lambda x: x[1])
        for prod_idx, cons_idx in shape_pairs:
            producer_indices.append(prod_idx)
            consumer_indices.append(cons_idx)
        
        return producer_indices, consumer_indices
    
    def update_for_next_fusion(self, consumer_meta: KernelMetadata, mapped_consumer_indices: List[int]):
        """
        更新状态，为下一次融合做准备
        
        Args:
            consumer_meta: Consumer kernel的元数据
            mapped_consumer_indices: 被映射的consumer参数索引列表
        """
        # 计算融合后的参数总数
        consumer_total = consumer_meta.get_total_param_count()
        mapped_count = len(set(mapped_consumer_indices))
        num_unmapped = consumer_total - mapped_count
        fused_param_count = self.accumulated_param_count + num_unmapped
        
        self.accumulated_param_count = fused_param_count
    
    def build_fused_metadata(
        self,
        producer_meta: KernelMetadata,
        consumer_meta: KernelMetadata,
        tensor_correspondence: Dict[int, int]  # producer_tensor_arg_idx -> consumer_tensor_arg_idx
    ) -> KernelMetadata:
        """
        根据映射关系构建融合后的kernel元数据
        
        Args:
            producer_meta: Producer kernel元数据
            consumer_meta: Consumer kernel元数据
            tensor_correspondence: Tensor对应关系
            
        Returns:
            融合后的KernelMetadata
        """
        # 计算融合后输出tensor的索引位置
        # 公式：producer参数总数 + consumer的output tensor在consumer中的原始索引 - 1
        fused_output_arg_index = self.accumulated_param_count + consumer_meta.get_output_tensors()[0].arg_index - len(tensor_correspondence)
        
        fused_name = f"{producer_meta.name}_{consumer_meta.name}_fused"
        
        # 找到consumer的输出tensor
        consumer_output: Optional[TensorSpec] = None
        for tensor in consumer_meta.tensors:
            if tensor.role == 'output':
                consumer_output = tensor
                break
        
        if consumer_output is None:
            raise ValueError(f"Consumer {consumer_meta.name} has no output tensor")
        
        # 构建融合后的输出tensor元数据
        # 输出tensor的stride继承自对应的producer tensor
        # 如果 tensor_correspondence 为空（并行分支），使用 producer 的输出 tensor
        if tensor_correspondence:
            prod_arg_idx = list(tensor_correspondence.keys())[0]
            prod_tensor = None
            for t in producer_meta.tensors:
                if t.arg_index == prod_arg_idx:
                    prod_tensor = t
                    break
        else:
            # 并行分支：使用 producer 的输出 tensor
            prod_tensor = None
            for t in producer_meta.tensors:
                if t.role == 'output':
                    prod_tensor = t
                    break
        
        if prod_tensor is None:
            # 回退：使用 producer 的第一个 tensor
            prod_tensor = producer_meta.tensors[0] if producer_meta.tensors else None
            if prod_tensor is None:
                raise ValueError(f"Producer {producer_meta.name} has no tensors")

        # 构建融合后的输出tensor（使用注册表的TensorSpec）
        fused_output = TensorSpec(
            name=consumer_output.name,
            role='output',
            dims=consumer_output.dims,
            dtype=consumer_output.dtype,
            arg_index=fused_output_arg_index,  # 使用传入的融合后索引
            stride_indices=list(prod_tensor.stride_indices)  # 继承producer tensor的stride索引
        )

        # 复制producer的所有tensor，并添加新的fused_output
        fused_tensors = [copy.deepcopy(t) for t in producer_meta.tensors]
        fused_tensors.append(fused_output)

        # 构建一个最小可用的KernelMetadata（用于后续映射推断）
        # dim_arg_indices 需要合并 producer 和 consumer 的维度参数
        # 但只保留实际需要的维度（以 consumer 的维度为准，补充 producer 独有的维度）
        merged_dim_arg_indices = {}
        
        # 1. 先添加 consumer 需要的维度（这些是下游节点会用到的）
        for dim_name, arg_idx in consumer_meta.dim_arg_indices.items():
            # 在 producer 中查找对应维度的 arg_index
            if dim_name in producer_meta.dim_arg_indices:
                merged_dim_arg_indices[dim_name] = producer_meta.dim_arg_indices[dim_name]
            else:
                # consumer 独有的维度，需要新分配位置
                # 使用 accumulated_param_count 来计算新位置
                new_idx = self.accumulated_param_count + arg_idx - len(tensor_correspondence)
                merged_dim_arg_indices[dim_name] = new_idx
        
        # 2. 添加 producer 独有的维度（例如 gemm_pv 的 K，后续可能被其他节点引用）
        for dim_name, arg_idx in producer_meta.dim_arg_indices.items():
            if dim_name not in merged_dim_arg_indices:
                merged_dim_arg_indices[dim_name] = arg_idx
        
        fused_meta = KernelMetadata(
            name=fused_name,
            kernel_type=KernelType.CUSTOM,
            ttir_generator=lambda **kwargs: "",
            ttir_symbol="fused_stub",
            tensors=fused_tensors,
            grid_template=("1", "1", "1"),
            block_params=[],
            required_params=[],
            dim_arg_indices=merged_dim_arg_indices,
            description="Auto-generated fused kernel metadata"
        )

        return fused_meta
    
    def reset(self):
        """重置mapper状态（开始新的计算图）"""
        self.accumulated_param_count = 0
        # print("\n🔄 Mapper reset for new computation graph")

