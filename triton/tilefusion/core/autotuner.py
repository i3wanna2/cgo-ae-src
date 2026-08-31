"""
Generic autotuner for kernel fusion.
Provides a framework-agnostic interface for automatically tuning kernel hyperparameters.

This module is designed to be kernel-agnostic and can be used with any fused kernel
by providing appropriate callback functions.
"""

import time
import torch
import itertools
import gc
from typing import List, Dict, Any, Tuple, Callable, Optional


def get_autotune_config(
    bm_list: List[int] = [32, 64, 128], 
    bn_list: List[int] = [32, 64, 128], 
    warps_list: List[int] = [4, 8], 
    stages_list: List[int] = [2, 3, 4, 5]
) -> List[Dict[str, Any]]:
    """Standalone function to generate configurations for autotuning."""
    configs = []
    for bm, bn, w, s in itertools.product(bm_list, bn_list, warps_list, stages_list):
        configs.append({
            'BM': bm,
            'BN': bn,
            'num_warps': w,
            'num_stages': s
        })
    return configs


class AutoTuner:
    def __init__(self):
        pass

    def get_configs(
        self, 
        bm_list: List[int] = [32, 64, 128], 
        bn_list: List[int] = [32, 64, 128], 
        warps_list: List[int] = [4, 8], 
        stages_list: List[int] = [2, 3, 4, 5]
    ) -> List[Dict[str, Any]]:
        return get_autotune_config(bm_list, bn_list, warps_list, stages_list)

    def tune_subgraph(
        self,
        subgraph: List[Any],
        subgraph_idx: int,
        is_flash_attn: bool,
        global_inputs: Dict[str, torch.Tensor],
        params_dict: Dict[str, Any],
        num_warmup: int,
        num_repeat: int,
        device: str,
        enable_causal_opt: bool,
        compile_fn: Callable,
        build_args_fn: Callable,
        full_partition: List[List[Any]],
        configs: Optional[List[Dict[str, Any]]] = None,
        tuning_log_callback: Optional[Callable[[str, float, float], None]] = None,
        partition_current_time: float = 0.0,
        remaining_subgraphs_best_times: Optional[List[float]] = None,
        tuning_start_time: Optional[float] = None,
        aggressive_gc: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[Any, Any]], float]:
        """Tune a single subgraph across multiple configurations."""
        
        if configs is None:
            # Get configs to test
            bm_list = [32, 64, 128]
            bn_list = [32, 64, 128]
            
            configs = self.get_configs(
                bm_list=bm_list,
                bn_list=bn_list,
                warps_list=[4, 8],
                stages_list=[2, 3, 4, 5]
            )
        
        best_time = float('inf')
        best_config = None
        best_compiled = None
        
        # Track the current best partition total time for logging
        # partition_current_time = time already spent on previous subgraphs in this partition
        # remaining_subgraphs_best_times = best times for remaining subgraphs (initially 0 or inf)
        current_subgraph_best_time = float('inf')
        
        kernel_names = ','.join([n.kernel_name for n in subgraph])
        print(f"    🔍 Tuning Subgraph {subgraph_idx}: [{kernel_names}] ({len(configs)} configs)")
        
        # Build args once for the whole partition. 
        # Since problem dimensions (M, N, K, etc.) are constant during autotuning,
        # the pointers and strides in fused_args_list will be the same for all configs.
        # This avoids redundant allocations and reduces OOM risk.
        try:
            fused_args_list, _, _, _ = build_args_fn(full_partition, global_inputs, params_dict, device)
            subgraph_args = fused_args_list[subgraph_idx]
        except Exception as e:
            print(f"      ❌ Failed to build args for subgraph {subgraph_idx}: {e}")
            return None, None, float('inf')

        # ── 并发预编译所有 configs（纯 CPU LLVM/NVCC，可多线程并行）──
        import os
        import copy
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 默认 8 线程；可通过 TILEFUSION_COMPILE_WORKERS 环境变量覆盖
        default_workers = 8
        max_workers = min(int(os.environ.get("TILEFUSION_COMPILE_WORKERS", default_workers)), len(configs))

        def _compile_one(cfg_idx, config, params_snapshot):
            """
            在独立线程中编译单个 config。
            params_snapshot: {node_idx: params_copy}，只 copy params 不 copy metadata。
            _compile_nodes 通过 _cache_lock 保护共享缓存，线程安全。
            """
            bm, bn, w, s = config['BM'], config['BN'], config['num_warps'], config['num_stages']
            # 临时把 params 替换为快照（线程局部操作，subgraph 本体不在并发期间被修改）
            # 注意：这里每个线程传入的是独立的 params_snapshot dict，不共享
            # 构造一个轻量的 proxy node list（只替换 params，metadata 仍然共享原始引用）
            proxy_nodes = []
            for i, node in enumerate(subgraph):
                p = params_snapshot[i].copy()
                if 'BM' in p: p['BM'] = bm
                if 'BN' in p: p['BN'] = bn
                # 用 dataclass replace 创建新 node（浅拷贝，metadata 共享）
                import dataclasses
                proxy_nodes.append(dataclasses.replace(node, params=p))
            try:
                compiled, grid = compile_fn(
                    proxy_nodes,
                    w,
                    s,
                    is_flash_attn=is_flash_attn,
                    enable_causal_opt=enable_causal_opt,
                )
                return cfg_idx, compiled, grid, None
            except Exception as e:
                return cfg_idx, None, None, e

        # 每个 config 的 params 快照（只做 dict 浅拷贝，非常快）
        base_params_snapshot = {i: node.params.copy() for i, node in enumerate(subgraph)}

        precompiled = {}  # cfg_idx -> (compiled, grid) or Exception
        print(f"      ⚡ Pre-compiling {len(configs)} configs with {max_workers} threads...")
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for cfg_idx, config in enumerate(configs):
                future = executor.submit(_compile_one, cfg_idx, config, base_params_snapshot)
                futures[future] = cfg_idx

            for future in as_completed(futures):
                cfg_idx, compiled, grid, err = future.result()
                precompiled[cfg_idx] = (compiled, grid) if err is None else err

        # ── 串行 benchmark（GPU 只有一个）──
        for cfg_idx, config in enumerate(configs):
            bm, bn, w, s = config['BM'], config['BN'], config['num_warps'], config['num_stages']
            
            if cfg_idx % 10 == 0 and cfg_idx > 0:
                print(f"      ... progress: {cfg_idx}/{len(configs)} configs tested")

            # 恢复 node params（供后续逻辑使用）
            for node in subgraph:
                if 'BM' in node.params: node.params['BM'] = bm
                if 'BN' in node.params: node.params['BN'] = bn

            compile_result = precompiled.get(cfg_idx)
            if compile_result is None or isinstance(compile_result, Exception):
                err = compile_result
                if "out of memory" in str(err).lower():
                    print(f"      ⚠️ Config {cfg_idx} failed: Out of memory")
                elif cfg_idx == 0 or (best_config is None and cfg_idx == len(configs) - 1):
                    print(f"      ⚠️ Config {cfg_idx} failed: {err}")
                if aggressive_gc:
                    gc.collect()
                    torch.cuda.empty_cache()
                continue

            compiled, grid = compile_result

            try:
                # Warmup
                for _ in range(num_warmup):
                    compiled[grid](*subgraph_args)
                torch.cuda.synchronize()
                
                # Benchmark
                start = time.perf_counter()
                for _ in range(num_repeat):
                    compiled[grid](*subgraph_args)
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                exec_time = (end - start) / num_repeat * 1000
                
                if exec_time < best_time:
                    best_time = exec_time
                    best_config = config
                    best_compiled = (compiled, grid)
                
                # Log every config if callback is provided (Phase 2 tuning)
                # Calculate the partition total time with current subgraph's best time
                if tuning_log_callback is not None:
                    remaining_time = sum(remaining_subgraphs_best_times) if remaining_subgraphs_best_times else 0.0
                    # Use best_time (current subgraph's best so far) for partition total calculation
                    new_partition_total = partition_current_time + best_time + remaining_time
                    timestamp = time.perf_counter() - tuning_start_time if tuning_start_time else 0.0
                    tuning_log_callback(
                        f"Subgraph {subgraph_idx} config {cfg_idx} (BM={bm}, BN={bn}, W={w}, S={s})",
                        new_partition_total,
                        timestamp
                    )
                
                # Cleanup after each config to prevent OOM
                del compiled
                if aggressive_gc:
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                if "out of memory" in str(e).lower():
                    print(f"      ⚠️ Config {cfg_idx} failed: Out of memory")
                elif cfg_idx == 0 or (best_config is None and cfg_idx == len(configs) - 1):
                    print(f"      ⚠️ Config {cfg_idx} failed: {e}")
                if aggressive_gc:
                    gc.collect()
                    torch.cuda.empty_cache()
                continue
                
        if best_config:
            print(f"      ✅ Best: BM={best_config['BM']}, BN={best_config['BN']}, W={best_config['num_warps']}, S={best_config['num_stages']} -> {best_time:.4f} ms")
            # Restore best params to nodes
            for node in subgraph:
                if 'BM' in node.params: node.params['BM'] = best_config['BM']
                if 'BN' in node.params: node.params['BN'] = best_config['BN']
        else:
            print(f"      ❌ All configs failed for subgraph {subgraph_idx}")
        
        # Final cleanup for this subgraph - delete both subgraph_args and fused_args_list
        del subgraph_args
        del fused_args_list
        gc.collect()
        torch.cuda.empty_cache()
            
        return best_config, best_compiled, best_time


