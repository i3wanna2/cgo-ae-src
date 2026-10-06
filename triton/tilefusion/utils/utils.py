import triton
import re
import time
import torch
global DEVICE 
DEVICE = 'cuda'

ALIAS_LINE_RE = re.compile(r"^#[a-zA-Z0-9_]+\s*=.*$", re.M)
ALIAS_ID_RE = re.compile(r"^#([a-zA-Z0-9_]+)\s*=.*$")


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()

def _extract_func(ttir_text, symbol_name):
    header_re = re.compile(rf"^\s*tt\.func(?:\s+public)?\s+@{re.escape(symbol_name)}\(", re.M)
    m = header_re.search(ttir_text)
    if not m:
        raise RuntimeError(f"function @{symbol_name} not found in TTIR")

    def _scan_balanced(s: str, start_idx: int, open_ch: str, close_ch: str) -> int:
        depth = 0
        for i in range(start_idx, len(s)):
            c = s[i]
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return i
        return -1

    args_close = _scan_balanced(ttir_text, m.end() - 1, '(', ')')
    if args_close == -1:
        raise RuntimeError(f"malformed tt.func for @{symbol_name}: unterminated argument list")

    i = args_close + 1
    while i < len(ttir_text) and ttir_text[i].isspace():
        i += 1
    if i + 1 < len(ttir_text) and ttir_text[i] == '-' and ttir_text[i + 1] == '>':
        i += 2
        while i < len(ttir_text) and ttir_text[i].isspace():
            i += 1
        if i < len(ttir_text) and ttir_text[i] == '(':
            res_close = _scan_balanced(ttir_text, i, '(', ')')
            if res_close == -1:
                raise RuntimeError(f"malformed tt.func for @{symbol_name}: unterminated results list")
            i = res_close + 1
        while i < len(ttir_text) and ttir_text[i].isspace():
            i += 1
    if ttir_text.startswith('attributes', i):
        i += len('attributes')
        while i < len(ttir_text) and ttir_text[i].isspace():
            i += 1
        if i >= len(ttir_text) or ttir_text[i] != '{':
            raise RuntimeError(f"malformed tt.func for @{symbol_name}: expected '{{' after 'attributes'")
        attr_close = _scan_balanced(ttir_text, i, '{', '}')
        if attr_close == -1:
            raise RuntimeError(f"malformed tt.func for @{symbol_name}: unterminated attributes block")
        i = attr_close + 1
        while i < len(ttir_text) and ttir_text[i].isspace():
            i += 1
    if i >= len(ttir_text) or ttir_text[i] != '{':
        raise RuntimeError(f"malformed tt.func for @{symbol_name}: missing body '{{' after signature/attributes")
    body_close = _scan_balanced(ttir_text, i, '{', '}')
    if body_close == -1:
        raise RuntimeError(f"malformed tt.func for @{symbol_name}: unterminated body")
    func_block = ttir_text[m.start():body_close + 1]

    def _strip_loc_balanced(s: str) -> str:
        out = []
        i = 0
        n = len(s)
        while i < n:
            if s.startswith(' loc(', i) or s.startswith('\nloc(', i) or s.startswith('\tloc(', i) or s.startswith('loc(', i):
                # Consume optional leading whitespace before 'loc('
                leading_ws = ''
                while i < n and s[i].isspace() and not s.startswith('loc(', i):
                    leading_ws += s[i]
                    i += 1
                if i >= n or not s.startswith('loc(', i):
                    # Not actually a loc, emit captured ws and continue
                    out.append(leading_ws)
                    continue
                # Skip 'loc(...)' with balanced parentheses
                j = i + 4
                depth = 1
                while j < n and depth > 0:
                    if s[j] == '(':
                        depth += 1
                    elif s[j] == ')':
                        depth -= 1
                    j += 1
                # Skip trailing whitespace after loc(...), but preserve at most one newline
                had_nl = False
                k = j
                while k < n and s[k].isspace():
                    if s[k] == '\n':
                        had_nl = True
                    k += 1
                # Remove a single preceding space to avoid double spaces
                if out and out[-1] == ' ':
                    out.pop()
                if had_nl:
                    out.append('\n')
                i = k
                continue
            else:
                out.append(s[i]); i += 1
        return ''.join(out)

    func_block = _strip_loc_balanced(func_block)
    func_block = func_block.strip()
    return func_block


def _maybe_rename_conflicting_kernel(producer_name: str, consumer_name: str, softmax_ttir: str) -> tuple:
    """如果 producer 与 consumer 名称相同，则把 consumer 的函数名加上后缀 "_1"，并返回修改后的 TTIR 与新名称。

    仅替换 softmax 的函数定义头部（`tt.func ... @name(`），以避免改动函数内其他标识符。
    """
    if producer_name == consumer_name:
        new_name = consumer_name + "_1"
        pattern = re.compile(rf"(\s*tt\.func(?:\s+public)?\s+)@{re.escape(consumer_name)}\(")
        # 只替换第一个匹配（函数定义头部）
        new_softmax = pattern.sub(rf"\1@{new_name}(", softmax_ttir, count=1)
        return new_softmax, new_name
    return softmax_ttir, consumer_name


def build_combined_module(gemm_ttir: str, softmax_ttir: str, producer_kernel_name: str, consumer_kernel_name: str) -> str:
    # Extract alias lines separately and avoid redefinitions by renaming softmax aliases
    gemm_aliases = []
    softmax_aliases = []
    for m in ALIAS_LINE_RE.finditer(gemm_ttir):
        line = m.group(0).strip()
        if line.startswith('#loc'):
            continue  # drop loc aliases; we strip loc(...) already
        gemm_aliases.append(line)
    softmax_alias_map = {}
    for m in ALIAS_LINE_RE.finditer(softmax_ttir):
        line = m.group(0).strip()
        m_id = ALIAS_ID_RE.match(line)
        if not m_id:
            continue
        aid = m_id.group(1)
        if aid.startswith('loc'):
            continue  # drop loc aliases
        new_aid = aid + '_s'
        softmax_alias_map[aid] = new_aid
        # rename id on the alias line itself
        line = line.replace(f'#{aid}', f'#{new_aid}', 1)
        softmax_aliases.append(line)

    # 如果名字冲突（两个 kernel 同名），重命名 consumer 的函数为 name + "_1"
    softmax_ttir, consumer_kernel_name = _maybe_rename_conflicting_kernel(
        producer_kernel_name, consumer_kernel_name, softmax_ttir
    )

    gemm_func = _extract_func(gemm_ttir, producer_kernel_name)
    softmax_func = _extract_func(softmax_ttir, consumer_kernel_name)
    # Replace alias references inside softmax function block
    for old, new in softmax_alias_map.items():
        softmax_func = re.sub(rf"(?<![A-Za-z0-9_])#{re.escape(old)}(?![A-Za-z0-9_])", f"#{new}", softmax_func)

    parts = []
    # De-duplicate gemm aliases
    alias_block = "\n".join(sorted(set(gemm_aliases + softmax_aliases)))
    if alias_block:
        parts.append(alias_block + "\n\n")
    parts.append("module {\n")
    parts.append(gemm_func + "\n\n" + softmax_func + "\n}\n")
    return "".join(parts)





def benchmark_performance(*launchers, num_warmup=10, num_runs=100, labels=None):
    """
    对比多个 kernel 的性能。支持旧 API（两个参数）和新 API（可变参数）。
    
    Args:
        *launchers:        可变数量的无参 callable。
                           如果传入 2 个参数（旧 API），第一个为 baseline，第二个为 fused。
                           如果传入 3+ 个参数，第一个默认为 baseline，其他为待比较的实现。
        num_warmup:        预热轮数（默认 5）
        num_runs:          正式计时轮数（默认 10）
        labels:            可选，为每个 launcher 指定名称的列表。
                           如果未提供，使用默认名称（"Baseline", "Fused", "Impl-3", ...）
    
    Returns:
        dict: 如果是旧 API（2 个 launcher），返回 {'baseline_ms': ..., 'fused_ms': ..., 'speedup': ...}
              如果是新 API（3+ 个 launcher），返回 {'timings': [(label, ms), ...], 'speedups': [(label, speedup), ...]}
    """
    if len(launchers) < 2:
        raise ValueError("Need at least 2 launcher functions")
    
    # 生成默认标签
    if labels is None:
        labels = [launcher.__name__ for launcher in launchers]
    elif len(labels) != len(launchers):
        raise ValueError(f"len(labels) ({len(labels)}) must match len(launchers) ({len(launchers)})")
    
    # Warmup - 对所有 launcher 进行预热
    for _ in range(num_warmup):
        for launcher in launchers:
            launcher()
    torch.cuda.synchronize()

    # Timing - 测量每个 launcher 的性能
    timings = []
    for launcher in launchers:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(num_runs):
            launcher()
        torch.cuda.synchronize()
        t1 = time.time()
        elapsed_time = (t1 - t0) / num_runs  # seconds
        timings.append(elapsed_time * 1000)  # convert to ms
    
    # 计算加速比（相对于第一个实现，即 baseline）
    baseline_ms = timings[0]
    speedups = [baseline_ms / t if t > 0 else float('inf') for t in timings]
    
    # 打印结果
    print(f"\nPerformance comparison (warmup={num_warmup}, runs={num_runs}):")
    for label, ms, speedup in zip(labels, timings, speedups):
        print(f"  {label:12s}: {ms:7.3f} ms  (speedup: {speedup:.2f}x)")
    
    # 找出最快的实现
    min_time = min(timings)
    fastest_idx = timings.index(min_time)
    print(f"\nFastest: {labels[fastest_idx]} ({min_time:.3f} ms)")
    
    # 返回结果 - 兼容旧 API
    if len(launchers) == 2:
        return {
            "baseline_ms": timings[0],
            "fused_ms": timings[1],
            "speedup": speedups[1]
        }
    else:
        return {
            "timings": list(zip(labels, timings)),
            "speedups": list(zip(labels, speedups)),
            "fastest": (labels[fastest_idx], min_time)
        }
    

import torch
from typing import Union, Tuple, List, Callable, Optional

def validate_correctness(
    baseline_launcher,
    fused_launcher,
    Y_baseline: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    Y_fused: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    torch_reference_fn: Optional[Callable] = None,
    rtol=1e-3,
    atol=1e-5,
):
    """
    验证多输出 fused kernel 的正确性。
    Y_baseline 和 Y_fused 应为相同结构的 tuple/list，包含多个张量。
    """
    assert len(Y_baseline) == len(Y_fused), "Baseline and fused must have same number of outputs"

    # === Step 1: 执行 kernel ===
    torch.cuda.synchronize()
    fused_launcher()
    torch.cuda.synchronize()
    baseline_launcher()
    torch.cuda.synchronize()

    # === Step 2: 逐个比较每个输出 ===
    print("\nCorrectness check (multi-output):")
    all_passed = True

    for i, (y_fused, y_baseline) in enumerate(zip(Y_fused, Y_baseline)):
        print(f"\n  --- Output {i} ---")
        # 指标计算时，对非浮点类型先转为 float32，避免 norm 报错
        yf = y_fused.to(torch.float32) if not y_fused.is_floating_point() else y_fused
        yb = y_baseline.to(torch.float32) if not y_baseline.is_floating_point() else y_baseline

        max_diff = torch.max(torch.abs(yf - yb)).item()
        norm_baseline = torch.norm(yb)
        relative_error = (torch.norm(yf - yb) / (norm_baseline + 1e-12)).item()

        print(f"    max abs diff: {max_diff:.6e}")
        print(f"    relative error: {relative_error:.6e}")

        try:
            # 非浮点输出用精确相等比较；浮点输出用 assert_close
            if (not y_fused.is_floating_point()) and (y_fused.dtype == y_baseline.dtype):
                if torch.equal(y_fused, y_baseline):
                    print(f"    ✅ Output {i} matches baseline (exact equality for {y_fused.dtype}).")
                else:
                    raise AssertionError("Non-floating outputs differ (expected exact match)")
            else:
                torch.testing.assert_close(y_fused, y_baseline, rtol=rtol, atol=atol, equal_nan=True)
                print(f"    ✅ Output {i} matches baseline.")
        except AssertionError as e:
            print(f"    ❌ Output {i} failed: {e}")
            all_passed = False

    if not all_passed:
        raise AssertionError("One or more outputs failed correctness check.")

    # === Step 3: 对比参考实现（如果提供）===
    if torch_reference_fn is not None:
        Y_ref = torch_reference_fn()
        assert len(Y_ref) == len(Y_fused), "Reference must have same number of outputs"

        print("\n  Fused vs PyTorch Reference:")
        for i, (y_fused, y_ref) in enumerate(zip(Y_fused, Y_ref)):
            max_diff_ref = torch.max(torch.abs(y_fused - y_ref)).item()
            print(f"    Output {i} max abs diff: {max_diff_ref:.6e}")

            try:
                torch.testing.assert_close(y_fused, y_ref, rtol=rtol, atol=atol, equal_nan=True)
                print(f"    ✅ Output {i} matches reference.")
            except AssertionError as e:
                print(f"    ⚠️ Output {i} reference mismatch: {e}")

    print("\n✅ All outputs passed correctness validation!")