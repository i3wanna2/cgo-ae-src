#!/usr/bin/env python3
"""
Parse tuning log and save as JSON format for visualization.
"""

import re
import json
import argparse
from typing import List, Dict, Any


def parse_tuning_log(log_file: str) -> List[Dict[str, Any]]:
    """Parse the tuning log file and extract performance over time."""
    
    results = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # New format: [timestamp] perf ms
    # Example: [2.38s] 2.5679 ms
    pattern = r'\[(\d+\.?\d*)s\]\s+(\d+\.?\d*)\s+ms'
    
    lines = content.split('\n')
    
    phase = None
    for line in lines:
        line = line.strip()
        
        # Detect phase
        if 'Phase 1 Tuning Log' in line:
            phase = 1
            continue
        elif 'Phase 2 Tuning Log' in line:
            phase = 2
            continue
        elif line.startswith('Starting:') or line.startswith('Final:'):
            continue
        
        if phase in [1, 2]:
            match = re.search(pattern, line)
            if match:
                timestamp = float(match.group(1))
                perf = float(match.group(2))
                
                results.append({
                    'phase': phase,
                    'timestamp': timestamp,
                    'perf': perf,
                })
    
    return results


def convert_to_output_format(
    results: List[Dict[str, Any]],
    hw: str = "NVIDIA_H100_80GB_HBM3",
    op_type: str = "op",
    op: str = "kf",
    seqlen: int = 4096,
    shape: str = "[1,32,4096,128]",
) -> List[Dict[str, Any]]:
    """Convert parsed results to the output format."""
    
    output = []
    for r in results:
        # Phase 1 = graph tuning, Phase 2 = parameter tuning
        method = "graph" if r['phase'] == 1 else "parameter-tuning"
        entry = {
            'hw': hw,
            'method': method,
            'type': op_type,
            'op': op,
            'seqlen': seqlen,
            'shape': shape,
            'avg': r['perf'],
            'real_time': r['timestamp'],
        }
        output.append(entry)
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Parse tuning log and save as JSON')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input log file')
    parser.add_argument('--output', '-o', type=str, default='tuning_results.json', help='Output JSON file')
    parser.add_argument('--hw', type=str, default='NVIDIA_H100_80GB_HBM3', help='Hardware name')
    parser.add_argument('--op', type=str, default='kf', help='Operation name')
    parser.add_argument('--seqlen', type=int, default=4096, help='Sequence length')
    parser.add_argument('--shape', type=str, default='[1,32,4096,128]', help='Tensor shape')
    
    args = parser.parse_args()
    
    # Parse the log
    results = parse_tuning_log(args.input)
    
    # Convert to output format
    output = convert_to_output_format(
        results,
        hw=args.hw,
        op=args.op,
        seqlen=args.seqlen,
        shape=args.shape,
    )
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Parsed {len(output)} entries")
    print(f"Saved to {args.output}")
    
    # Print summary
    if output:
        print(f"\nSummary:")
        print(f"  Total entries: {len(output)}")
        
        if output:
            best_perf = min(e['avg'] for e in output)
            print(f"  Best performance: {best_perf:.4f} ms")
        
        total_time = max(e['real_time'] for e in output)
        print(f"  Total tuning time: {total_time:.2f}s")


if __name__ == '__main__':
    main()
