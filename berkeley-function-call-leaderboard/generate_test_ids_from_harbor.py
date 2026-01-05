#!/usr/bin/env python3
"""
Generate test_case_ids_to_generate.json from Harbor's parity_sample_source_ids.txt

This ensures the same 123 parity task IDs are used in both BFCL and Harbor evaluations.

Usage:
    python generate_test_ids_from_harbor.py
"""

import json
import urllib.request
from collections import defaultdict
from pathlib import Path

HARBOR_PARITY_IDS_URL = "https://raw.githubusercontent.com/laude-institute/harbor/main/adapters/bfcl/parity_sample_source_ids.txt"


def get_category(task_id: str) -> str:
    """Extract category from task ID."""
    if task_id.startswith('live_'):
        parts = task_id.split('_')
        for i, part in enumerate(parts):
            if '-' in part or part.isdigit():
                return '_'.join(parts[:i])
        return task_id
    
    parts = task_id.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return task_id


def main():
    print(f"Downloading parity IDs from Harbor repository...")
    
    with urllib.request.urlopen(HARBOR_PARITY_IDS_URL) as response:
        content = response.read().decode('utf-8')
    
    task_ids = []
    for line in content.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            task_ids.append(line)
    
    by_category = defaultdict(list)
    for task_id in task_ids:
        category = get_category(task_id)
        by_category[category].append(task_id)
    
    result = {category: sorted(ids) for category, ids in sorted(by_category.items())}
    
    output_file = Path(__file__).parent / "test_case_ids_to_generate.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Generated {output_file}")
    print(f"Total: {len(task_ids)} tasks across {len(result)} categories")
    for category, ids in result.items():
        print(f"  {category}: {len(ids)} tasks")


if __name__ == "__main__":
    main()
