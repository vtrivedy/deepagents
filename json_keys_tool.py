#!/usr/bin/env python3
import sys
import json

def top_level_keys(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.keys())
    else:
        raise ValueError('JSON root is not an object/dict')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python json_keys_tool.py <path_to_json_file>')
        sys.exit(2)
    path = sys.argv[1]
    try:
        keys = top_level_keys(path)
        print(json.dumps(keys))
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
