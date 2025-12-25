"""Shared I/O functions for MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from pathlib import Path
from typing import Union, Any, List, Dict
import json
import csv


def load_json(file_path: Union[str, Path]) -> dict:
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)


def save_json(data: dict, file_path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_fasta_simple(file_path: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Simple FASTA file loader. Inlined from mxfold2.dataset.

    Returns:
        List of dicts with 'header' and 'sequence' keys
    """
    sequences = []
    current_header = None
    current_seq = ""

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header:
                    sequences.append({
                        'header': current_header,
                        'sequence': current_seq
                    })
                current_header = line[1:]  # Remove '>'
                current_seq = ""
            else:
                current_seq += line

        # Add last sequence
        if current_header:
            sequences.append({
                'header': current_header,
                'sequence': current_seq
            })

    return sequences


def save_csv(data: List[Dict], file_path: Union[str, Path]) -> None:
    """Save list of dicts to CSV file."""
    if not data:
        return

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = data[0].keys()
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def save_bpseq(sequence: str, structure: str, header: str, file_path: Union[str, Path]) -> None:
    """
    Save sequence and structure in BPSEQ format.
    Inlined from mxfold2 output functions.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse dot-bracket to base pairs
    pairs = parse_dot_bracket(structure)

    with open(file_path, 'w') as f:
        f.write(f"# {header}\n")
        for i, nt in enumerate(sequence, 1):
            pair_with = pairs.get(i, 0)
            f.write(f"{i}\t{nt}\t{pair_with}\n")


def parse_dot_bracket(structure: str) -> Dict[int, int]:
    """
    Parse dot-bracket notation to base pairs.
    Returns dict mapping position -> paired_position (1-indexed).
    """
    pairs = {}
    stack = []

    for i, char in enumerate(structure):
        pos = i + 1  # 1-indexed
        if char == '(':
            stack.append(pos)
        elif char == ')' and stack:
            pair_pos = stack.pop()
            pairs[pair_pos] = pos
            pairs[pos] = pair_pos

    return pairs