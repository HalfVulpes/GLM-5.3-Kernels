#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only AST audit of public kernel bodies against a caller-provided source.

No reference checkout is bundled or assumed. --reference-dir must contain
titan_kv.py, titan_mla.py and titan_moe_align.py. Function signatures, bodies
and decorators must match (docstrings/locations are ignored). MLA target and
launch constants must also match. Import adapters may differ; the public core
must not import vLLM. AST equality is not a numerical GPU validation.
"""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = {
    "titan_kv.py": ("_store_int8", "store_int8"),
    "titan_mla.py": ("_sparse_mla_compute_tile", "_sparse_mla_kernel_final",
                     "_sparse_mla_kernel_split", "_sparse_mla_merge_kernel",
                     "_choose_num_kv_splits", "triton_mla_sparse_attention"),
    "titan_moe_align.py": ("_align_small", "titan_moe_align_small"),
}


def normalized(node):
    node = copy.deepcopy(node)
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if child.body and isinstance(child.body[0], ast.Expr) and isinstance(child.body[0].value, ast.Constant) \
                    and isinstance(child.body[0].value.value, str):
                child.body = child.body[1:]
    return ast.dump(node, include_attributes=False)


def check_core_imports(kernel_dir):
    checked = []
    for name in FUNCTIONS:
        tree = ast.parse((kernel_dir / name).read_text())
        for node in ast.walk(tree):
            imports = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                       else [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            assert all(module != "vllm" and not module.startswith("vllm.") for module in imports), name
        compile(tree, name, "exec")  # Compile only; never execute imports/decorators.
        checked.append(name)
    return checked


def verify(reference_dir, kernel_dir):
    results = []
    for filename, names in FUNCTIONS.items():
        actual = ast.parse((kernel_dir / filename).read_text())
        original = ast.parse((reference_dir / filename).read_text())
        left = {node.name: node for node in original.body if isinstance(node, ast.FunctionDef)}
        right = {node.name: node for node in actual.body if isinstance(node, ast.FunctionDef)}
        for name in names:
            assert name in left and name in right, (filename, name, "missing function")
            canonical = normalized(left[name])
            assert normalized(right[name]) == canonical, (filename, name, "AST differs")
            results.append({"file": filename, "function": name, "ast_sha256": hashlib.sha256(canonical.encode()).hexdigest()})
        if filename == "titan_mla.py":
            def constants(tree):
                return {target.id: normalized(node) for node in tree.body if isinstance(node, ast.Assign)
                        for target in node.targets if isinstance(target, ast.Name)
                        and (target.id.startswith("_") or target.id == "KV_SPLITS_CANDIDATES")}
            assert constants(actual) == constants(original), "MLA target/launch constants differ"
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--kernel-dir", type=Path, default=ROOT / "kernels")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"cpu_passed": True, "core_without_vllm_imports": check_core_imports(args.kernel_dir),
              "reference_compared": args.reference_dir is not None, "gpu_tested": False}
    if args.reference_dir is not None:
        report["equivalent_functions"] = verify(args.reference_dir, args.kernel_dir)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(rendered + "\n")


if __name__ == "__main__": main()
