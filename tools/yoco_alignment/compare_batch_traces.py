# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Find the first changed token-aligned boundary in YOCO prefill traces."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ref = torch.load(args.reference, weights_only=True)
    test = torch.load(args.candidate, weights_only=True)
    ref_ids = next(r for r in ref if r["name"] == "embed_tokens")["inputs"][0]
    count = ref_ids.shape[0]
    starts = [i for i, r in enumerate(test) if r["name"] == "embed_tokens"]
    matches = []
    for frame, start in enumerate(starts):
        ids = test[start]["inputs"][0]
        for offset in range(ids.numel() - count + 1):
            if torch.equal(ref_ids, ids[offset : offset + count]):
                end = starts[frame + 1] if frame + 1 < len(starts) else len(test)
                matches.append((frame, offset, test[start:end]))
    if len(matches) != 1:
        raise ValueError(f"Expected one matching token prefix, got {len(matches)}")
    frame, offset, test = matches[0]
    rows = []
    assert len(ref) == len(test)
    for index, (a, b) in enumerate(zip(ref, test)):
        assert a["name"] == b["name"]
        for boundary in ("inputs", "output"):
            lhs = a[boundary] if isinstance(a[boundary], list) else [a[boundary]]
            rhs = b[boundary] if isinstance(b[boundary], list) else [b[boundary]]
            for tensor_index, (x, y) in enumerate(zip(lhs, rhs)):
                if (
                    not isinstance(x, torch.Tensor)
                    or x.ndim == 0
                    or x.shape[0] != count
                ):
                    continue
                y = y[offset : offset + count]
                assert x.shape == y.shape, (a["name"], x.shape, y.shape)
                row = {
                    "event": index,
                    "module": a["name"],
                    "boundary": boundary,
                    "tensor_index": tensor_index,
                    "equal": torch.equal(x, y),
                    "different_elements": int((x != y).sum()),
                    "max_abs_diff": float((x.float() - y.float()).abs().max()),
                }
                rows.append(row)
    result = {
        "candidate_frame": frame,
        "offset": offset,
        "tokens": count,
        "boundaries": rows,
        "first_difference": next((r for r in rows if not r["equal"]), None),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["first_difference"], indent=2))


if __name__ == "__main__":
    main()
