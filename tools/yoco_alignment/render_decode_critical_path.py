# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Embed audited trace data into the standalone decode timeline viewer."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = (
        Path(__file__).with_name("decode_critical_path_template.html").read_text()
    )
    data = json.loads(args.data.read_text())
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    args.output.write_text(template.replace("__AUDITED_TRACE_DATA__", payload))


if __name__ == "__main__":
    main()
