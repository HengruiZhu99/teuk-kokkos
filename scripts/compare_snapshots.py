#!/usr/bin/env python3
"""Compare two teuk-kokkos interleaved complex128 checkpoint states."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def read_complex128(path: Path) -> list[complex]:
    payload = path.read_bytes()
    if len(payload) % 16 != 0:
        raise ValueError(f"{path}: byte count is not a multiple of complex128")
    return [complex(real, imag) for real, imag in struct.iter_unpack("=dd", payload)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--absolute", type=float, default=5.0e-12)
    parser.add_argument("--relative", type=float, default=5.0e-11)
    arguments = parser.parse_args()

    reference = read_complex128(arguments.reference)
    candidate = read_complex128(arguments.candidate)
    if len(reference) != len(candidate):
        raise ValueError(
            f"value-count mismatch: {len(reference)} != {len(candidate)}"
        )
    squared_difference = 0.0
    squared_reference = 0.0
    maximum_difference = 0.0
    maximum_reference = 0.0
    for expected, actual in zip(reference, candidate, strict=True):
        difference = abs(actual - expected)
        squared_difference += difference * difference
        squared_reference += abs(expected) ** 2
        maximum_difference = max(maximum_difference, difference)
        maximum_reference = max(maximum_reference, abs(expected))
    count = max(1, len(reference))
    rms_difference = math.sqrt(squared_difference / count)
    rms_reference = math.sqrt(squared_reference / count)
    relative_rms = rms_difference / max(rms_reference, arguments.absolute)
    relative_maximum = maximum_difference / max(
        maximum_reference, arguments.absolute
    )
    passed = maximum_difference <= (
        arguments.absolute + arguments.relative * maximum_reference
    )
    print(f"values={len(reference)}")
    print(f"rms_difference={rms_difference:.17g}")
    print(f"maximum_difference={maximum_difference:.17g}")
    print(f"relative_rms={relative_rms:.17g}")
    print(f"relative_maximum={relative_maximum:.17g}")
    print(f"result={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
