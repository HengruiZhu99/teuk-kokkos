#!/usr/bin/env python3
"""Small dependency-free audit of the Teukolsky angular eigenvalue and padding nullity."""

from __future__ import annotations


def teukolsky_eigenvalue(ell: int, spin: int) -> int:
    return -(ell - spin) * (ell + spin + 1)


def current_code_eigenvalue(ell: int, spin: int) -> int:
    return spin * spin - ell * (ell + 1)


def retained_modes(spin: int, m: int, ell_max: int) -> int:
    ell_min = max(abs(spin), abs(m))
    return max(0, ell_max - ell_min + 1)


def main() -> None:
    print("Angular eigenvalue comparison for s=-2")
    for ell in range(2, 8):
        correct = teukolsky_eigenvalue(ell, -2)
        current = current_code_eigenvalue(ell, -2)
        print(
            f"ell={ell}: correct={correct:3d}, current={current:3d}, "
            f"current-correct={current-correct:+d}"
        )

    ell_max = 4
    theta_nodes = 7
    print(f"\nNullity for ell_max={ell_max}, theta_nodes={theta_nodes}, s=-2")
    for m in range(-ell_max, ell_max + 1):
        count = retained_modes(-2, m, ell_max)
        if count:
            print(
                f"m={m:+d}: retained={count}, nodal={theta_nodes}, "
                f"uncontrolled_nullity={theta_nodes-count}"
            )

    assert teukolsky_eigenvalue(2, -2) == -4
    assert current_code_eigenvalue(2, -2) == -2
    assert all(
        current_code_eigenvalue(ell, -2)
        - teukolsky_eigenvalue(ell, -2)
        == 2
        for ell in range(2, 20)
    )
    print("\nAudit assertions passed.")


if __name__ == "__main__":
    main()
