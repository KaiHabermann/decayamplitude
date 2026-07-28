"""
Numerical comparison of wigner_small_d (JAX, explicit sum) vs
wigner_small_d_sympy (sympy/lambdify reference implementation).

Tests all (j2, m1_2, m2_2) combinations up to j=5/2 at a range of angles,
including boundary values 0 and π.
"""
import numpy as np
import pytest
from decayamplitude.rotation import wigner_small_d, wigner_small_d_sympy

# Test angles: interior, boundary, and near-boundary values
THETAS = [0.0, 0.1, 0.5, 1.0, 1.5707963, 2.0, 2.5, 3.0, 3.1415926]

# Maximum 2j to test (j up to J2_MAX/2)
J2_MAX = 5  # tests j = 0, 1/2, 1, 3/2, 2, 5/2


def all_qn_combinations(j2_max):
    for j2 in range(0, j2_max + 1):
        for m1_2 in range(-j2, j2 + 1, 2):
            for m2_2 in range(-j2, j2 + 1, 2):
                yield j2, m1_2, m2_2


@pytest.mark.parametrize("j2,m1_2,m2_2", list(all_qn_combinations(J2_MAX)))
def test_jax_matches_sympy(j2, m1_2, m2_2):
    thetas = np.array(THETAS)
    ref = np.array(wigner_small_d_sympy(thetas, j2, m1_2, m2_2))
    got = np.array(wigner_small_d(thetas, j2, m1_2, m2_2))
    np.testing.assert_allclose(
        got, ref, atol=1e-12, rtol=1e-10,
        err_msg=f"Mismatch for j2={j2}, m1_2={m1_2}, m2_2={m2_2}",
    )


def test_d0_is_one():
    """d^0_{0,0}(β) = 1 for all β."""
    for theta in THETAS:
        val = complex(wigner_small_d(np.array([theta]), 0, 0, 0)[0])
        assert abs(val - 1.0) < 1e-14, f"d^0_{{0,0}}({theta}) = {val}, expected 1"


def test_identity_at_zero():
    """d^j_{m'm}(0) = δ_{m',m}."""
    for j2 in range(0, J2_MAX + 1):
        for m1_2 in range(-j2, j2 + 1, 2):
            for m2_2 in range(-j2, j2 + 1, 2):
                val = complex(wigner_small_d(np.array([0.0]), j2, m1_2, m2_2)[0])
                expected = 1.0 if m1_2 == m2_2 else 0.0
                assert abs(val - expected) < 1e-13, (
                    f"d^{{j2={j2}}}_{{m1={m1_2},m2={m2_2}}}(0) = {val}, expected {expected}"
                )


def test_unitarity():
    """Σ_m |d^j_{m'm}(β)|² = 1 for all m', β (unitarity of d-matrix)."""
    for j2 in range(0, J2_MAX + 1):
        for m1_2 in range(-j2, j2 + 1, 2):
            for theta in [0.3, 1.0, 2.5]:
                thetas = np.array([theta])
                total = sum(
                    abs(complex(wigner_small_d(thetas, j2, m1_2, m2_2)[0])) ** 2
                    for m2_2 in range(-j2, j2 + 1, 2)
                )
                assert abs(total - 1.0) < 1e-12, (
                    f"Unitarity failed for j2={j2}, m1_2={m1_2}, theta={theta}: sum={total}"
                )
