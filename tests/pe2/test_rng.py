"""Tests for pe2.rng — the L1.1 determinism contract."""

import numpy as np
import pytest

from pe2.rng import derive_seed, make_rng


class TestMakeRng:
    def test_same_seed_identical_stream(self):
        a = make_rng(42).standard_normal(100)
        b = make_rng(42).standard_normal(100)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_stream(self):
        a = make_rng(0).standard_normal(100)
        b = make_rng(1).standard_normal(100)
        assert not np.array_equal(a, b)


class TestDeriveSeed:
    def test_deterministic(self):
        assert derive_seed(123, "env_reset") == derive_seed(123, "env_reset")

    def test_returns_int(self):
        assert isinstance(derive_seed(123, "env_reset"), (int, np.integer))

    def test_labels_give_distinct_seeds(self):
        labels = ["env_reset", "eval_init", "trainer", "opponent_sampling"]
        seeds = [derive_seed(0, lab) for lab in labels]
        assert len(set(seeds)) == len(labels)

    def test_label_order_does_not_matter(self):
        """Adding/deriving another stream first must not shift existing streams."""
        first = derive_seed(7, "env_reset")
        derive_seed(7, "a_new_stream_added_later")
        assert derive_seed(7, "env_reset") == first

    def test_child_streams_are_independent(self):
        a = make_rng(derive_seed(7, "env_reset")).standard_normal(1000)
        b = make_rng(derive_seed(7, "eval_init")).standard_normal(1000)
        assert not np.array_equal(a, b)

    def test_different_root_different_child(self):
        assert derive_seed(1, "env_reset") != derive_seed(2, "env_reset")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
