"""Tests for pe2.sensing — Step 2 automated verify (ray-cast + masking)."""

from typing import ClassVar

import numpy as np
import pytest

from pe2.config import SensingConfig
from pe2.sensing import (
    in_fov,
    is_visible,
    los_clear,
    segment_intersects_disk,
    wrap_angle,
)

CFG = SensingConfig()  # 90 deg, 8m, LOS on


def pose(x, y, theta):
    return np.array([x, y, theta], dtype=float)


class TestInFov:
    def test_dead_ahead_visible(self):
        assert in_fov(pose(0, 0, 0.0), (3.0, 0.0), CFG.fov_angle, CFG.fov_range)

    def test_outside_angle(self):
        # 90 deg cone -> +-45 deg; target at 60 deg bearing is outside
        assert not in_fov(pose(0, 0, 0.0), (1.0, 2.0), CFG.fov_angle, CFG.fov_range)

    def test_behind_not_visible(self):
        assert not in_fov(pose(0, 0, 0.0), (-1.0, 0.0), CFG.fov_angle, CFG.fov_range)

    def test_cone_tracks_heading(self):
        own = pose(0, 0, np.pi / 2)  # facing +y
        assert in_fov(own, (0.0, 3.0), CFG.fov_angle, CFG.fov_range)
        assert not in_fov(own, (3.0, 0.0), CFG.fov_angle, CFG.fov_range)

    def test_beyond_range(self):
        assert not in_fov(pose(0, 0, 0.0), (8.5, 0.0), CFG.fov_angle, CFG.fov_range)

    def test_edge_of_cone_included(self):
        # exactly at 45 deg bearing, within range
        assert in_fov(pose(0, 0, 0.0), (1.0, 1.0), CFG.fov_angle, CFG.fov_range)

    def test_heading_wraparound(self):
        # heading near -pi, target just past the wrap
        own = pose(0, 0, -np.pi + 0.1)
        target = (-2.0, -0.5)  # bearing ~ -pi + 0.24, within +-45 deg of heading
        assert in_fov(own, target, CFG.fov_angle, CFG.fov_range)


class TestSegmentDisk:
    def test_through_disk(self):
        assert segment_intersects_disk((0, 0), (4, 0), (2, 0), 0.5)

    def test_clear_pass(self):
        assert not segment_intersects_disk((0, 0), (4, 0), (2, 2), 0.5)

    def test_endpoint_inside(self):
        assert segment_intersects_disk((0, 0), (1, 0), (1, 0), 0.5)

    def test_tangent_is_clear(self):
        # tangent: distance from center to line == radius exactly
        assert not segment_intersects_disk((0, 0), (4, 0), (2, 0.5), 0.5)

    def test_closest_point_beyond_segment(self):
        # line passes through disk but the segment ends before it
        assert not segment_intersects_disk((0, 0), (1, 0), (5, 0), 0.5)

    def test_degenerate_segment(self):
        assert segment_intersects_disk((1, 0), (1, 0), (1, 0), 0.5)
        assert not segment_intersects_disk((0, 0), (0, 0), (5, 5), 0.5)


class TestLosClear:
    OBS: ClassVar = [{"x": 2.0, "y": 0.0, "radius": 0.5}]

    def test_blocked(self):
        assert not los_clear((0, 0), (4, 0), self.OBS)

    def test_clear(self):
        assert los_clear((0, 1.0), (4, 1.0), self.OBS)

    def test_no_obstacles(self):
        assert los_clear((0, 0), (4, 0), [])


class TestIsVisible:
    OBS: ClassVar = [{"x": 2.0, "y": 0.0, "radius": 0.6}]

    def test_visible_when_clear(self):
        assert is_visible(pose(0, 0, 0.0), (1.0, 0.0), self.OBS, CFG)

    def test_no_see_through_obstacle(self):
        """D6 regression: radius-only bug — target behind obstacle in cone
        and in range must NOT be visible."""
        assert not is_visible(pose(0, 0, 0.0), (4.0, 0.0), self.OBS, CFG)

    def test_occlusion_disabled_by_config(self):
        cfg = SensingConfig(los_occlusion=False)
        assert is_visible(pose(0, 0, 0.0), (4.0, 0.0), self.OBS, cfg)

    def test_in_cone_but_out_of_fov_behind(self):
        # behind obstacle AND outside cone: still not visible (both masks)
        assert not is_visible(pose(0, 0, np.pi), (4.0, 0.0), self.OBS, CFG)

    def test_symmetry_swap_agents(self):
        """Swapping agents gives consistent visibility (symmetric config)."""
        a = pose(0, 0, 0.3)
        b = pose(3.0, 1.0, -2.1)
        # symmetric FOV+LOS: A sees B iff same geometric LOS/cone test on B
        vis_ab = is_visible(a, (b[0], b[1]), self.OBS, CFG)
        vis_ba = is_visible(b, (a[0], a[1]), self.OBS, CFG)
        # LOS is symmetric; FOV is heading-dependent so visibility can differ
        # — but if both headings face each other, both must agree.
        bearing_ab = np.arctan2(b[1] - a[1], b[0] - a[0])
        a2 = pose(a[0], a[1], bearing_ab)
        b2 = pose(b[0], b[1], wrap_angle(bearing_ab + np.pi))
        assert is_visible(a2, (b[0], b[1]), self.OBS, CFG) == is_visible(
            b2, (a[0], a[1]), self.OBS, CFG
        )
        assert isinstance(vis_ab, bool) and isinstance(vis_ba, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
