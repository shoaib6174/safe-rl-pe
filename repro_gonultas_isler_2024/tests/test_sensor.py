import numpy as np
import pytest
from envs.sensor import WedgeSensor


def test_sensor_detects_inside_fov():
    sensor = WedgeSensor(fov_angle=2.094, fov_range=6.0)
    assert sensor.is_visible(
        agent_pos=np.array([0.0, 0.0]),
        agent_yaw=0.0,
        opponent_pos=np.array([3.0, 0.0]),
    )


def test_sensor_rejects_outside_angle():
    sensor = WedgeSensor(fov_angle=2.094, fov_range=6.0)
    assert not sensor.is_visible(
        agent_pos=np.array([0.0, 0.0]),
        agent_yaw=0.0,
        opponent_pos=np.array([0.0, 3.0]),
    )


def test_sensor_rejects_outside_range():
    sensor = WedgeSensor(fov_angle=2.094, fov_range=6.0)
    assert not sensor.is_visible(
        agent_pos=np.array([0.0, 0.0]),
        agent_yaw=0.0,
        opponent_pos=np.array([7.0, 0.0]),
    )


def test_omnidirectional_sensor():
    sensor = WedgeSensor(fov_angle=6.283, fov_range=1000.0)
    assert sensor.is_visible(
        agent_pos=np.array([0.0, 0.0]),
        agent_yaw=0.0,
        opponent_pos=np.array([500.0, 500.0]),
    )
