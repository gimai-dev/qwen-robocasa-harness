"""Transform anchored tool poses using the current public robot base pose."""
import numpy as np

from .image_servo import _rotation_xyzw


def base_pose(world_pos, world_rot, state):
    """Return (position, rotation) NumPy arrays in the current robot base frame."""
    rotation = np.asarray(_rotation_xyzw(state['state.base_rotation']))
    translation = np.asarray(state['state.base_position'])
    return (rotation.T @ (np.asarray(world_pos) - translation),
            rotation.T @ np.asarray(world_rot))


def world_pose(base_pos, base_rot, state):
    """Return (position, rotation) NumPy arrays in world, including base tilt."""
    rotation = np.asarray(_rotation_xyzw(state['state.base_rotation']))
    translation = np.asarray(state['state.base_position'])
    return (translation + rotation @ np.asarray(base_pos),
            rotation @ np.asarray(base_rot))
