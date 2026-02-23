"""Complex environment layouts for the generalization study.

These layouts are for Post-Stage 5 evaluation ONLY (not used in core training).
They test how policies trained on the simple arena generalize to unseen complex environments.

Layout types:
- Corridor: 4x20m narrow passage
- L-shaped: 15x15m with rectangular cutout
- Warehouse: 20x20m with shelf grid (parallel wall segments)
"""

from __future__ import annotations

from envs.sensors import WallSegment


def create_corridor_layout() -> dict:
    """Corridor layout: 4x20m narrow passage.

    Tests pursuit in confined space. Two parallel walls create a corridor.

    Returns:
        Dict with 'arena_width', 'arena_height', 'walls', 'obstacles',
        'description'.
    """
    # 20m long, 4m wide corridor centered in a 20x20 arena
    # Walls run along y-axis, creating a passage from x=8 to x=12
    walls = [
        WallSegment(8.0, 0.0, 8.0, 20.0),   # Left wall
        WallSegment(12.0, 0.0, 12.0, 20.0),  # Right wall
    ]
    return {
        "arena_width": 20.0,
        "arena_height": 20.0,
        "walls": walls,
        "obstacles": [],
        "spawn_region": {"x_min": 8.5, "x_max": 11.5, "y_min": 1.0, "y_max": 19.0},
        "description": "Corridor (4x20m): narrow passage, tests confined pursuit",
    }


def create_l_shaped_layout() -> dict:
    """L-shaped room: 15x15m with rectangular cutout.

    Tests corner exploitation and line-of-sight breaks.

    Returns:
        Layout dict.
    """
    # L-shape: remove top-right quadrant (x > 8, y > 8) from 15x15 arena
    walls = [
        WallSegment(8.0, 8.0, 15.0, 8.0),   # Horizontal wall (bottom of cutout)
        WallSegment(8.0, 8.0, 8.0, 15.0),    # Vertical wall (left of cutout)
    ]
    return {
        "arena_width": 15.0,
        "arena_height": 15.0,
        "walls": walls,
        "obstacles": [],
        "spawn_region": {"x_min": 1.0, "x_max": 7.0, "y_min": 1.0, "y_max": 7.0},
        "description": "L-shaped room (15x15m with cutout): corner exploitation",
    }


def create_warehouse_layout() -> dict:
    """Warehouse: 20x20m with shelf grid (parallel wall segments).

    Dense obstacles forming aisles. Tests navigation in structured environments.

    Returns:
        Layout dict.
    """
    walls = []
    # Create 3 rows of shelves, each shelf is 4m long, spaced 3m apart
    for row_y in [5.0, 10.0, 15.0]:
        for col_x in [3.0, 9.0, 15.0]:
            # Each shelf: horizontal segment of 4m length
            walls.append(WallSegment(col_x, row_y, col_x + 4.0, row_y))

    return {
        "arena_width": 20.0,
        "arena_height": 20.0,
        "walls": walls,
        "obstacles": [],
        "spawn_region": {"x_min": 1.0, "x_max": 19.0, "y_min": 1.0, "y_max": 19.0},
        "description": "Warehouse (20x20m with shelf grid): dense navigation",
    }


LAYOUTS = {
    "corridor": create_corridor_layout,
    "l_shaped": create_l_shaped_layout,
    "warehouse": create_warehouse_layout,
}


def get_layout(name: str) -> dict:
    """Get a layout by name.

    Args:
        name: One of 'corridor', 'l_shaped', 'warehouse'.

    Returns:
        Layout dict.

    Raises:
        KeyError: If layout name is not found.
    """
    if name not in LAYOUTS:
        raise KeyError(f"Unknown layout '{name}'. Available: {list(LAYOUTS.keys())}")
    return LAYOUTS[name]()
