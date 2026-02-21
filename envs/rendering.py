"""Pygame visualization for the pursuit-evasion environment.

Supports:
- render_mode='human': Real-time window display
- render_mode='rgb_array': Returns numpy frames for video recording

Design:
- Separate renderer class (composition, not inheritance)
- Lazy pygame initialization (training without render never imports pygame)
- Trail buffers for trajectory visualization
- CBF overlay stub for Phase 2+
"""

from collections import deque

import numpy as np


class PERenderer:
    """Pygame renderer for the pursuit-evasion environment.

    Lazy-initializes pygame only when first render is called.
    """

    COLORS = {
        "bg": (30, 30, 30),
        "arena": (50, 50, 60),
        "arena_border": (200, 200, 200),
        "pursuer": (0, 120, 255),
        "evader": (255, 80, 80),
        "heading": (255, 255, 255),
        "trail_pursuer": (0, 60, 130),
        "trail_evader": (130, 40, 40),
        "obstacle": (100, 100, 100),
        "cbf_safe": (0, 200, 0),
        "cbf_warning": (255, 165, 0),
        "cbf_danger": (255, 50, 50),
        "hud_text": (200, 200, 200),
    }

    def __init__(
        self,
        arena_w: float,
        arena_h: float,
        window_size: int = 800,
        render_fps: int = 30,
    ):
        self.arena_w = arena_w
        self.arena_h = arena_h
        self.window_size = window_size
        self.render_fps = render_fps

        # Scale: fit arena (with 1m margin) into window
        max_dim = max(arena_w + 2, arena_h + 2)
        self.scale = window_size / max_dim
        self.offset_x = window_size / 2
        self.offset_y = window_size / 2

        # Lazy init
        self._pygame = None
        self.window = None
        self.clock = None
        self._font = None

        # Trail buffers
        self.pursuer_trail = deque(maxlen=200)
        self.evader_trail = deque(maxlen=200)

    def _world_to_pixel(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world coords (meters) to pixel coords."""
        px = int(self.offset_x + wx * self.scale)
        py = int(self.offset_y - wy * self.scale)  # flip Y
        return (px, py)

    def _ensure_pygame(self, render_mode: str):
        """Lazy-initialize pygame on first render call."""
        if self._pygame is None:
            import pygame
            self._pygame = pygame

        if render_mode == "human" and self.window is None:
            self._pygame.init()
            self._pygame.display.init()
            self.window = self._pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
            self._pygame.display.set_caption("Pursuit-Evasion")

        if self.clock is None:
            self.clock = self._pygame.time.Clock()

        if self._font is None:
            self._pygame.font.init()
            self._font = self._pygame.font.SysFont("monospace", 14)

    def render_frame(
        self,
        render_mode: str,
        env_state: dict,
    ) -> np.ndarray | None:
        """Main render method.

        Args:
            render_mode: 'human' or 'rgb_array'.
            env_state: Dict with keys:
                pursuer_pos, pursuer_heading, evader_pos, evader_heading,
                step, dt, distance, reward, obstacles (optional),
                cbf_values (optional).

        Returns:
            numpy array (H, W, 3) if rgb_array mode, else None.
        """
        self._ensure_pygame(render_mode)
        pg = self._pygame

        canvas = pg.Surface((self.window_size, self.window_size))
        canvas.fill(self.COLORS["bg"])

        # Layer 1: Arena
        self._draw_arena(canvas)
        # Layer 2: Obstacles (if any)
        self._draw_obstacles(canvas, env_state.get("obstacles", []))
        # Layer 3: CBF safety boundaries (Phase 2+)
        self._draw_cbf_overlay(canvas, env_state)
        # Layer 4: Trajectory trails
        self._draw_trails(canvas, env_state)
        # Layer 5: Agents
        self._draw_agent(
            canvas, env_state["pursuer_pos"],
            env_state["pursuer_heading"], self.COLORS["pursuer"], "P",
        )
        self._draw_agent(
            canvas, env_state["evader_pos"],
            env_state["evader_heading"], self.COLORS["evader"], "E",
        )
        # Layer 6: HUD
        self._draw_hud(canvas, env_state)

        if render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pg.event.pump()  # CRITICAL: prevents OS "not responding"
            pg.display.update()
            self.clock.tick(self.render_fps)
            return None
        else:  # rgb_array
            return np.transpose(
                np.array(pg.surfarray.pixels3d(canvas)),
                axes=(1, 0, 2),
            )

    def _draw_arena(self, canvas):
        """Rectangular arena with border."""
        pg = self._pygame
        tl = self._world_to_pixel(-self.arena_w / 2, self.arena_h / 2)
        w_px = int(self.arena_w * self.scale)
        h_px = int(self.arena_h * self.scale)
        pg.draw.rect(canvas, self.COLORS["arena"], (tl[0], tl[1], w_px, h_px))
        pg.draw.rect(
            canvas, self.COLORS["arena_border"],
            (tl[0], tl[1], w_px, h_px), width=3,
        )

    def _draw_agent(self, canvas, pos, heading, color, label):
        """Draw agent as circle with heading indicator."""
        pg = self._pygame
        center = self._world_to_pixel(pos[0], pos[1])
        r = max(int(0.3 * self.scale), 5)
        pg.draw.circle(canvas, color, center, r)

        # Heading line
        nose_x = pos[0] + 0.5 * np.cos(heading)
        nose_y = pos[1] + 0.5 * np.sin(heading)
        nose = self._world_to_pixel(nose_x, nose_y)
        pg.draw.line(canvas, self.COLORS["heading"], center, nose, width=3)

        # Label
        surf = self._font.render(label, True, color)
        canvas.blit(surf, (center[0] - 4, center[1] - r - 16))

    def _draw_trails(self, canvas, env_state):
        """Breadcrumb trail for both agents."""
        pg = self._pygame
        self.pursuer_trail.append(
            (float(env_state["pursuer_pos"][0]), float(env_state["pursuer_pos"][1]))
        )
        self.evader_trail.append(
            (float(env_state["evader_pos"][0]), float(env_state["evader_pos"][1]))
        )

        for trail, color in [
            (self.pursuer_trail, self.COLORS["trail_pursuer"]),
            (self.evader_trail, self.COLORS["trail_evader"]),
        ]:
            if len(trail) > 1:
                points = [self._world_to_pixel(p[0], p[1]) for p in trail]
                pg.draw.lines(canvas, color, False, points, width=1)

    def _draw_obstacles(self, canvas, obstacles):
        """Draw circular obstacles."""
        pg = self._pygame
        for obs in obstacles:
            center = self._world_to_pixel(obs["x"], obs["y"])
            r = int(obs["radius"] * self.scale)
            pg.draw.circle(canvas, self.COLORS["obstacle"], center, r)

    def _draw_cbf_overlay(self, canvas, env_state):
        """Visualize CBF safety margins. Stub for Phase 1, active Phase 2+."""
        cbf_values = env_state.get("cbf_values", None)
        if cbf_values is None:
            return

        pg = self._pygame
        for pos, h_min in cbf_values:
            center = self._world_to_pixel(pos[0], pos[1])
            if h_min < 0.5:
                color = self.COLORS["cbf_danger"] + (120,)
            elif h_min < 2.0:
                color = self.COLORS["cbf_warning"] + (80,)
            else:
                color = self.COLORS["cbf_safe"] + (40,)
            surf = pg.Surface(
                (self.window_size, self.window_size), pg.SRCALPHA,
            )
            pg.draw.circle(
                surf, color, center,
                int(h_min * self.scale * 0.3), width=2,
            )
            canvas.blit(surf, (0, 0))

    def _draw_hud(self, canvas, env_state):
        """Heads-up display with key metrics."""
        lines = [
            f"t={env_state.get('step', 0):04d}  dt={env_state.get('dt', 0.05):.3f}",
            f"dist={env_state.get('distance', 0):.2f}m",
            f"reward={env_state.get('reward', 0):.3f}",
        ]
        cbf_val = env_state.get("min_cbf_value", None)
        if cbf_val is not None:
            lines.append(f"h_cbf={cbf_val:.3f}")

        for i, line in enumerate(lines):
            surf = self._font.render(line, True, self.COLORS["hud_text"])
            canvas.blit(surf, (10, 10 + i * 18))

    def reset_trails(self):
        """Call on env.reset() to clear trail buffers."""
        self.pursuer_trail.clear()
        self.evader_trail.clear()

    def close(self):
        """Close pygame window and quit."""
        if self._pygame is not None and self.window is not None:
            self._pygame.display.quit()
            self._pygame.quit()
            self.window = None
            self._pygame = None
