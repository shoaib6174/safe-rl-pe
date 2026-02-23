"""Pygame visualization for the pursuit-evasion environment.

Supports:
- render_mode='human': Real-time window display
- render_mode='rgb_array': Returns numpy frames for video recording

Design:
- Separate renderer class (composition, not inheritance)
- Lazy pygame initialization (training without render never imports pygame)
- Trail buffers for trajectory visualization
- CBF overlay for Phase 2+ (danger zones, intervention flash, margin HUD)
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
        "obstacle_danger": (180, 50, 50),
        "cbf_safe": (0, 200, 0),
        "cbf_warning": (255, 165, 0),
        "cbf_danger": (255, 50, 50),
        "cbf_intervention": (255, 140, 0),
        "hud_text": (200, 200, 200),
        # Phase 3: Partial observability overlays
        "fov_pursuer": (0, 120, 255),
        "fov_evader": (255, 80, 80),
        "lidar_hit": (100, 255, 100),
        "lidar_miss": (60, 60, 60),
        "belief_ellipse": (255, 255, 0),
        "ghost": (255, 255, 255),
        "wall_segment": (140, 140, 140),
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
                cbf_margins (optional), min_cbf_margin (optional),
                cbf_intervened (optional), cbf_feasible (optional).

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
        # Layer 2b: Wall segments (if any)
        self._draw_wall_segments(canvas, env_state.get("wall_segments", []))
        # Layer 3: CBF safety overlays
        self._draw_cbf_overlay(canvas, env_state)
        # Layer 3b: FOV cones (Phase 3 partial observability)
        sensor_info = env_state.get("sensor_info")
        if sensor_info is not None:
            self._draw_fov_cone(
                canvas, env_state["pursuer_pos"], env_state["pursuer_heading"],
                sensor_info.get("fov_half_angle", np.radians(60)),
                sensor_info.get("fov_range", 10.0),
                self.COLORS["fov_pursuer"],
            )
        # Layer 3c: Lidar rays
        lidar_readings = env_state.get("lidar_readings")
        if lidar_readings is not None:
            self._draw_lidar_rays(
                canvas, env_state["pursuer_pos"], env_state["pursuer_heading"],
                lidar_readings,
                n_rays=len(lidar_readings),
                max_range=env_state.get("lidar_max_range", 5.0),
            )
        # Layer 3d: Belief distribution
        self._draw_belief(canvas, env_state.get("belief_state"))
        # Layer 3e: Ghost marker (undetected opponent)
        if sensor_info is not None and not sensor_info.get("fov_detected", True):
            self._draw_ghost(
                canvas,
                sensor_info.get("last_known_opp_pos"),
                sensor_info.get("steps_since_seen", 0),
            )
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
        """Draw circular obstacles with danger zone outline."""
        pg = self._pygame
        for obs in obstacles:
            center = self._world_to_pixel(obs["x"], obs["y"])
            r = max(int(obs["radius"] * self.scale), 3)
            pg.draw.circle(canvas, self.COLORS["obstacle"], center, r)
            # Danger zone ring (obstacle radius + safety margin)
            r_danger = int((obs["radius"] + 0.15) * self.scale)  # +robot_radius
            pg.draw.circle(canvas, self.COLORS["obstacle_danger"], center, r_danger, width=1)

    def _draw_cbf_overlay(self, canvas, env_state):
        """Visualize CBF safety margins and intervention status.

        Shows:
        - Danger zones around obstacles (colored by margin)
        - Intervention flash when CBF modifies action
        """
        pg = self._pygame

        # Intervention flash overlay
        if env_state.get("cbf_intervened", False):
            surf = pg.Surface((self.window_size, self.window_size), pg.SRCALPHA)
            surf.fill((*self.COLORS["cbf_intervention"], 30))
            canvas.blit(surf, (0, 0))

        # CBF margin visualization around obstacles
        cbf_margins = env_state.get("cbf_margins", None)
        if cbf_margins is not None:
            surf = pg.Surface((self.window_size, self.window_size), pg.SRCALPHA)
            for pos, h_val in cbf_margins:
                center = self._world_to_pixel(pos[0], pos[1])
                if h_val <= 0.1:
                    color = (*self.COLORS["cbf_danger"], 120)
                elif h_val <= 0.3:
                    color = (*self.COLORS["cbf_warning"], 80)
                else:
                    color = (*self.COLORS["cbf_safe"], 40)
                r = max(int(abs(h_val) * self.scale * 0.3), 3)
                pg.draw.circle(surf, color, center, r, width=2)
            canvas.blit(surf, (0, 0))

    def _draw_fov_cone(self, canvas, agent_pos, agent_heading, fov_half_angle,
                       fov_range, color):
        """Draw translucent FOV cone for an agent."""
        pg = self._pygame
        cx, cy = self._world_to_pixel(agent_pos[0], agent_pos[1])
        range_px = int(fov_range * self.scale)

        # Create semi-transparent surface
        fov_surf = pg.Surface((self.window_size, self.window_size), pg.SRCALPHA)

        # Draw cone as a filled polygon (two edge lines + arc approximation)
        n_arc = 20
        points = [(cx, cy)]
        for i in range(n_arc + 1):
            angle = agent_heading - fov_half_angle + (2 * fov_half_angle * i / n_arc)
            # Note: pygame Y is flipped
            px = cx + int(range_px * np.cos(angle))
            py = cy - int(range_px * np.sin(angle))
            points.append((px, py))

        if len(points) >= 3:
            pg.draw.polygon(fov_surf, (*color, 40), points)
            pg.draw.polygon(fov_surf, (*color, 80), points, width=1)

        canvas.blit(fov_surf, (0, 0))

    def _draw_lidar_rays(self, canvas, agent_pos, agent_heading, lidar_readings,
                         n_rays, max_range):
        """Draw lidar rays from agent position."""
        pg = self._pygame
        cx, cy = self._world_to_pixel(agent_pos[0], agent_pos[1])
        angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)

        lidar_surf = pg.Surface((self.window_size, self.window_size), pg.SRCALPHA)

        for i, dist in enumerate(lidar_readings):
            ray_angle = agent_heading + angles[i]
            end_dist = min(float(dist), max_range)
            ex = agent_pos[0] + end_dist * np.cos(ray_angle)
            ey = agent_pos[1] + end_dist * np.sin(ray_angle)
            epx, epy = self._world_to_pixel(ex, ey)

            if dist < max_range:
                color = (*self.COLORS["lidar_hit"], 80)
            else:
                color = (*self.COLORS["lidar_miss"], 40)
            pg.draw.line(lidar_surf, color, (cx, cy), (epx, epy), 1)

        canvas.blit(lidar_surf, (0, 0))

    def _draw_belief(self, canvas, belief_state):
        """Draw BiMDN belief distribution as Gaussian ellipses.

        Args:
            belief_state: Dict with 'means' [(M, 2)], 'stds' [(M, 2)],
                         'weights' [(M,)] from BiMDN output.
        """
        if belief_state is None:
            return

        pg = self._pygame
        belief_surf = pg.Surface((self.window_size, self.window_size), pg.SRCALPHA)

        for mu, sigma, weight in zip(
            belief_state["means"], belief_state["stds"], belief_state["weights"],
        ):
            if weight < 0.05:
                continue  # Skip negligible components
            cx, cy = self._world_to_pixel(mu[0], mu[1])
            rx = max(int(sigma[0] * self.scale * 2), 3)
            ry = max(int(sigma[1] * self.scale * 2), 3)
            alpha = min(int(weight * 120), 120)
            color = (*self.COLORS["belief_ellipse"], alpha)
            pg.draw.ellipse(belief_surf, color,
                           (cx - rx, cy - ry, rx * 2, ry * 2))
            pg.draw.ellipse(belief_surf, (*self.COLORS["belief_ellipse"], alpha + 40),
                           (cx - rx, cy - ry, rx * 2, ry * 2), width=1)

        canvas.blit(belief_surf, (0, 0))

    def _draw_ghost(self, canvas, last_known_pos, steps_since_seen,
                    fade_steps=60):
        """Draw ghost marker at last known opponent position.

        Fades over time as the information becomes stale.
        """
        if last_known_pos is None:
            return

        pg = self._pygame
        alpha = max(10, int(80 * (1.0 - steps_since_seen / fade_steps)))
        if alpha <= 10:
            return  # Fully faded

        gx, gy = self._world_to_pixel(last_known_pos[0], last_known_pos[1])
        ghost_surf = pg.Surface((24, 24), pg.SRCALPHA)
        pg.draw.circle(ghost_surf, (*self.COLORS["ghost"], alpha), (12, 12), 10, width=2)
        # Question mark
        if self._font:
            q_surf = self._font.render("?", True, (*self.COLORS["ghost"], alpha))
            ghost_surf.blit(q_surf, (7, 2))
        canvas.blit(ghost_surf, (gx - 12, gy - 12))

    def _draw_wall_segments(self, canvas, walls):
        """Draw wall segment obstacles."""
        if not walls:
            return
        pg = self._pygame
        for wall in walls:
            p1 = self._world_to_pixel(wall.p1[0], wall.p1[1])
            p2 = self._world_to_pixel(wall.p2[0], wall.p2[1])
            pg.draw.line(canvas, self.COLORS["wall_segment"], p1, p2, width=3)

    def _draw_hud(self, canvas, env_state):
        """Heads-up display with key metrics."""
        lines = [
            f"t={env_state.get('step', 0):04d}  dt={env_state.get('dt', 0.05):.3f}",
            f"dist={env_state.get('distance', 0):.2f}m",
            f"reward={env_state.get('reward', 0):.3f}",
        ]

        # CBF margin with color coding
        cbf_margin = env_state.get("min_cbf_margin", None)
        if cbf_margin is not None:
            if cbf_margin > 0.3:
                color = self.COLORS["cbf_safe"]
            elif cbf_margin > 0.1:
                color = self.COLORS["cbf_warning"]
            else:
                color = self.COLORS["cbf_danger"]
            margin_text = f"h_cbf={cbf_margin:.3f}"
            surf = self._font.render(margin_text, True, color)
            canvas.blit(surf, (10, 10 + len(lines) * 18))
            lines.append("")  # placeholder for spacing

        # Infeasibility indicator
        if env_state.get("cbf_feasible") is False:
            surf = self._font.render("INFEASIBLE", True, self.COLORS["cbf_danger"])
            canvas.blit(surf, (10, 10 + len(lines) * 18))
            lines.append("")

        # Obstacles count
        n_obs = len(env_state.get("obstacles", []))
        if n_obs > 0:
            lines.append(f"obstacles={n_obs}")

        for i, line in enumerate(lines):
            if line:  # skip empty placeholders
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
