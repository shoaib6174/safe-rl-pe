"""Step 1: matplotlib -> GIF renderer.

Renderer-first principle (plan): the renderer ships alongside the env and
every later step adds overlays cumulatively (FOV cone in Step 2, belief in
Step 4, CBF margins in Step 6, interventions in Step 7). Overlay hooks are
therefore part of `draw()` from day one: `overlays` is a list of callables
`(ax, env) -> None` drawn after the base scene.

Determinism: rendering is a pure function of env state + trails; gate
artifacts fix the seed per gate (plan, assumption 5).
"""

from collections.abc import Callable

import imageio
import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow

from pe2.config import EnvConfig

ROLE_STYLE = {
    "pursuer": {"color": "#d62728", "label": "Pursuer"},
    "evader": {"color": "#1f77b4", "label": "Evader"},
}
OBSTACLE_COLOR = "#888888"


class Trails:
    """Per-role position history for trajectory rendering."""

    def __init__(self) -> None:
        self.points: dict[str, list[tuple[float, float]]] = {
            "pursuer": [],
            "evader": [],
        }

    def append(self, role: str, x: float, y: float) -> None:
        self.points[role].append((x, y))

    def reset(self) -> None:
        for role in self.points:
            self.points[role].clear()


class Renderer:
    """Draws the arena scene to an RGB array (reused figure, Agg backend)."""

    def __init__(
        self, cfg: EnvConfig, dpi: int = 100, figsize: tuple[float, float] = (6, 6)
    ):
        self.cfg = cfg
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        half = cfg.arena_size / 2.0
        self.ax.set_xlim(-half - 0.5, half + 0.5)
        self.ax.set_ylim(-half - 0.5, half + 0.5)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def draw(
        self,
        env,
        trails: Trails | None = None,
        overlays: list[Callable] | None = None,
        title_prefix: str = "",
    ) -> np.ndarray:
        """Render the current env state; returns (H, W, 3) uint8."""
        ax = self.ax
        cfg = self.cfg
        half = cfg.arena_size / 2.0
        ax.clear()
        ax.set_xlim(-half - 0.5, half + 0.5)
        ax.set_ylim(-half - 0.5, half + 0.5)

        # Arena boundary
        ax.plot(
            [-half, half, half, -half, -half],
            [-half, -half, half, half, -half],
            color="black",
            linewidth=2,
        )

        # Obstacles
        for obs in env.obstacles:
            ax.add_patch(
                Circle(
                    (obs["x"], obs["y"]), obs["radius"], color=OBSTACLE_COLOR, zorder=2
                )
            )

        # Trails (under agents)
        if trails is not None:
            for role, style in ROLE_STYLE.items():
                pts = trails.points[role]
                if len(pts) >= 2:
                    xs, ys = zip(*pts, strict=True)
                    ax.plot(
                        xs, ys, color=style["color"], alpha=0.4, linewidth=1, zorder=3
                    )

        # Agents: body disk + heading arrow
        for role, style in ROLE_STYLE.items():
            pose = env.pursuer_pose if role == "pursuer" else env.evader_pose
            x, y, theta = pose
            ax.add_patch(
                Circle((x, y), cfg.agent_radius, color=style["color"], zorder=4)
            )
            arrow_len = 3.0 * cfg.agent_radius
            ax.add_patch(
                FancyArrow(
                    x,
                    y,
                    arrow_len * np.cos(theta),
                    arrow_len * np.sin(theta),
                    width=0.02,
                    head_width=0.12,
                    head_length=0.12,
                    color=style["color"],
                    zorder=5,
                )
            )

        # Overlays from later steps (FOV cone, belief, CBF margins, ...)
        for overlay in overlays or []:
            overlay(ax, env)

        prefix = f"{title_prefix} · " if title_prefix else ""
        ax.set_title(
            f"{prefix}step {env.current_step} / {cfg.max_steps} — {env.outcome}",
            fontsize=10,
        )

        self.fig.canvas.draw()
        frame = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]
        return frame.copy()

    def close(self) -> None:
        plt.close(self.fig)


class FovOverlay:
    """Step 2 overlay: FOV wedge + occluded shadow + observation blip.

    Drawn from one agent's perspective (`role`):
    - translucent wedge = the FOV cone (angle + range)
    - darker sector behind each obstacle = the occluded shadow
      (tangent construction: angular half-width asin(r/d))
    - opponent marker ringed in the viewer's color when VISIBLE to the
      viewer, dimmed with a gray X when occluded/out of view — i.e., the
      marker shows the *observation content*, not just the true state
    """

    def __init__(self, role: str):
        self.role = role
        self.other = "evader" if role == "pursuer" else "pursuer"

    def __call__(self, ax, env) -> None:
        from matplotlib.patches import Wedge

        from pe2.sensing import is_visible

        cfg = env.cfg
        sensing = cfg.sensing
        own = env.pursuer_pose if self.role == "pursuer" else env.evader_pose
        opp = env.evader_pose if self.role == "pursuer" else env.pursuer_pose
        color = ROLE_STYLE[self.role]["color"]
        fov_deg = np.degrees(sensing.fov_angle)
        heading_deg = np.degrees(own[2])

        # FOV wedge
        ax.add_patch(
            Wedge(
                (own[0], own[1]),
                sensing.fov_range,
                heading_deg - fov_deg / 2,
                heading_deg + fov_deg / 2,
                color=color,
                alpha=0.10,
                zorder=1,
            )
        )

        # Occluded shadow sector behind each obstacle
        for obs in env.obstacles:
            dx = obs["x"] - own[0]
            dy = obs["y"] - own[1]
            d = np.hypot(dx, dy)
            if d <= obs["radius"] or d - obs["radius"] > sensing.fov_range:
                continue  # standing on it (degenerate) or fully out of range
            theta_c = np.degrees(np.arctan2(dy, dx))
            alpha = np.degrees(np.arcsin(min(1.0, obs["radius"] / d)))
            ax.add_patch(
                Wedge(
                    (own[0], own[1]),
                    sensing.fov_range,
                    theta_c - alpha,
                    theta_c + alpha,
                    width=sensing.fov_range - (d - obs["radius"]),
                    color="black",
                    alpha=0.25,
                    zorder=2,
                )
            )

        # Observation content: opponent blip state
        visible = is_visible(own, (opp[0], opp[1]), env.obstacles, sensing)
        if visible:
            ax.add_patch(
                Circle(
                    (opp[0], opp[1]),
                    cfg.agent_radius * 1.8,
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                    zorder=6,
                )
            )
        else:
            ax.plot(
                opp[0],
                opp[1],
                marker="x",
                color="#666666",
                markersize=10,
                markeredgewidth=2,
                zorder=6,
            )
        ax.text(
            0.02,
            0.03,
            f"{self.role} sees opponent: {'YES' if visible else 'NO'}",
            transform=ax.transAxes,
            fontsize=9,
            color=color,
        )


class BeliefOverlay:
    """Step 4 overlay: belief ghost for one LastSeenBelief tracker.

    - dashed ring at the belief mean (the "ghost")
    - translucent disk of radius sigma (grows during occlusion)
    - velocity arrow (0.5s lookahead)
    - status text: sigma + visible/occluded-since
    """

    def __init__(self, belief, color: str = "#9467bd"):
        self.belief = belief
        self.color = color

    def __call__(self, ax, env) -> None:
        b = self.belief
        if not b.seen:
            ax.text(
                0.02,
                0.10,
                "belief: never seen",
                transform=ax.transAxes,
                fontsize=9,
                color=self.color,
            )
            return
        ax.add_patch(Circle(b.mean, b.sigma, color=self.color, alpha=0.12, zorder=2))
        ax.add_patch(
            Circle(
                b.mean,
                env.cfg.agent_radius,
                fill=False,
                edgecolor=self.color,
                linewidth=2,
                linestyle="--",
                zorder=6,
            )
        )
        ax.add_patch(
            FancyArrow(
                b.mean[0],
                b.mean[1],
                b.velocity[0] * 0.5,
                b.velocity[1] * 0.5,
                width=0.02,
                head_width=0.10,
                head_length=0.10,
                color=self.color,
                zorder=6,
            )
        )
        state = (
            "visible"
            if not b.just_lost
            else f"occluded {(b.t_since_seen or 0) * env.cfg.dt:.1f}s"
        )
        ax.text(
            0.02,
            0.10,
            f"belief sigma={b.sigma:.2f}m - {state}",
            transform=ax.transAxes,
            fontsize=9,
            color=self.color,
        )


def save_gif(frames: list[np.ndarray], path: str, fps: float = 20.0) -> None:
    """Write frames to an animated GIF."""
    imageio.mimsave(path, frames, fps=fps, loop=0)


class ShieldOverlay:
    """Step 6 overlay: nominal vs executed action, constraint surfaces, flash.

    Draws, for one role:
    - the CBF constraint surfaces the shield actually used: a dashed circle at
      `obstacle radius + agent_radius + margin` around each obstacle and a
      dashed rectangle at the (margin-shrunk) boundary limits, so "slides
      along the obstacle boundary" is a visible geometric claim
    - the NOMINAL action as a dashed grey unicycle arc over `lookahead`
      seconds, and the EXECUTED action as a solid arc in the role's colour;
      the two coincide exactly when the shield does not intervene
    - an intervention flash (ring + caption) on exactly the steps whose
      logged `info["intervention"]` is True
    - a readout: fallback tier and the reportable clearance in metres
      (`min_clearance`; `min_h` is a mixed-unit QP margin, D89, and is never
      drawn as a distance)

    `live` is a dict the CALLER refreshes each step before drawing, with keys
    "nominal", "executed" and "info" (the dict returned by
    `CBFShield.filter`). The overlay never runs the shield itself, so what is
    drawn is exactly what was logged — which is what makes the flash rate and
    the logged intervention rate the same number by construction.
    """

    FLASH_COLOR = "#ff7f0e"
    NOMINAL_COLOR = "#444444"

    def __init__(
        self,
        role: str,
        shield,
        live: dict,
        lookahead: float = 1.0,
        substeps: int = 20,
    ):
        self.role = role
        self.shield = shield
        self.live = live
        self.lookahead = lookahead
        self.substeps = substeps

    @staticmethod
    def _arc(pose, v: float, omega: float, horizon: float, n: int) -> np.ndarray:
        """Unicycle roll-out of a constant (v, omega) over `horizon` seconds."""
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        dt = horizon / n
        pts = [(x, y)]
        for _ in range(n):
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            theta += omega * dt
            pts.append((x, y))
        return np.array(pts)

    def __call__(self, ax, env) -> None:
        info = self.live.get("info")
        if info is None:
            return
        cfg = env.cfg
        pose = env.pursuer_pose if self.role == "pursuer" else env.evader_pose
        color = ROLE_STYLE[self.role]["color"]

        # Constraint surfaces as the shield sees them (margin included).
        for obs in env.obstacles:
            ax.add_patch(
                Circle(
                    (obs["x"], obs["y"]),
                    obs["radius"] + cfg.agent_radius + self.shield.margin,
                    fill=False,
                    edgecolor="#555555",
                    linestyle=":",
                    linewidth=1,
                    zorder=3,
                )
            )
        lim = self.shield._x_lim
        ax.plot(
            [-lim, lim, lim, -lim, -lim],
            [-lim, -lim, lim, lim, -lim],
            color="#555555",
            linestyle=":",
            linewidth=1,
            zorder=3,
        )

        # Nominal (dashed grey) vs executed (solid, role colour) arcs.
        for key, style, width, arc_color in (
            ("nominal", "--", 1.6, self.NOMINAL_COLOR),
            ("executed", "-", 2.2, color),
        ):
            v, omega = self.live[key]
            pts = self._arc(pose, v, omega, self.lookahead, self.substeps)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=arc_color,
                linestyle=style,
                linewidth=width,
                zorder=7,
            )
            head = pts[-1] - pts[-2]
            norm = float(np.hypot(*head))
            if norm > 1e-9:
                ax.add_patch(
                    FancyArrow(
                        pts[-1, 0],
                        pts[-1, 1],
                        0.12 * head[0] / norm,
                        0.12 * head[1] / norm,
                        width=0.01,
                        head_width=0.16,
                        head_length=0.16,
                        color=arc_color,
                        zorder=7,
                    )
                )

        if info["intervention"]:
            ax.add_patch(
                Circle(
                    (pose[0], pose[1]),
                    cfg.agent_radius * 2.2,
                    fill=False,
                    edgecolor=self.FLASH_COLOR,
                    linewidth=3,
                    zorder=8,
                )
            )
            ax.text(
                0.02,
                0.10,
                "SHIELD INTERVENING",
                transform=ax.transAxes,
                fontsize=9,
                color=self.FLASH_COLOR,
                fontweight="bold",
            )
        ax.text(
            0.02,
            0.03,
            f"tier {info['method']} - clearance {info['min_clearance']:+.3f} m",
            transform=ax.transAxes,
            fontsize=9,
            color=color,
        )
