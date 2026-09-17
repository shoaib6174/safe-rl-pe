"""Frozen, hashable configuration for pe2.

Rules (docs/pathway/from_scratch_build_plan.md, hard rule: cite a lesson or
log a decision for every parameter):
- Defaults are LOCKED to the proven values from
  docs/reference/lessons_learned.md §4. Deviations require an experiment,
  not taste.
- All configs are frozen dataclasses: immutable after creation and hashable,
  so a config can be logged / dict-keyed and any run traces to exactly one
  config+seed.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SensingConfig:
    """Sensing parameters (Step 2), applied symmetrically to both roles.

    Locked defaults, with evidence:
    - fov_angle=pi/2 (90 deg), fov_range=8.0: D18 — the proven primary
      sensing config (SP13 campaign). 3m omni abandoned (D17).
    - los_occlusion=True: D6 — combined masking (cone + range + LOS).
      LOS-only too restrictive; radius-only sees through obstacles (the
      see-through bug that must not regress).
    - Symmetric across roles: the proven SP13 primary. Asymmetric sensing
      (e.g., GI's omnidirectional evader) is a config deviation requiring a
      logged decision.
    """

    fov_angle: float = 1.5707963267948966  # pi/2
    fov_range: float = 8.0
    los_occlusion: bool = True


@dataclass(frozen=True)
class EnvConfig:
    """Environment parameters (Step 1).

    Locked defaults, with evidence:
    - dt=0.05, v_max=1.0, omega_max=2.84: proven unicycle params
      (plan Step 1; root conf/ env yaml).
    - arena 10x10, 2 obstacles: D7 (4 too cluttered, 0 removes occlusion).
    - n_physical_obstacles=None: the number of obstacles actually PLACED in
      the arena; None means "same as n_obstacles" (the locked default, and
      bit-identical to it). n_obstacles alone keeps owning the observation
      layout (pe2/obs.py reserves 2 * n_obstacles slots), so setting this
      field evaluates a frozen policy in a denser or emptier arena without
      changing its input shape (S9 attack-surface sweep).
    - evader_speed_ratio=1.2: D5 (equal speed degenerate, 1.05 insufficient).
    - agent_radius=0.15, capture at d <= 2*agent_radius: per GI 2024
      reproduction (repro_gonultas_isler_2024/config.yaml).
    - max_steps=1200 (60 s at dt=0.05), min_init_distance=3.0,
      obstacle radius (0.3, 1.0), margin 0.5: proven root-codebase env
      defaults (conf/env yaml).
    - prep_steps=0: mechanism implemented (pursuer frozen, may turn);
      off by default — the proven chassis default (train_amsdrl --prep_steps
      default 0). Step 3's viability gate decides whether the game needs it.
    """

    dt: float = 0.05
    v_max: float = 1.0
    omega_max: float = 2.84
    arena_size: float = 10.0
    n_obstacles: int = 2
    n_physical_obstacles: int | None = None
    evader_speed_ratio: float = 1.2
    agent_radius: float = 0.15
    max_steps: int = 1200
    prep_steps: int = 0
    min_init_distance: float = 3.0
    obstacle_radius_min: float = 0.3
    obstacle_radius_max: float = 1.0
    obstacle_margin: float = 0.5
    sensing: SensingConfig = field(default_factory=SensingConfig)

    @property
    def capture_distance(self) -> float:
        """Capture when center distance <= 2 * agent_radius (GI convention)."""
        return 2.0 * self.agent_radius

    @property
    def evader_v_max(self) -> float:
        """Evader top speed (D5: 1.2x pursuer)."""
        return self.evader_speed_ratio * self.v_max
