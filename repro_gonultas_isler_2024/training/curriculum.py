class Curriculum:
    def __init__(
        self,
        total_episodes: int = 20000,
        t_end_ratio: float = 0.3,
        fov_start: float = 6.283,
        fov_end: float = 2.094,
        evader_vmax_start: float = 0.0,
        evader_vmax_end: float = 1.5,
    ):
        self.total_episodes = total_episodes
        self.t_end = int(t_end_ratio * total_episodes)
        self.fov_start = fov_start
        self.fov_end = fov_end
        self.evader_vmax_start = evader_vmax_start
        self.evader_vmax_end = evader_vmax_end

    def _progress(self, episode: int) -> float:
        return min(1.0, episode / self.t_end)

    def get_fov(self, episode: int) -> float:
        p = self._progress(episode)
        return self.fov_start * (1 - p) + self.fov_end * p

    def get_evader_vmax(self, episode: int) -> float:
        p = self._progress(episode)
        return self.evader_vmax_start * (1 - p) + self.evader_vmax_end * p
