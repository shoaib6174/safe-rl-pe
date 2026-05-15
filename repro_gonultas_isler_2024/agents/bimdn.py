import torch
import torch.nn as nn


class BiMDN(nn.Module):
    def __init__(
        self,
        obs_dim: int = 11,
        hidden_dim: int = 128,
        feature_dim: int = 32,
        num_components: int = 3,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_components = num_components

        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(2 * hidden_dim, feature_dim)
        self.relu = nn.ReLU()

        # Intermediate FC layers for each head (per paper Fig 4)
        self.pi_fc = nn.Linear(feature_dim, feature_dim)
        self.mu_fc = nn.Linear(feature_dim, feature_dim)
        self.sigma_fc = nn.Linear(feature_dim, feature_dim)

        self.pi_head = nn.Linear(feature_dim, num_components)
        self.mu_head = nn.Linear(feature_dim, num_components * 2)
        self.sigma_head = nn.Linear(feature_dim, num_components * 2)

    def forward(self, obs_history: torch.Tensor) -> tuple:
        lstm_out, _ = self.lstm(obs_history)
        last_hidden = lstm_out[:, -1, :]
        features = self.relu(self.fc(last_hidden))

        pi = torch.softmax(self.pi_head(self.relu(self.pi_fc(features))), dim=-1)
        mu = self.mu_head(self.relu(self.mu_fc(features))).view(-1, self.num_components, 2)

        sigma = nn.functional.elu(self.sigma_head(self.relu(self.sigma_fc(features)))) + 1.0
        sigma = sigma.view(-1, self.num_components, 2)

        return pi, mu, sigma

    def compute_nll_loss(
        self,
        obs_history: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        pi, mu, sigma = self.forward(obs_history)
        target = target_pos.unsqueeze(1)

        log_prob = (
            -0.5 * torch.log(2 * torch.tensor(3.14159265))
            - torch.log(sigma)
            - 0.5 * ((target - mu) / sigma) ** 2
        )
        log_prob = log_prob.sum(dim=-1)

        log_prob_weighted = torch.logsumexp(
            torch.log(pi + 1e-8) + log_prob, dim=-1
        )

        return -log_prob_weighted.mean()

    def get_weighted_mean(self, obs_history: torch.Tensor) -> torch.Tensor:
        pi, mu, _ = self.forward(obs_history)
        weighted_mean = (pi.unsqueeze(-1) * mu).sum(dim=1)
        return weighted_mean
