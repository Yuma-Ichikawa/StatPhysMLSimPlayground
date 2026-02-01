"""
Sequence/Token Datasets.

Datasets for sequence modeling, including Markov chains, copy tasks,
Potts models, and attention-based token sequences.

All models follow the scaling convention: z = w^T x / √d = O(1).

References:
- MarkovChain/CopyTask: Induction head analysis (Edelman et al., NeurIPS 2024)
- GeneralizedPotts: Phys. Rev. Research 2024 (language-like sequences)
- TiedLowRankAttention: NeurIPS 2024 (position-semantics phase transition)
- MixedGaussianSequence: Latent cluster structure in token sequences

"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseDataset


class MarkovChainDataset(BaseDataset):
    """
    Markov Chain dataset for studying induction heads.

    Generates sequences from randomly sampled Markov chains.
    Each task has a randomly drawn transition matrix P.

    Data generation:
        1. Sample transition matrix P ~ Dirichlet(α) for each row
        2. Sample initial state x_1 ~ Uniform([S])
        3. Sample x_{t+1} ~ Categorical(P[x_t, :])

    This is the canonical setup for analyzing induction heads and
    statistical copying mechanisms in transformers.

    Args:
        d: Embedding dimension for tokens
        n_states: Number of states in Markov chain (S)
        seq_len: Sequence length (T)
        dirichlet_alpha: Concentration parameter for transition prior
        device: Device for tensors

    """

    def __init__(
        self,
        d: int,
        n_states: int = 10,
        seq_len: int = 20,
        dirichlet_alpha: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(d=d, device=device)
        self.n_states = n_states
        self.seq_len = seq_len
        self.dirichlet_alpha = dirichlet_alpha

        # Create random embeddings for each state
        self.state_embeddings = torch.randn(n_states, d, device=device)
        self.state_embeddings = self.state_embeddings / self.state_embeddings.norm(
            dim=1, keepdim=True
        )
        self.state_embeddings = self.state_embeddings * np.sqrt(d)  # ||e_s||² = d

    def _sample_transition_matrix(self, batch_size: int) -> torch.Tensor:
        """Sample transition matrices from Dirichlet prior."""
        alpha = torch.ones(self.n_states, device=self.device) * self.dirichlet_alpha
        P_np = np.random.dirichlet(alpha.cpu().numpy(), size=(batch_size, self.n_states))
        return torch.tensor(P_np, dtype=torch.float32, device=self.device)

    def _generate_sequence(self, P: torch.Tensor) -> torch.Tensor:
        """Generate a sequence from Markov chain with transition matrix P."""
        states = torch.zeros(self.seq_len, dtype=torch.long, device=self.device)
        states[0] = torch.randint(0, self.n_states, (1,), device=self.device)

        for t in range(1, self.seq_len):
            probs = P[states[t - 1]]
            states[t] = torch.multinomial(probs, 1)

        return states

    def generate_sample(self) -> tuple[torch.Tensor, ...]:
        """Generate a single Markov chain sequence."""
        P = self._sample_transition_matrix(1)[0]
        states = self._generate_sequence(P)
        tokens = self.state_embeddings[states]
        targets = states[1:]
        return tokens, states, targets, P

    def generate_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Generate a batch of Markov chain sequences."""
        P = self._sample_transition_matrix(batch_size)
        states = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=self.device)

        for b in range(batch_size):
            states[b] = self._generate_sequence(P[b])

        tokens = self.state_embeddings[states]
        targets = states[:, 1:]

        return {
            "tokens": tokens,
            "states": states,
            "targets": targets,
            "P": P,
        }

    def generate_dataset(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.generate_batch(n_samples)
        X = batch["tokens"][:, :-1].reshape(n_samples, -1)
        y = batch["targets"][:, -1].float()
        return X, y

    def get_teacher_params(self) -> dict[str, Any]:
        return {
            "n_states": self.n_states,
            "seq_len": self.seq_len,
            "dirichlet_alpha": self.dirichlet_alpha,
        }

    def __repr__(self) -> str:
        return (
            f"MarkovChainDataset(d={self.d}, n_states={self.n_states}, "
            f"seq_len={self.seq_len}, dirichlet_alpha={self.dirichlet_alpha})"
        )


class CopyTaskDataset(BaseDataset):
    """
    Copy/Induction Task dataset.

    Generates sequences where the model must copy a token that appeared
    after a specific trigger token.

    Pattern: ... [trigger] [target] ... [trigger] [?]
    The model should predict [target] when it sees [trigger] the second time.

    Args:
        d: Embedding dimension
        vocab_size: Vocabulary size
        seq_len: Total sequence length
        n_triggers: Number of trigger-target pairs in sequence
        device: Device for tensors

    """

    def __init__(
        self,
        d: int,
        vocab_size: int = 50,
        seq_len: int = 20,
        n_triggers: int = 3,
        device: str = "cpu",
    ):
        super().__init__(d=d, device=device)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_triggers = n_triggers

        self.embeddings = torch.randn(vocab_size, d, device=device)
        self.embeddings = self.embeddings / self.embeddings.norm(dim=1, keepdim=True) * np.sqrt(d)

    def generate_sample(self) -> tuple[torch.Tensor, ...]:
        """Generate a single copy task sequence."""
        batch = self.generate_batch(1)
        return (
            batch["tokens"][0],
            batch["token_ids"][0],
            batch["triggers"][0],
            batch["targets"][0],
        )

    def generate_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Generate copy task sequences."""
        tokens = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=self.device)
        triggers = torch.randint(
            0, self.vocab_size, (batch_size, self.n_triggers), device=self.device
        )
        targets = torch.randint(
            0, self.vocab_size, (batch_size, self.n_triggers), device=self.device
        )

        first_positions = torch.randint(
            0, self.seq_len // 2 - 1, (batch_size, self.n_triggers), device=self.device
        )
        second_positions = torch.randint(
            self.seq_len // 2,
            self.seq_len - 1,
            (batch_size, self.n_triggers),
            device=self.device,
        )

        for b in range(batch_size):
            for i in range(self.n_triggers):
                pos1 = first_positions[b, i].item()
                pos2 = second_positions[b, i].item()
                tokens[b, pos1] = triggers[b, i]
                tokens[b, pos1 + 1] = targets[b, i]
                tokens[b, pos2] = triggers[b, i]

        embeddings = self.embeddings[tokens]

        return {
            "tokens": embeddings,
            "token_ids": tokens,
            "triggers": triggers,
            "targets": targets,
            "first_positions": first_positions,
            "second_positions": second_positions,
        }

    def generate_dataset(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.generate_batch(n_samples)
        X = batch["tokens"].reshape(n_samples, -1)
        y = batch["targets"][:, -1].float()
        return X, y

    def get_teacher_params(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "n_triggers": self.n_triggers,
        }

    def __repr__(self) -> str:
        return (
            f"CopyTaskDataset(d={self.d}, vocab_size={self.vocab_size}, "
            f"seq_len={self.seq_len}, n_triggers={self.n_triggers})"
        )


class GeneralizedPottsDataset(BaseDataset):
    """
    Generalized Potts Model Dataset (Phys. Rev. Research 2024).

    Generates "language-like" sequences using Potts spins with position coupling.

    Data generation (Gaussianized version):
        1. Sample coupling matrix: Ω ~ GOE(L)
        2. Compute covariance: Σ = (Ω/√L + ν I)^{-1}
        3. Sample magnetization: m ~ N(0, Σ) for each sample

    For discrete Potts:
        1. Sample tokens: s_i ∈ {1,...,C} with conditional distribution
           P(s_i = α | s_{-i}) ∝ exp(β Σ_j J_{ij} (U s_j)_α)

    Args:
        d: Embedding dimension
        seq_len: Sequence length (L)
        vocab_size: Number of tokens/colors (C)
        coupling_strength: Strength of position coupling (ν)
        temperature: Inverse temperature (β)
        mode: 'gaussian' (analytically tractable) or 'discrete' (MCMC)
        device: Device for tensors

    """

    def __init__(
        self,
        d: int,
        seq_len: int = 20,
        vocab_size: int = 10,
        coupling_strength: float = 3.0,
        temperature: float = 1.0,
        mode: str = "gaussian",
        device: str = "cpu",
    ):
        super().__init__(d=d, device=device)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.coupling_strength = coupling_strength
        self.temperature = temperature
        self.mode = mode

        # Generate GOE coupling matrix
        Omega_np = np.random.randn(seq_len, seq_len)
        Omega_np = (Omega_np + Omega_np.T) / 2
        self.Omega = torch.tensor(Omega_np, dtype=torch.float32, device=device)

        # Compute covariance for Gaussianized version
        A = self.Omega / np.sqrt(seq_len) + coupling_strength * torch.eye(seq_len, device=device)
        self.Sigma = torch.inverse(A)
        self.L_chol = torch.linalg.cholesky(self.Sigma)

        # Random embedding matrix for tokens
        self.embedding = torch.randn(vocab_size, d, device=device) / np.sqrt(d)

    def generate_sample(self) -> tuple[torch.Tensor, ...]:
        if self.mode == "gaussian":
            return self._generate_gaussian_sample()
        else:
            return self._generate_discrete_sample()

    def _generate_gaussian_sample(self) -> tuple[torch.Tensor, ...]:
        z = torch.randn(self.seq_len, device=self.device)
        m = self.L_chol @ z
        x = m.unsqueeze(-1) * torch.randn(self.seq_len, self.d, device=self.device)
        x = x / np.sqrt(self.d)
        return x, m

    def _generate_discrete_sample(self, n_steps: int = 100) -> tuple[torch.Tensor, ...]:
        s = torch.randint(0, self.vocab_size, (self.seq_len,), device=self.device)

        for _ in range(n_steps):
            for i in range(self.seq_len):
                logits = torch.zeros(self.vocab_size, device=self.device)
                for alpha in range(self.vocab_size):
                    for j in range(self.seq_len):
                        if j != i:
                            J_ij = self.Omega[i, j] / np.sqrt(self.seq_len)
                            logits[alpha] += J_ij * (self.embedding[s[j]] @ self.embedding[alpha])

                probs = F.softmax(self.temperature * logits, dim=0)
                s[i] = torch.multinomial(probs, 1)

        x = self.embedding[s]
        return x, s

    def generate_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self.mode == "gaussian":
            z = torch.randn(batch_size, self.seq_len, device=self.device)
            m = z @ self.L_chol.T
            x = m.unsqueeze(-1) * torch.randn(batch_size, self.seq_len, self.d, device=self.device)
            x = x / np.sqrt(self.d)
            return {"x": x, "magnetization": m, "Sigma": self.Sigma}
        else:
            xs, ss = [], []
            for _ in range(batch_size):
                x, s = self._generate_discrete_sample()
                xs.append(x)
                ss.append(s)
            return {"x": torch.stack(xs), "tokens": torch.stack(ss)}

    def generate_dataset(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.generate_batch(n_samples)
        X = batch["x"].reshape(n_samples, -1)
        if "magnetization" in batch:
            y = batch["magnetization"][:, -1]
        else:
            y = batch["tokens"][:, -1].float()
        return X, y

    def get_teacher_params(self) -> dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
            "coupling_strength": self.coupling_strength,
            "temperature": self.temperature,
            "mode": self.mode,
        }

    def __repr__(self) -> str:
        return (
            f"GeneralizedPottsDataset(d={self.d}, seq_len={self.seq_len}, "
            f"vocab_size={self.vocab_size}, mode={self.mode})"
        )


class TiedLowRankAttentionDataset(BaseDataset):
    """
    Tied Low-Rank Self-Attention Dataset (NeurIPS 2024).

    Generates token sequences where the teacher uses attention to mix tokens.

    Data generation:
        1. Sample token embeddings: x_ℓ ~ N(0, Σ_ℓ) for ℓ = 1,...,L
        2. Teacher computes attention: y = T[x Q_* / √d] @ x

    Args:
        d: Embedding dimension
        seq_len: Sequence length (L)
        teacher_rank: Rank of teacher projection (r_t)
        position_covariances: List of covariance matrices
        attention_type: Type of attention ('softmax', 'linear', 'identity')
        temperature: Softmax temperature
        device: Device for tensors

    """

    def __init__(
        self,
        d: int,
        seq_len: int = 10,
        teacher_rank: int = 5,
        position_covariances: list[torch.Tensor] | None = None,
        attention_type: str = "linear",
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(d=d, device=device)
        self.seq_len = seq_len
        self.teacher_rank = teacher_rank
        self.attention_type = attention_type
        self.temperature = temperature

        if position_covariances is None:
            self.position_covariances = [torch.eye(d, device=device) for _ in range(seq_len)]
        else:
            self.position_covariances = [cov.to(device) for cov in position_covariances]

        self.Q_teacher = torch.randn(d, teacher_rank, device=device) / np.sqrt(d)

    def _compute_attention(self, scores: torch.Tensor) -> torch.Tensor:
        if self.attention_type == "softmax":
            return F.softmax(self.temperature * scores, dim=-1)
        elif self.attention_type == "linear":
            return scores / self.seq_len
        elif self.attention_type == "identity":
            return torch.eye(self.seq_len, device=scores.device).expand(scores.shape[0], -1, -1)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

    def generate_sample(self) -> tuple[torch.Tensor, ...]:
        x = torch.zeros(self.seq_len, self.d, device=self.device)
        for l in range(self.seq_len):
            cov = self.position_covariances[l]
            L_chol = torch.linalg.cholesky(cov)
            z = torch.randn(self.d, device=self.device)
            x[l] = L_chol @ z

        proj = x @ self.Q_teacher / np.sqrt(self.d)
        scores = proj @ proj.T
        attn = self._compute_attention(scores.unsqueeze(0)).squeeze(0)
        y = attn @ x

        return x, y, attn

    def generate_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        x = torch.zeros(batch_size, self.seq_len, self.d, device=self.device)
        for l in range(self.seq_len):
            cov = self.position_covariances[l]
            L_chol = torch.linalg.cholesky(cov)
            z = torch.randn(batch_size, self.d, device=self.device)
            x[:, l] = z @ L_chol.T

        proj = x @ self.Q_teacher / np.sqrt(self.d)
        scores = torch.bmm(proj, proj.transpose(-2, -1))
        attn = self._compute_attention(scores)
        y = torch.bmm(attn, x)

        return {"x": x, "y": y, "attention": attn, "Q_teacher": self.Q_teacher}

    def generate_dataset(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.generate_batch(n_samples)
        X = batch["x"].reshape(n_samples, -1)
        y = batch["y"].reshape(n_samples, -1)
        return X, y

    def get_teacher_params(self) -> dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "teacher_rank": self.teacher_rank,
            "attention_type": self.attention_type,
            "Q_teacher": self.Q_teacher,
        }

    def __repr__(self) -> str:
        return (
            f"TiedLowRankAttentionDataset(d={self.d}, seq_len={self.seq_len}, "
            f"teacher_rank={self.teacher_rank}, attention_type={self.attention_type})"
        )


class MixedGaussianSequenceDataset(BaseDataset):
    """
    Mixed Gaussian Sequence Dataset with latent cluster structure.

    Data generation:
        1. Sample latent cluster assignments: c = (c_1, ..., c_L) ~ ρ(c)
        2. Sample tokens: x_ℓ | c_ℓ=k ~ N(μ_{ℓ,k}, Σ_{ℓ,k})

    Args:
        d: Embedding dimension
        seq_len: Sequence length
        n_clusters: Number of clusters per position
        cluster_separation: Signal strength for cluster means
        correlation_type: How cluster assignments correlate
        device: Device for tensors

    """

    def __init__(
        self,
        d: int,
        seq_len: int = 10,
        n_clusters: int = 3,
        cluster_separation: float = 1.0,
        correlation_type: str = "independent",
        device: str = "cpu",
    ):
        super().__init__(d=d, device=device)
        self.seq_len = seq_len
        self.n_clusters = n_clusters
        self.cluster_separation = cluster_separation
        self.correlation_type = correlation_type

        self.means = []
        self.covariances = []

        for _l in range(seq_len):
            mu = torch.randn(n_clusters, d, device=device) * cluster_separation / np.sqrt(d)
            self.means.append(mu)
            cov = torch.eye(d, device=device).unsqueeze(0).expand(n_clusters, -1, -1).clone()
            self.covariances.append(cov)

        self.priors = torch.ones(n_clusters, device=device) / n_clusters

        if correlation_type == "markov":
            self.transition = torch.eye(n_clusters, device=device) * 0.7
            self.transition += 0.3 / n_clusters

    def _sample_cluster_assignments(self, batch_size: int) -> torch.Tensor:
        c = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=self.device)

        if self.correlation_type == "independent":
            for l in range(self.seq_len):
                c[:, l] = torch.multinomial(self.priors.expand(batch_size, -1), 1).squeeze(-1)
        elif self.correlation_type == "markov":
            c[:, 0] = torch.multinomial(self.priors.expand(batch_size, -1), 1).squeeze(-1)
            for l in range(1, self.seq_len):
                for b in range(batch_size):
                    c[b, l] = torch.multinomial(self.transition[c[b, l - 1]], 1)
        elif self.correlation_type == "uniform":
            cluster = torch.multinomial(self.priors.expand(batch_size, -1), 1).squeeze(-1)
            c = cluster.unsqueeze(1).expand(-1, self.seq_len)

        return c

    def generate_sample(self) -> tuple[torch.Tensor, ...]:
        c = self._sample_cluster_assignments(1).squeeze(0)
        x = torch.zeros(self.seq_len, self.d, device=self.device)

        for l in range(self.seq_len):
            k = c[l].item()
            mu = self.means[l][k]
            cov = self.covariances[l][k]
            L_chol = torch.linalg.cholesky(cov)
            z = torch.randn(self.d, device=self.device)
            x[l] = mu + L_chol @ z

        return x, c

    def generate_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        c = self._sample_cluster_assignments(batch_size)
        x = torch.zeros(batch_size, self.seq_len, self.d, device=self.device)

        for l in range(self.seq_len):
            for k in range(self.n_clusters):
                mask = c[:, l] == k
                if mask.any():
                    n_k = mask.sum().item()
                    mu = self.means[l][k]
                    cov = self.covariances[l][k]
                    L_chol = torch.linalg.cholesky(cov)
                    z = torch.randn(n_k, self.d, device=self.device)
                    x[mask, l] = mu + z @ L_chol.T

        return {"x": x, "clusters": c}

    def generate_dataset(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.generate_batch(n_samples)
        X = batch["x"].reshape(n_samples, -1)
        y = batch["clusters"][:, -1].float()
        return X, y

    def get_teacher_params(self) -> dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "n_clusters": self.n_clusters,
            "cluster_separation": self.cluster_separation,
            "correlation_type": self.correlation_type,
        }

    def __repr__(self) -> str:
        return (
            f"MixedGaussianSequenceDataset(d={self.d}, seq_len={self.seq_len}, "
            f"n_clusters={self.n_clusters}, correlation_type={self.correlation_type})"
        )
