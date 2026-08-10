"""Layer 1 — Bayesian cause distributions.

Each cause owns its own marginal distribution and knows how to learn from data
and how to generate samples. Every model here is **conjugate**, so learning is
closed-form and exact — a few arithmetic updates to the hyperparameters. There
is no SVI, no gradient fitting, no optimiser. Pyro is used purely as a sampler.

| Class                      | Variable type   | Prior → likelihood            |
|----------------------------|-----------------|-------------------------------|
| ``BayesianBernoulliCause`` | binary {0, 1}   | Beta → Bernoulli              |
| ``BayesianCategoricalCause`` | K classes     | Dirichlet → Categorical       |
| ``BayesianNormalCause``    | continuous      | Normal-Inverse-Gamma → Normal |

**Posterior predictive, not posterior.** Each drawn "world" redraws the
*parameters* first and then the *value*. A world is therefore an i.i.d. draw
from the posterior predictive distribution, which carries parameter uncertainty
forward into the utility rather than freezing the parameters at a point
estimate. That is what makes the plain ``std / sqrt(n)`` standard error in
:mod:`src.causal.decision` valid.

Ported from ``causal_model.ipynb`` §1 (cells 6–16).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Hashable, Literal

import pyro
import pyro.distributions as dist
import torch

Tensor = torch.Tensor

# A decision is anything usable as a dict key. The worked problem uses 0/1/2,
# but strings would work equally well *at this layer* — the LLM-facing schema in
# models.py is what narrows it to int.
Decision = Hashable

InterventionMode = Literal["fixed_policy", "reoptimise_policy"]


class CauseDistribution(ABC):
    """Base class for a learnable marginal distribution associated with one
    independent causal variable C_j.

    Each cause must support:

    1. Updating its posterior from observations.
    2. Posterior-predictive sampling through Pyro.
    3. Returning a summary of its learned posterior.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def update(self, observations: list[Any]) -> None:
        """Update the posterior distribution from observations."""
        raise NotImplementedError

    @abstractmethod
    def pyro_sample(self, num_samples: int) -> Tensor:
        """Draw posterior-predictive samples using Pyro."""
        raise NotImplementedError

    @abstractmethod
    def posterior_summary(self) -> dict[str, Any]:
        """Return posterior hyperparameters and useful moments."""
        raise NotImplementedError

    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """Return metadata that can be shown to the LLM."""
        raise NotImplementedError


class BayesianBernoulliCause(CauseDistribution):
    """Binary causal variable.

        theta ~ Beta(alpha, beta)
        C | theta ~ Bernoulli(theta)

    After observing s ones and f zeros:

        theta | data ~ Beta(alpha + s, beta + f)
    """

    def __init__(
        self,
        name: str,
        alpha: float = 1.0,
        beta: float = 1.0,
        description: str = "",
    ) -> None:
        super().__init__(name=name, description=description)

        if alpha <= 0 or beta <= 0:
            raise ValueError("Beta prior parameters must be positive.")

        self.alpha = float(alpha)
        self.beta = float(beta)

    def update(self, observations: list[int | float]) -> None:
        if not observations:
            return

        values = torch.as_tensor(observations, dtype=torch.float32)
        valid = torch.logical_or(values == 0, values == 1)
        if not bool(valid.all()):
            raise ValueError(
                f"Cause '{self.name}' only accepts 0/1 observations."
            )

        successes = float(values.sum())
        failures = float(values.numel()) - successes

        self.alpha += successes
        self.beta += failures

    def pyro_sample(self, num_samples: int) -> Tensor:
        """For every Monte Carlo world m:

            theta^(m) ~ posterior Beta
            C^(m) ~ Bernoulli(theta^(m))
        """
        alpha = torch.tensor(self.alpha, dtype=torch.float32)
        beta = torch.tensor(self.beta, dtype=torch.float32)

        probabilities = pyro.sample(
            f"{self.name}__probability",
            dist.Beta(alpha, beta).expand([num_samples]).to_event(1),
        )
        values = pyro.sample(
            f"{self.name}__value",
            dist.Bernoulli(probabilities).to_event(1),
        )
        return values

    def posterior_summary(self) -> dict[str, Any]:
        total = self.alpha + self.beta
        posterior_mean = self.alpha / total
        posterior_variance = (
            self.alpha * self.beta / (total**2 * (total + 1.0))
        )
        return {
            "distribution": "Beta-Bernoulli",
            "alpha": self.alpha,
            "beta": self.beta,
            "posterior_probability_mean": posterior_mean,
            "posterior_probability_variance": posterior_variance,
        }

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "binary",
            "allowed_values": [0, 1],
            "description": self.description,
        }


class BayesianCategoricalCause(CauseDistribution):
    """Categorical causal variable.

        theta ~ Dirichlet(alpha)
        C | theta ~ Categorical(theta)

    Categories are represented internally by integer indexes 0, 1, ..., K - 1.
    Human-readable category labels can also be supplied.
    """

    def __init__(
        self,
        name: str,
        categories: list[str],
        concentration: list[float] | None = None,
        description: str = "",
    ) -> None:
        super().__init__(name=name, description=description)

        if len(categories) < 2:
            raise ValueError(
                "A categorical cause requires at least two categories."
            )
        if len(set(categories)) != len(categories):
            raise ValueError("Category labels must be unique.")

        self.categories = list(categories)

        if concentration is None:
            concentration = [1.0] * len(categories)
        if len(concentration) != len(categories):
            raise ValueError(
                "There must be one concentration parameter for each category."
            )
        if any(value <= 0 for value in concentration):
            raise ValueError(
                "Dirichlet concentration parameters must be positive."
            )

        self.concentration = torch.tensor(concentration, dtype=torch.float32)
        self.category_to_index = {
            category: index for index, category in enumerate(categories)
        }

    def _convert_observation(self, value: int | str) -> int:
        if isinstance(value, str):
            if value not in self.category_to_index:
                raise ValueError(
                    f"Unknown category '{value}' for cause '{self.name}'."
                )
            return self.category_to_index[value]

        index = int(value)
        if index < 0 or index >= len(self.categories):
            raise ValueError(
                f"Category index {index} is invalid for cause '{self.name}'."
            )
        return index

    def update(self, observations: list[int | str]) -> None:
        if not observations:
            return

        indexes = torch.tensor(
            [self._convert_observation(value) for value in observations],
            dtype=torch.long,
        )
        counts = torch.bincount(
            indexes, minlength=len(self.categories)
        ).float()
        self.concentration = self.concentration + counts

    def pyro_sample(self, num_samples: int) -> Tensor:
        probabilities = pyro.sample(
            f"{self.name}__probabilities",
            dist.Dirichlet(self.concentration).expand([num_samples]).to_event(1),
        )
        values = pyro.sample(
            f"{self.name}__value",
            dist.Categorical(probabilities),
        )
        return values.float()

    def posterior_summary(self) -> dict[str, Any]:
        probabilities = self.concentration / self.concentration.sum()
        return {
            "distribution": "Dirichlet-Categorical",
            "categories": self.categories,
            "concentration": self.concentration.tolist(),
            "posterior_probabilities": {
                category: float(probabilities[index])
                for index, category in enumerate(self.categories)
            },
        }

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "categorical",
            "allowed_values": list(range(len(self.categories))),
            "category_labels": {
                index: category
                for index, category in enumerate(self.categories)
            },
            "description": self.description,
        }


class BayesianNormalCause(CauseDistribution):
    """Continuous causal variable with unknown mean and variance.

        C_i | mu, sigma^2 ~ Normal(mu, sigma^2)
        sigma^2 ~ InverseGamma(alpha, beta)
        mu | sigma^2 ~ Normal(mu_0, sigma^2 / kappa)

    The class stores the current posterior hyperparameters (mu, kappa, alpha,
    beta). Updates are exact conjugate Bayesian updates; posterior-predictive
    samples are drawn through Pyro.
    """

    def __init__(
        self,
        name: str,
        mu: float = 0.0,
        kappa: float = 1.0,
        alpha: float = 2.0,
        beta: float = 2.0,
        description: str = "",
    ) -> None:
        super().__init__(name=name, description=description)

        if kappa <= 0:
            raise ValueError("kappa must be positive.")
        if alpha <= 0 or beta <= 0:
            raise ValueError("Inverse-Gamma parameters must be positive.")

        self.mu = float(mu)
        self.kappa = float(kappa)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def update(self, observations: list[int | float]) -> None:
        if not observations:
            return

        values = torch.as_tensor(observations, dtype=torch.float32)
        n = int(values.numel())
        sample_mean = values.mean()
        centred_sum_squares = ((values - sample_mean) ** 2).sum()

        old_mu = torch.tensor(self.mu, dtype=torch.float32)
        old_kappa = torch.tensor(self.kappa, dtype=torch.float32)
        old_alpha = torch.tensor(self.alpha, dtype=torch.float32)
        old_beta = torch.tensor(self.beta, dtype=torch.float32)

        new_kappa = old_kappa + n
        new_mu = (old_kappa * old_mu + n * sample_mean) / new_kappa
        new_alpha = old_alpha + n / 2.0
        mean_difference_adjustment = (
            old_kappa * n * (sample_mean - old_mu) ** 2 / (2.0 * new_kappa)
        )
        new_beta = (
            old_beta + 0.5 * centred_sum_squares + mean_difference_adjustment
        )

        self.mu = float(new_mu)
        self.kappa = float(new_kappa)
        self.alpha = float(new_alpha)
        self.beta = float(new_beta)

    def pyro_sample(self, num_samples: int) -> Tensor:
        """For each simulated world:

            sigma^2 ~ InverseGamma(alpha, beta)
            mu ~ Normal(posterior_mu, sqrt(sigma^2 / kappa))
            C ~ Normal(mu, sqrt(sigma^2))
        """
        alpha = torch.tensor(self.alpha, dtype=torch.float32)
        beta = torch.tensor(self.beta, dtype=torch.float32)
        posterior_mu = torch.tensor(self.mu, dtype=torch.float32)
        kappa = torch.tensor(self.kappa, dtype=torch.float32)

        variance = pyro.sample(
            f"{self.name}__variance",
            dist.InverseGamma(alpha, beta).expand([num_samples]).to_event(1),
        )
        sampled_mean = pyro.sample(
            f"{self.name}__mean",
            dist.Normal(
                posterior_mu.expand(num_samples),
                torch.sqrt(variance / kappa),
            ).to_event(1),
        )
        values = pyro.sample(
            f"{self.name}__value",
            dist.Normal(sampled_mean, torch.sqrt(variance)).to_event(1),
        )
        return values

    def posterior_summary(self) -> dict[str, Any]:
        if self.alpha > 1:
            expected_variance = self.beta / (self.alpha - 1.0)
        else:
            expected_variance = math.inf

        return {
            "distribution": "Normal-Inverse-Gamma",
            "posterior_mean": self.mu,
            "kappa": self.kappa,
            "alpha": self.alpha,
            "beta": self.beta,
            "expected_sampling_variance": expected_variance,
        }

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "continuous",
            "description": self.description,
        }


@dataclass(frozen=True)
class Intervention:
    """Represents a query involving ``do(variable = value)``.

    baseline_value:
        If supplied, compare do(variable=value) against do(variable=baseline_value).
        If absent, compare the intervention against the ordinary observational
        model where that cause remains stochastic.

    mode:
        fixed_policy — hold D fixed.
        reoptimise_policy — recalculate the optimal action under each condition.
    """

    variable: str
    value: float
    baseline_value: float | None = None
    mode: InterventionMode = "reoptimise_policy"
    fixed_decision: Decision | None = None
