from typing import Optional

import numpy as np
from scipy.special import softmax

from responserank.synthetic.util import get_log_normal_mean_sd


class UniformItemGenerator:
    def __init__(
        self,
        num_features,
        utility_function,
        utility_sd: Optional[float],
    ):
        """Initialize a uniform item generator.

        Args:
            num_features: Dimensionality of generated items.
            utility_function: Callable mapping items to utilities.
            utility_sd: Desired standard deviation for utilities. Pass ``None`` to
                disable rescaling.
        """
        self.num_features = num_features
        self.utility_function = utility_function
        self.utility_sd = utility_sd

    def generate_items(self, random_state, num_items):
        items = random_state.rand(num_items, self.num_features)
        utilities = self.utility_function(items)
        utilities, scaling_factor = self._rescale_utilities(utilities)
        return items, utilities, scaling_factor

    def _rescale_utilities(self, utilities):
        if self.utility_sd is not None:
            current_sd = np.std(utilities)
            scaling_factor = self.utility_sd / current_sd
        else:
            scaling_factor = 1
        return utilities * scaling_factor, scaling_factor


class GaussianItemGenerator:
    def __init__(self, num_features, utility_function, utility_sd):
        self.num_features = num_features
        self.utility_function = utility_function
        self.utility_sd = utility_sd

    def generate_items(self, random_state, num_items):
        items = random_state.normal(size=(num_items, self.num_features), scale=1.0)
        utilities = self.utility_function(items)
        utilities, scaling_factor = self._rescale_utilities(utilities)
        return items, utilities, scaling_factor

    def _rescale_utilities(self, utilities):
        current_sd = np.std(utilities)
        scaling_factor = self.utility_sd / current_sd
        return utilities * scaling_factor, scaling_factor


class UdiffRtRel:
    """An invertible relationship between utility and response time."""

    def udiff_to_rt(self, udiff):
        raise NotImplementedError()

    def rt_to_udiff(self, rt):
        raise NotImplementedError()


class HyperbolicUdiffRtRel(UdiffRtRel):
    def __init__(self, min_rt, max_rt):
        self.min_rt = min_rt
        self.max_rt = max_rt

    def udiff_to_rt(self, udiff):
        return self.min_rt + (self.max_rt - self.min_rt) / (udiff + 1)

    def rt_to_udiff(self, rt):
        return (self.max_rt - self.min_rt) / (rt - self.min_rt) - 1


class StochasticTrialGenerator:
    def __init__(
        self,
        udiff_rt_rel,
        response_time_sd,
        deterministic_choice: bool,
        deterministic_rt: bool,
        choice_temperature=1.0,
    ):
        self.udiff_rt_rel = udiff_rt_rel
        self.response_time_sd = response_time_sd
        self.deterministic_choice = deterministic_choice
        self.deterministic_rt = deterministic_rt
        self.choice_temperature = choice_temperature

    def generate_trials(self, u1, u2, random_state):
        y = self._generate_choices(u1, u2, random_state)
        t = self._generate_response_times(u1, u2, random_state)
        return y, t

    def _generate_choices(self, u1, u2, random_state):
        # Use softmax with temperature to calculate probabilities
        # Higher temperature (>1.0) makes choices more random
        # Lower temperature (<1.0) makes choices more deterministic
        utilities = np.column_stack([u1, u2]) / self.choice_temperature
        probabilities = softmax(utilities, axis=1)

        if np.isnan(probabilities).any():
            nan_indices = np.where(np.isnan(probabilities))
            print(
                f"WARNING: NaN values detected in probabilities at indices {nan_indices}"
            )
            print(
                f"Utilities at these indices: {utilities[np.isnan(probabilities).any(axis=1)]}"
            )
            print(
                f"Original u1, u2 at these indices: {u1[np.isnan(probabilities).any(axis=1)]}, {u2[np.isnan(probabilities).any(axis=1)]}"
            )
            print(f"Choice temperature: {self.choice_temperature}")

        binomial_p = probabilities[:, 0]  # Probability of choosing the first option

        if self.deterministic_choice:
            return binomial_p > 0.5
        else:
            return random_state.binomial(n=1, p=binomial_p, size=len(u1))

    def _generate_response_times(self, u1, u2, random_state):
        if self.deterministic_rt:
            return self.udiff_rt_rel.udiff_to_rt(np.abs(u1 - u2))
        else:
            all_means = self.udiff_rt_rel.udiff_to_rt(np.abs(u1 - u2))
            all_sds = np.ones_like(all_means) * self.response_time_sd
            mean, sd = get_log_normal_mean_sd(all_means, all_sds)
            return random_state.lognormal(mean, sd, size=len(u1))


def generate_preference_dataset(
    item_generator,
    trial_generator,
    random_state,
    num_comparisons,
    num_partitions,
    partition_rt_variability,
):
    """
    Generate a synthetic preference dataset.

    The generated preferences are based on the utility function, either
    stochastically following a Bradley-Terry model or deterministically.

    The response times are generated either deterministically or from a log-normal distribution.

    Parameters:
    -----------
    item_generator : UniformItemGenerator
        An instance of UniformItemGenerator to generate items and their utilities.
    trial_generator : StochasticTrialGenerator
        An instance of StochasticTrialGenerator to generate choices and response times.
    random_state : numpy.random.RandomState
        Random state for reproducibility.
    num_comparisons : int
        Number of comparisons to generate.
    num_partitions : int
        Number of partitions to generate.
    partition_rt_variability : float
        Amount of variability in response times between partitions. Each partition gets
        a random RT multiplier sampled from a normal distribution with mean 1.0 and
        this standard deviation.

    Returns:
    --------
    x1, x2 : numpy.ndarray
        Feature vectors for the first and second items in each comparison.
    u1, u2 : numpy.ndarray
        Utility values for the first and second items in each comparison.
    y : numpy.ndarray
        Binary choices (0 or 1) for each comparison.
    t : numpy.ndarray
        Response times for each comparison.
    scaling_factor : float
        The scaling factor applied to utilities (if any).
    partition_ids : numpy.ndarray
        Partition IDs for each comparison.

    Example:
    --------
    >>> import numpy as np
    >>> def utility_function(x):
    ...     return np.sum(x, axis=-1)
    >>> item_gen = UniformItemGenerator(2, utility_function, utility_sd=0.5)
    >>> udiff_rt_rel = HyperbolicUdiffRtRel(min_rt=0, max_rt=10)
    >>> trial_gen = StochasticTrialGenerator(
    ...     udiff_rt_rel,
    ...     1,
    ...     deterministic_choice=True,
    ...     deterministic_rt=False,
    ... )
    >>> x1, x2, u1, u2, y, t, scaling_factor, partition_ids = generate_preference_dataset(item_gen, trial_gen, np.random.RandomState(0), 10, 1, 0.0)
    >>> x1.shape
    (10, 2)
    >>> x2.shape
    (10, 2)
    >>> u1.shape
    (10,)
    >>> u2.shape
    (10,)
    >>> y.shape
    (10,)
    >>> t.shape
    (10,)
    >>> partition_ids.shape
    (10,)
    >>> bool(np.isclose(np.std(np.concatenate([u1, u2])), 0.5, atol=1e-2))
    True
    """
    total_items = num_comparisons * 2
    x, u, scaling_factor = item_generator.generate_items(random_state, total_items)

    x1, x2 = x[:num_comparisons], x[num_comparisons:]
    u1, u2 = u[:num_comparisons], u[num_comparisons:]

    y, t = trial_generator.generate_trials(u1, u2, random_state)
    partition_ids = random_state.randint(0, num_partitions, size=num_comparisons)

    # If partition_rt_variability is specified, modify response times by partition
    if partition_rt_variability > 0:
        partition_multipliers = random_state.normal(
            1.0, partition_rt_variability, size=num_partitions
        )
        # Ensure all multipliers are positive
        partition_multipliers = np.abs(partition_multipliers)

        for partition_id in range(num_partitions):
            partition_mask = partition_ids == partition_id
            t[partition_mask] *= partition_multipliers[partition_id]

    return x1, x2, u1, u2, y, t, scaling_factor, partition_ids


class DriftDiffusionTrialGenerator(StochasticTrialGenerator):
    """A trial generator based on the drift-diffusion model.

    Parameters:
    -----------
    decision_threshold : float
        The threshold at which the decision is made. Increasing this should
        increase accuracy as well as response time.
    non_decision_time : float
        The time taken to encode the stimulus and execute the response.
        Increasing this should increase response time and keep everything else
        constant.
    drift_rate_multiplier : float
        Multiplier for the drift rate. The drift rate is primarily determined by
        the utility difference, but this multiplier can be used to scale it.
        Increasing this should increase accuracy and response time (since noise
        is not scaled).
    noise_std : float
        Standard deviation of the normally distributed noise. Increasing this
        should decrease accuracy and increase response time.
    normalize_drift_rates : bool
        Whether to normalize the noise by the mean drift rate. This is useful
        to be more independend of the distribution of the drift rates.
    drift_rate_variability : float
        Variability in the drift rate.
    starting_point_variability : float
        Variability in the starting point.
    non_decision_time_variability : float
        Variability in the non-decision time.
    dt : float
        Time step for the simulation.
    """

    def __init__(
        self,
        decision_threshold,
        non_decision_time,
        normalize_drift_rate_sd: bool,
        normalize_drift_rate_mean: bool,
        drift_rate_multiplier=1.0,
        noise_std=1.0,
        drift_rate_variability=0,
        starting_point_variability=0,
        non_decision_time_variability=0,
        dt=0.001,  # Time step for simulation
    ):
        super().__init__(None, None, deterministic_choice=False, deterministic_rt=False)
        self.decision_threshold = decision_threshold
        self.non_decision_time = non_decision_time
        self.drift_rate_multiplier = drift_rate_multiplier
        self.noise_std = noise_std
        self.normalize_drift_rate_sd = normalize_drift_rate_sd
        # If all items are from the same distribution (or at least item order is
        # random), then the mean drift rate is already 0 in expectation.
        # Shifting it could falsify the results and is not recommended.
        self.normalize_drift_rate_mean = normalize_drift_rate_mean
        self.drift_rate_variability = drift_rate_variability
        self.starting_point_variability = starting_point_variability
        self.non_decision_time_variability = non_decision_time_variability
        self.dt = dt

    def generate_trials(self, u1, u2, random_state):
        drift_rates = (u1 - u2) * self.drift_rate_multiplier
        normalized_drift_rates = drift_rates
        if self.normalize_drift_rate_mean:
            normalized_drift_rates -= np.mean(drift_rates)
        if self.normalize_drift_rate_sd:
            normalized_drift_rates /= np.std(drift_rates)

        choices = []
        response_times = []

        for drift_rate in normalized_drift_rates:
            # Add variability to drift rate
            drift_rate += random_state.normal(0, self.drift_rate_variability)

            # Determine starting point
            starting_point = self.decision_threshold / 2 + random_state.uniform(
                -self.starting_point_variability, self.starting_point_variability
            )

            # Simulate the diffusion process
            time = 0
            position = starting_point
            while True:
                noise = random_state.normal(0, self.noise_std)
                # The noise term is scaled by sqrt(dt) because the variance of Brownian motion
                # scales linearly with time, so the standard deviation scales with sqrt(time)
                scaled_noise = np.sqrt(self.dt) * noise
                position += drift_rate * self.dt + scaled_noise
                time += self.dt
                if position <= 0 or position >= self.decision_threshold:
                    break

            # Add non-decision time
            ndt = self.non_decision_time + random_state.uniform(
                -self.non_decision_time_variability, self.non_decision_time_variability
            )
            rt = time + ndt

            choices.append(True if position >= self.decision_threshold else False)
            response_times.append(rt)

        return np.array(choices), np.array(response_times)
