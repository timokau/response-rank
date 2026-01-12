import numpy as np

from responserank.synthetic.util import UtilityFunction, module_to_numpy_uf


class FixedNetUtility:
    def __init__(
        self,
        num_features,
        hidden_layers,
        normalize_on_random,
        emphasize_first_on_random,
        feature_subset_size,
        random_state,
    ):
        self.num_features = num_features
        self.hidden_layers = hidden_layers
        self.normalize_on_random = normalize_on_random
        self.emphasize_first_on_random = emphasize_first_on_random
        self.feature_subset_size = feature_subset_size
        self.random_state = random_state

        if self.feature_subset_size > 0:
            if self.feature_subset_size < 1.0:
                self.feature_subset_size = int(
                    self.feature_subset_size * self.num_features
                )
            self.feature_subset = self.random_state.choice(
                self.num_features, self.feature_subset_size, replace=False
            )
        else:
            self.feature_subset = range(self.num_features)
            self.feature_subset_size = self.num_features

        # Uses "kaiming_uniform" initialization.
        # https://discuss.pytorch.org/t/107437
        uf = UtilityFunction(
            self.feature_subset_size,
            hidden_layers=self.hidden_layers,
            dropout_rate=None,
        )
        self.fixed_nn_utility = module_to_numpy_uf(uf)

        all_xs = self.random_state.rand(1000, self.num_features)
        all_utilities = self.fixed_nn_utility(self._choose_feature_subset(all_xs))
        self.mean_utility_on_random = np.mean(all_utilities)
        self.mean_first_feature_on_random = np.mean(all_xs[:, 0])

    def _choose_feature_subset(self, x):
        return x[:, self.feature_subset]

    def __call__(self, x):
        u = self.fixed_nn_utility(self._choose_feature_subset(x))

        if self.normalize_on_random:
            u /= self.mean_utility_on_random

        if self.emphasize_first_on_random:
            normalized_first_feature = x[:, 0] / self.mean_first_feature_on_random
            if not self.normalize_on_random:
                # Scale up to match the mean utility
                normalized_first_feature *= self.mean_utility_on_random
            u = u + 2 * normalized_first_feature

        return u


class FirstFeatureUtility:
    def __init__(self, num_features, random_state):
        pass

    def __call__(self, x):
        return x[:, 0]
