"""Sanity checks for fitting functions. This is not intended to test performance."""

import numpy as np

from responserank.synthetic.evaluation import pearson_utility_correlation
from responserank.synthetic.fitting import (
    BaseUtilityFunctionFitter,
    BTUtilityFunctionFitter,
)
from responserank.synthetic.losses import (
    DeterministicRTLoss,
    RTRankAnchoredLoss,
)
from responserank.synthetic.metric_tracker import MetricTracker
from responserank.synthetic.synthetic_data import (
    HyperbolicUdiffRtRel,
    StochasticTrialGenerator,
    UniformItemGenerator,
    generate_preference_dataset,
)
from responserank.synthetic.util import (
    UtilityFunction,
    dataset_to_dataframe,
    evaluate_utility_function,
    module_to_numpy_uf,
)

DEFAULT_DROPOUT_RATE = None
DEFAULT_HIDDEN_LAYERS = [10]


def fit_and_evaluate_model(
    train_df,
    test_df,
    num_features,
    fitter_class,
    num_epochs=100,
    fitter_kwargs=None,
    fit_kwargs=None,
):
    fitter_kwargs = fitter_kwargs or {}
    fit_kwargs = fit_kwargs or {}
    fit_kwargs.setdefault("early_stopping_patience", None)
    utility_function = UtilityFunction(
        num_features,
        hidden_layers=DEFAULT_HIDDEN_LAYERS,
        dropout_rate=DEFAULT_DROPOUT_RATE,
    )
    metric_tracker = MetricTracker(metric_prefix="", backend=None)
    fitter = fitter_class(
        utility_function, metric_tracker=metric_tracker, **fitter_kwargs
    )
    fitted_uf = fitter.fit(train_df, test_df, num_epochs=num_epochs, **fit_kwargs)
    fitted_numpy_uf = module_to_numpy_uf(fitted_uf)
    accuracy = evaluate_utility_function(test_df, fitted_numpy_uf)
    correlation = pearson_utility_correlation(test_df, fitted_numpy_uf)
    return fitted_numpy_uf, accuracy, correlation


def generate_and_split_data(
    num_features,
    num_train_samples,
    num_test_samples,
    true_utility,
    response_time_sd,
    udiff_rt_rel,
    deterministic_choice,
    deterministic_rt,
):
    num_samples = num_train_samples + num_test_samples
    item_generator = UniformItemGenerator(num_features, true_utility, utility_sd=None)
    trial_generator = StochasticTrialGenerator(
        udiff_rt_rel,
        response_time_sd=response_time_sd,
        deterministic_choice=deterministic_choice,
        deterministic_rt=deterministic_rt,
    )
    x1, x2, u1, u2, y, t, _, _ = generate_preference_dataset(
        item_generator,
        trial_generator,
        np.random.RandomState(42),
        num_samples,
        num_partitions=1,
        partition_rt_variability=0.0,
    )
    df = dataset_to_dataframe(x1, x2, u1, u2, y, t, _)
    return df.iloc[:num_train_samples], df.iloc[num_train_samples:]


def test_fit_utility_function_bt_simple():
    np.random.seed(42)
    num_features = 3
    num_train_samples = 20
    num_test_samples = 100

    def true_utility(x):
        return np.sum(x, axis=1)

    udiff_rt_rel = HyperbolicUdiffRtRel(min_rt=0, max_rt=10)
    df_train, df_test = generate_and_split_data(
        num_features,
        num_train_samples,
        num_test_samples,
        true_utility,
        response_time_sd=0.1,
        deterministic_choice=True,
        deterministic_rt=False,
        udiff_rt_rel=udiff_rt_rel,
    )

    fitted_numpy_uf, accuracy, correlation = fit_and_evaluate_model(
        df_train,
        df_test,
        num_features,
        BTUtilityFunctionFitter,
    )

    assert accuracy > 0.7, f"Expected accuracy > 0.7, but got {accuracy}"
    assert correlation > 0.7, f"Expected correlation > 0.7, but got {correlation}"


def test_fit_utility_function_anchored_rr():
    np.random.seed(42)
    num_features = 3
    num_train_samples = 20
    num_test_samples = 100

    def true_utility(x):
        return np.sum(x, axis=1)

    true_response_time_sd = 0.05

    udiff_rt_rel = HyperbolicUdiffRtRel(min_rt=0, max_rt=10)
    df_train, df_test = generate_and_split_data(
        num_features,
        num_train_samples,
        num_test_samples,
        true_utility,
        true_response_time_sd,
        deterministic_choice=True,
        deterministic_rt=False,
        udiff_rt_rel=udiff_rt_rel,
    )

    fitted_numpy_uf, accuracy, correlation = fit_and_evaluate_model(
        df_train,
        df_test,
        num_features,
        BaseUtilityFunctionFitter,
        fitter_kwargs=dict(
            loss=RTRankAnchoredLoss(
                unreduce=False,
                misclassification_penalty=False,
                use_strata=False,
                worth_transform="exp",
                mean_reduce=True,
            ),
            learning_rate=0.01,
        ),
    )

    assert accuracy > 0.7, f"Expected accuracy > 0.7, but got {accuracy}"
    assert correlation > 0.7, f"Expected correlation > 0.7, but got {correlation}"


def test_fit_utility_function_rt_regression():
    np.random.seed(42)
    num_features = 3
    num_train_samples = 20
    num_test_samples = 100

    def true_utility(x):
        return np.sum(x, axis=1)

    true_response_time_sd = 0.05

    udiff_rt_rel = HyperbolicUdiffRtRel(min_rt=0, max_rt=10)
    df_train, df_test = generate_and_split_data(
        num_features,
        num_train_samples,
        num_test_samples,
        true_utility,
        true_response_time_sd,
        deterministic_choice=True,
        deterministic_rt=False,
        udiff_rt_rel=udiff_rt_rel,
    )

    fitted_numpy_uf, accuracy, correlation = fit_and_evaluate_model(
        df_train,
        df_test,
        num_features,
        BaseUtilityFunctionFitter,
        fitter_kwargs=dict(
            loss=DeterministicRTLoss(
                assumed_udiff_rt_rel=udiff_rt_rel,
            ),
            learning_rate=0.01,
        ),
    )

    assert accuracy > 0.7, f"Expected accuracy > 0.7, but got {accuracy}"
    assert correlation > 0.7, f"Expected correlation > 0.7, but got {correlation}"

    # Check if the response times are reasonably close
    x1 = df_test[[f"x1_{i}" for i in range(num_features)]].values
    x2 = df_test[[f"x2_{i}" for i in range(num_features)]].values
    t = df_test["t"].values
    fitted_udiff = np.abs(fitted_numpy_uf(x1) - fitted_numpy_uf(x2))
    fitted_rt = udiff_rt_rel.udiff_to_rt(fitted_udiff)
    rt_correlation = np.corrcoef(t, fitted_rt)[0, 1]
    assert rt_correlation > 0.7, (
        f"Expected RT correlation > 0.7, but got {rt_correlation}"
    )
