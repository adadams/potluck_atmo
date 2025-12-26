from scipy.stats import rv_continuous, truncnorm


def construct_truncated_normal(
    mean: float,
    standard_deviation: float,
    lower_bound: float = None,
    upper_bound: float = None,
) -> rv_continuous:
    if lower_bound is None:
        lower_bound = mean - 3 * standard_deviation
    if upper_bound is None:
        upper_bound = mean + 3 * standard_deviation

    lower_bound_transformed = (lower_bound - mean) / standard_deviation
    upper_bound_transformed = (upper_bound - mean) / standard_deviation

    return truncnorm(
        lower_bound_transformed,
        upper_bound_transformed,
        loc=mean,
        scale=standard_deviation,
    )
