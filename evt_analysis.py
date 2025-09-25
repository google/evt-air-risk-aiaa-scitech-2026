"""Library for Extreme Value Theory (EVT) analysis of Closest Point of Approach (CPA).

This library provides tools to estimate the probability of collisions based on
observed CPA values using the Peaks Over Threshold (POT) method from EVT. It
includes functions for:

*   Data preparation and visualization.
*   Mean excess plot generation to validate the POT method.
*   Estimation of the Generalized Pareto Distribution (GPD) parameters.
*   Calculation of collision probabilities and confidence intervals.
*   Threshold stability analysis.

Copyright 2025 Wing Aviation LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, TypedDict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import optimize
from scipy import stats


pio.templates.default = "plotly_white"
pio.renderers.default = "plotly_mimetype+notebook_connected"

colorway = list(pio.templates["plotly_white"].layout.colorway)
colorway[0] = "#0a74ff"
pio.templates["plotly_white"].layout.colorway = tuple(colorway)

# Constants
_OPTIMIZATION_DELTA = 1e-7
_N_SAMPLES = 1_000_000
_DEFAULT_MIN_SAMPLES_IN_TAIL = 200
_DEFAULT_THRESHOLD_INTERVAL = 1.0
_DEFAULT_COLLISION_DISTANCE = 10.0


# Define a TypedDict for the result of the EVT POT pipeline
class EvtPotPipelineResult(TypedDict):
  """Structure of the dictionary returned by evt_pot_pipeline."""

  threshold: float
  n_tail: int
  p_threshold: float
  xi_hat: float
  xi_se: float
  p_collision: float
  p_collision_upper: float
  p_collision_lower: float
  error_plus: float
  error_minus: float
  exceedence_samples: np.ndarray
  mean_excess: float
  tail_data: pd.Series
  exceedence_collision: float


def _round_to_interval(
    number: float | np.ndarray, interval: float
) -> float | np.ndarray:
  """Rounds a given number to the nearest multiple of a specified interval using NumPy.

  Args:
      number (float or np.ndarray): The number(s) to be rounded. Can be a single
        float or a NumPy array.
      interval (float): The interval to which the number(s) should be rounded.
        Must be a positive float.

  Returns:
      float or np.ndarray: The rounded number(s).
  """
  if interval <= 0:
    raise ValueError("Interval must be a positive number.")

  # Divide the number by the interval, round to the nearest integer,
  # and then multiply by the interval.
  # np.round handles both single numbers and arrays correctly.
  rounded_number = np.round(number / interval) * interval
  return rounded_number


class ExtremeValuesAnalysis:
  """Class to analyze extreme values of CPA and collision risk.

  Attributes:
    cpa_values_df: Sequence of closest point of approach (CPA) values.
    results_df: Pandas DataFrame containing the analysis results.
    fig_cpa_dist: plotly.graph_objects.Figure - The figure object containing the
      CPA distribution plot.
    fig_cpa_neg_dist: plotly.graph_objects.Figure - The figure object containing
      the negative CPA distribution plot.
    fig_mean_excess: plotly.graph_objects.Figure - The figure object containing
      the mean excess plot.
    fig_xi_vs_threshold: plotly.graph_objects.Figure - The figure object
      containing the threshold stability plot for the estimated ξ parameter.
    fig_pmac_vs_threshold: plotly.graph_objects.Figure - The figure object
      containing the threshold stability plot for the estimated collision
      probability.
  """

  def __init__(
      self,
      cpa_values: list[float],
      collision_radius: float = _DEFAULT_COLLISION_DISTANCE,
      min_samples_in_tail: int = _DEFAULT_MIN_SAMPLES_IN_TAIL,
      threshold_interval: float = _DEFAULT_THRESHOLD_INTERVAL,
  ):
    """Initializes the ExtremeValuesAnalysis class.

    Args:
      cpa_values: A sequence of CPA values.
      collision_radius: The radius used to define a collision.
      min_samples_in_tail: The minimum number of samples required in the tail.
      threshold_interval: How finely to sample the threshold range.
    """
    self.cpa_values_df = pd.DataFrame({"cpa": cpa_values})
    if self.cpa_values_df.empty:
      raise ValueError("CPA values cannot be empty.")

    if collision_radius <= 0:
      raise ValueError("Collision radius must be positive.")

    # Find the minimum threshold to ensure that the tail data has at least
    # _MIN_SAMPLES_IN_TAIL samples.
    min_threshold = _round_to_interval(
        np.sort(np.array(cpa_values))[min_samples_in_tail] + threshold_interval,
        threshold_interval,
    )

    # Select the max threshold as the 99.5th percentile of the negative CPAs
    # (0.5th percentile of the CPAs).  If that doesn't allow enough samples in
    # the tail, use double the minimum threshold.
    max_threshold = _round_to_interval(
        np.percentile(np.array(cpa_values), 0.5), threshold_interval
    )
    if max_threshold < 2.0 * min_threshold:
      max_threshold = 2.0 * min_threshold

    thresholds = np.arange(min_threshold, max_threshold, threshold_interval)
    self.results_df = pd.DataFrame()

    self.fig_cpa_dist = px.histogram(
        self.cpa_values_df.query("cpa < " + str(max_threshold)),
        x="cpa",
        title="Observed CPA Distribution",
        labels={"cpa": "CPA distance per run [m]"},
        range_x=[0, max_threshold],
    )

    self.cpa_values_df["cpa_neg"] = -self.cpa_values_df["cpa"]
    self.fig_cpa_neg_dist = px.histogram(
        self.cpa_values_df.query("cpa_neg > " + str(-max_threshold)),
        x="cpa_neg",
        title="Negative CPA Distribution",
        labels={"cpa_neg": "-CPA distance per run [m]"},
        range_x=[-max_threshold, 0],
    )

    # Loop over candidate thresholds.
    # `results` is a dictionary of threshold values and their corresponding
    # POT results.
    results = {}
    for cand in thresholds:
      try:
        res = self.evt_pot_pipeline(
            self.cpa_values_df["cpa_neg"],
            threshold_cpa_neg=-cand,
            collision_dist_neg=-collision_radius,
        )
        results[-cand] = res
      except ValueError as e:
        print(f"Threshold {cand} failed: {e}")

    self.fig_mean_excess = self.mean_excess_plot(results)
    self.results_df = pd.DataFrame.from_dict(results, orient="index")
    self.results_df.reset_index(inplace=True)
    self.results_df.rename(columns={"index": "threshold_cpa_neg"}, inplace=True)

    self.fig_xi_vs_threshold = px.scatter(
        self.results_df,
        x="threshold_cpa_neg",
        y="xi_hat",
        error_y="xi_se",
        labels={"threshold": "Effective Threshold", "xi_hat": "Estimated ξ"},
        title="Threshold Stability Plot: Estimated ξ vs. Effective Threshold T",
    )
    self.fig_xi_vs_threshold.update_traces(mode="markers+lines")

    self.fig_pmac_vs_threshold = px.scatter(
        self.results_df,
        x="threshold_cpa_neg",
        y="p_collision",
        log_y=True,
        error_y="error_plus",
        error_y_minus="error_minus",
        labels={
            "threshold": "Effective Threshold",
            "p_collision": "Estimated Collision Probability",
        },
        title=(
            "Threshold Stability Plot: Estimated Collision Probability vs."
            " Effective Threshold"
        ),
    )
    self.fig_pmac_vs_threshold.update_yaxes(tickformat=".1e")
    self.fig_pmac_vs_threshold.update_traces(mode="markers+lines")

  def evt_pot_pipeline(
      self,
      cpa_neg: pd.Series,
      threshold_cpa_neg: float,
      collision_dist_neg: float,
  ) -> EvtPotPipelineResult:
    """Estimate the collision probability via EVT POT.

    Args:
      cpa_neg: A pandas Series of negative CPA values.
      threshold_cpa_neg: The threshold in the cpa_neg scale.
      collision_dist_neg: The negative distance value defining a collision.

    Returns:
      A dictionary containing the results of the POT analysis, including:
        - threshold: The positive threshold used in the transformation.
        - n_tail: Number of tail observations.
        - mean_excess: Mean excess of tail observations.
        - p_threshold: Exceedance probability.
        - xi_hat: Estimated shape parameter.
        - xi_se: Standard error of xi_hat.
        - p_collision: Overall collision probability.
        - p_collision_upper: Upper bound of 95% CI for collision probability.
        - p_collision_lower: Lower bound of 95% CI for collision probability.
        - exceedence_samples: The transformed exceedances.
        - tail_data: The original tail data from cpa_neg.
        - exceedence_collision: The transformed collision threshold.
    """
    # Define the positive threshold T from the negative threshold
    threshold = -threshold_cpa_neg  # T is positive

    # Select tail data: observations where cpa_neg > threshold_cpa_neg
    tail_data = cpa_neg[cpa_neg > threshold_cpa_neg]
    # Transform the tail data: y = T + cpa_neg
    exceedence_samples = threshold + tail_data.values

    # Compute exceedance probability
    n_total = len(cpa_neg)
    n_tail = len(tail_data)
    mean_excess = exceedence_samples.mean()
    p_threshold = n_tail / n_total if n_total > 0 else np.nan
    p_threshold_se = np.sqrt(p_threshold * (1 - p_threshold) / n_total)

    xi_hat, xi_se = self._estimate_gpd_parameters(exceedence_samples, threshold)

    # Define the collision event in the transformed scale
    exceedence_collision = threshold + collision_dist_neg

    # Compute the conditional collision probability using the GPD survival
    # function
    p_collision_cond = (1 - exceedence_collision / threshold) ** (-1 / xi_hat)

    # Compute the overall collision probability
    p_collision = p_threshold * p_collision_cond

    xi_samps = stats.norm.rvs(xi_hat, xi_se, size=_N_SAMPLES)
    p_threshold_samples = stats.norm.rvs(
        p_threshold, p_threshold_se, size=_N_SAMPLES
    )
    # only keep the samples where xi_samps < 0
    xi_samps = xi_samps[xi_samps < 0]
    p_collision_samples = (1 - exceedence_collision / threshold) ** (
        -1 / xi_samps
    ) * p_threshold_samples
    p_collision_ci = np.percentile(p_collision_samples, [2.5, 97.5])

    return {
        "threshold": threshold,
        "n_tail": n_tail,
        "p_threshold": p_threshold,
        "xi_hat": xi_hat,
        "xi_se": xi_se,
        "p_collision": p_collision,
        "p_collision_upper": p_collision_ci[1],
        "p_collision_lower": p_collision_ci[0],
        "error_plus": p_collision_ci[1] - p_collision,
        "error_minus": p_collision - p_collision_ci[0],
        "exceedence_samples": exceedence_samples,  # Transformed exceedances
        "mean_excess": mean_excess,
        "tail_data": tail_data,
        "exceedence_collision": exceedence_collision,
    }

  def _estimate_gpd_parameters(
      self, exceedence_samples: np.ndarray, threshold: float
  ) -> tuple[float, float]:
    """Estimates the GPD parameters using maximum likelihood estimation.

    Args:
      exceedence_samples: A numpy array of exceedance samples.
      threshold: The threshold value.

    Returns:
      A tuple containing the estimated xi and its standard error.
    """

    def neg_log_likelihood(xi, samples, threshold):
      # xi must be negative
      if xi >= 0:
        return np.inf
      # Ensure 1 - sample/threshold > 0 for all observations
      if np.any(1 - samples / threshold <= 0):
        return np.inf
      n = len(samples)
      return n * np.log(-xi * threshold) + (1 / xi + 1) * np.sum(
          np.log(1 - samples / threshold)
      )

    # Initial guess for xi (must be negative)
    xi0 = np.array([-0.2])

    # Optimize to estimate xi by minimizing the negative log-likelihood
    res = optimize.minimize(
        lambda xi: neg_log_likelihood(xi, exceedence_samples, threshold),
        xi0,
        method="BFGS",
        bounds=[(-2, -1e-2)],
    )

    if not res.success:
      try:
        res = optimize.minimize(
            lambda xi: neg_log_likelihood(xi[0], exceedence_samples, threshold),
            xi0,
            bounds=[(-2, -1e-2)],
        )
        xi_hat = res.x[0]
        # Estimate the standard error of xi using finite differences
        f = lambda xi_val: neg_log_likelihood(
            xi_val, exceedence_samples, threshold
        )
        f_plus = f(xi_hat + _OPTIMIZATION_DELTA)
        f_minus = f(xi_hat - _OPTIMIZATION_DELTA)
        f_current = f(xi_hat)
        second_deriv = (
            f_plus - 2 * f_current + f_minus
        ) / _OPTIMIZATION_DELTA**2
        if second_deriv <= 0:
          raise RuntimeError(
              "Non-positive second derivative, cannot compute standard error."
          )
        xi_se = np.sqrt(1 / second_deriv)
        if not res.success:
          raise RuntimeError("Optimization failed: " + res.message)
      except Exception as e:
        raise RuntimeError(f"Optimization failed: {e}") from e
    else:
      xi_hat = res.x[0]
      xi_se = np.sqrt(res.hess_inv[0, 0])

    return xi_hat, xi_se

  def mean_excess_plot(
      self,
      pot_results: dict[float, dict[str, Any]],
  ) -> go.Figure:
    """Plot the mean excess function for a given series of threshold values.

    Args:
        pot_results: dict - The output of the POT pipeline.

    Returns:
        fig: plotly.graph_objects.Figure - The figure object containing the mean
        excess plot.
    """
    thresholds = [-x["threshold"] for x in pot_results.values()]
    mean_excess = [x["mean_excess"] for x in pot_results.values()]
    tail_observations = [x["n_tail"] for x in pot_results.values()]
    fig = px.scatter(
        x=thresholds,
        y=mean_excess,
        color=tail_observations,
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Mean Excess Plot",
        labels={
            "x": "Threshold",
            "y": "Mean Excess",
            "color": "Number of Tail Observations",
        },
    )
    return fig

  def results_at_threshold(
      self, selected_threshold: float
  ) -> EvtPotPipelineResult:
    """Get the analysis results for the threshold closest to the selected value.

    This method finds the threshold in the pre-computed `results_df` that is
    numerically closest to the `selected_threshold` provided by the user.
    It then returns the full set of analysis results associated with that
    nearest threshold.

    Args:
      selected_threshold: The desired threshold for which to retrieve results.

    Returns:
      An EvtPotPipelineResult dictionary containing the analysis results
      for the threshold closest to `selected_threshold`.
    """
    nearest_key = min(
        self.results_df["threshold"], key=lambda x: abs(x - selected_threshold)
    )
    print(
        f"Requested threshold: {selected_threshold}, nearest_key: {nearest_key}"
    )

    selected_result = (
        self.results_df[self.results_df["threshold"] == nearest_key]
        .iloc[0]
        .to_dict()
    )
    return selected_result

  def pdf_fit_plot(self, selected_threshold: float) -> go.Figure:
    """Plots the PDF of the exceedances over a given threshold.

    Args:
        selected_threshold: float - The threshold value to use for the PDF fit.

    Returns:
        fig_pdf_fit: plotly.graph_objects.Figure - The figure object containing
        the PDF fit plot.
    """
    selected_result = self.results_at_threshold(selected_threshold)

    y = selected_result["exceedence_samples"]
    fig_pdf_fit = px.histogram(
        x=y,
        labels={"x": "Exceedance"},
        title="Exceedences over threshold probability density",
        histnorm="probability density",
        marginal="rug",
    )
    fig_pdf_fit.data[0].name = "Sample probability density"

    y_fit = np.linspace(y.min(), selected_result["threshold"])
    pdf = (
        1.0 / (-selected_result["xi_hat"] * selected_result["threshold"])
    ) * np.power(
        (1.0 - y_fit / selected_result["threshold"]),
        (-1.0 - 1.0 / selected_result["xi_hat"]),
    )
    fig_pdf_fit.add_trace(
        go.Scatter(
            x=y_fit,
            y=pdf,
            name="Fit GPD function",
        ),
    )
    fig_pdf_fit.update_xaxes(range=[0.0, selected_threshold])
    return fig_pdf_fit

  def qq_plot(self, selected_threshold: float) -> go.Figure:
    """Generates a Quantile-Quantile (QQ) plot for the GPD fit.

    The QQ plot compares the empirical quantiles of the exceedance data
    against the theoretical quantiles derived from the fitted Generalized
    Pareto Distribution (GPD) for a selected threshold. This plot helps
    to visually assess the goodness-of-fit of the GPD model.

    Args:
        selected_threshold: The threshold value for which to generate the QQ
          plot. The method will find the closest available threshold in the
          analysis results.

    Returns:
        A Plotly graph object figure representing the QQ plot.
    """
    # Sort the observed exceedances (y) and compute empirical probabilities
    selected_result = self.results_at_threshold(selected_threshold)

    y = selected_result["exceedence_samples"]
    y_sorted = np.sort(y)
    n = len(y_sorted)
    p_emp = (np.arange(1, n + 1) - 0.5) / n

    # Compute theoretical quantiles for the GPD model
    q_theoretical = selected_result["threshold"] * (
        1.0 - (1.0 - p_emp) ** (-selected_result["xi_hat"])
    )

    # Create a DataFrame for plotting
    df_qq = pd.DataFrame({
        "Theoretical Quantiles": q_theoretical,
        "Empirical Quantiles": y_sorted,
    })

    # Generate the QQ plot
    fig_qq = px.scatter(
        df_qq,
        x="Theoretical Quantiles",
        y="Empirical Quantiles",
        title="QQ-Plot for GPD Fit",
        labels={
            "Theoretical Quantiles": "Theoretical Quantiles (GPD Model)",
            "Empirical Quantiles": "Empirical Quantiles (Data)",
        },
    )

    # Add a 45-degree reference line
    min_val = min(q_theoretical.min(), y_sorted.min())
    max_val = max(q_theoretical.max(), y_sorted.max())
    fig_qq.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="red", dash="dash"),
    )

    return fig_qq
