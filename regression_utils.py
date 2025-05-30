import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

from typing import Iterator


def filter_df(df: pl.DataFrame, col_substrings: list[str], time_offset: float) -> pl.DataFrame:
    selected_cols = ["time"] + [col for col in df.columns if any(sub in col for sub in col_substrings)]
    
    filtered_cols_df = df.select(selected_cols)
    filtered_rows_df = filtered_cols_df.filter(pl.col("time") >= time_offset)

    result_df = filtered_rows_df.with_columns((pl.col("time") - time_offset).alias("time"))

    return result_df


def fit_single_k(time: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    A = values.max()

    def model(t, k):
        return A * (1 - np.exp(-k * t))

    def loss(k):
        y_pred = model(time, k)
        return np.sum((values - y_pred) ** 2)

    result = minimize_scalar(loss, bounds=(1e-6, 10), method="bounded")
    k = result.x

    y_pred = model(time, k)
    residuals = values - y_pred

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)

    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals ** 2))

    return A, k, r_squared, rmse


def yield_study_groups(df: pl.DataFrame, group_size) -> Iterator[tuple[pl.DataFrame, list[str], str]]:
    cols = df.columns
    time_col = cols[0]
    data_cols = cols[1:]

    for i in range(0, len(data_cols), group_size):
        group_cols = data_cols[i:i + group_size]
        group_df = df.select([time_col] + group_cols)

        group_name = group_cols[0].split(" ")[0]

        yield group_df, group_cols, group_name


def build_k_df(df: pl.DataFrame, fitting_cols) -> pl.DataFrame:
    study_rows = []
    param_names = None
    a_cols = [col + " A" for col in fitting_cols]
    k_cols = [col + " k" for col in fitting_cols]


    for group_df, group_cols, group_name in yield_study_groups(df, len(k_cols)):
        parts = group_name.split("_")
        params = {}
        for i in range(0, len(parts) - 1, 2):
            param_name = parts[i]
            param_value = float(parts[i + 1])
            params[param_name] = param_value

        if param_names is None:
            param_names = list(params.keys())

        time = group_df["time"].to_numpy()
        row = [group_name]

        row.extend(params[name] for name in param_names)

        r2_list = []
        rmse_list = []

        for col in group_cols:
            values = group_df[col].to_numpy()
            A, k, r2, rmse = fit_single_k(time, values)
            row.append(A)
            row.append(k)
            r2_list.append(r2)
            rmse_list.append(rmse)


        row.extend(r2_list)
        row.extend(rmse_list)
        study_rows.append(row)

    col_names = (
        ["study name"]
        + param_names
        + a_cols
        + k_cols
        + [f"{col} r_squared" for col in k_cols]
        + [f"{col} RMSE" for col in k_cols]
    )

    return pl.DataFrame(study_rows, schema=col_names, orient="row")