import polars as pl
from regression_utils import filter_df, build_k_df

sweep = "amp-cor"

TAG = "r1"
OFFSET = 2
FITTING_COLS = ["x lacey", "y lacey", "z lacey", "r lacey"]

lacey_csv_path = f"raw_lacey_csvs/{sweep}/lrv_{sweep}_{TAG}.csv"
raw_df = pl.read_csv(lacey_csv_path)

substrings = ["lacey"]

filtered_df = filter_df(raw_df, substrings, OFFSET)
k_df = build_k_df(filtered_df, FITTING_COLS)

k_df.write_csv("kdf_test.csv")