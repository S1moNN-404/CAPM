import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


st.set_page_config(page_title="NDX CAPM Dashboard (Beginner Friendly)", layout="wide")


def safe_download_adj_close(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    """
    Download adjusted close data for one ticker from yfinance.
    Returns an empty Series when data cannot be loaded.
    """
    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception:
        return pd.Series(dtype=float)

    if raw is None or raw.empty:
        return pd.Series(dtype=float)

    if "Adj Close" not in raw.columns:
        return pd.Series(dtype=float)

    series = raw["Adj Close"].copy()
    series.name = ticker.upper()
    return series


def main() -> None:
    st.title("Nasdaq-100 CAPM & Regression Dashboard")
    st.caption("Beginner-friendly project focused on a stock vs Nasdaq-100 proxy (QQQ).")

    st.markdown(
        """
        This dashboard explains each CAPM/regression step in plain English.
        You can change the stock ticker, date range, and risk-free rate.
        """
    )

    with st.sidebar:
        st.header("Inputs")
        ticker = st.text_input("Stock ticker (example: AAPL, MSFT, NVDA)", value="AAPL").strip().upper()
        benchmark = st.text_input("Market benchmark ticker (default Nasdaq-100 proxy)", value="QQQ").strip().upper()
        start_date = st.date_input("Start date", value=pd.Timestamp("2018-01-01"))
        end_date = st.date_input("End date", value=pd.Timestamp.today().date())
        annual_rf_percent = st.number_input(
            "Annual risk-free rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1,
            help="Used to compute daily excess returns.",
        )
        run_button = st.button("Run analysis")

    st.info(
        "Default setup: Stock = AAPL, Benchmark = QQQ (Nasdaq-100 ETF proxy). "
        "QQQ is used as a practical market benchmark for Nasdaq-100 style analysis."
    )

    if not run_button:
        st.stop()

    if ticker == "":
        st.error("Please enter a valid stock ticker.")
        st.stop()

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    st.subheader("1) Download and clean adjusted close data")
    stock_prices = safe_download_adj_close(ticker, pd.Timestamp(start_date), pd.Timestamp(end_date))
    bench_prices = safe_download_adj_close(benchmark, pd.Timestamp(start_date), pd.Timestamp(end_date))

    if stock_prices.empty:
        st.error(
            f"No adjusted close data found for stock ticker '{ticker}'. "
            "Check the ticker, date range, or internet connection."
        )
        st.stop()
    if bench_prices.empty:
        st.error(
            f"No adjusted close data found for benchmark '{benchmark}'. "
            "Check the ticker, date range, or internet connection."
        )
        st.stop()

    prices_raw = pd.concat([stock_prices, bench_prices], axis=1)
    prices_raw.columns = [ticker, benchmark]
    rows_before_cleaning = len(prices_raw)
    prices = prices_raw.dropna().copy()
    rows_after_cleaning = len(prices)

    min_date = prices.index.min().date()
    max_date = prices.index.max().date()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requested start date", str(start_date))
    c2.metric("Requested end date", str(end_date))
    c3.metric("Rows before cleaning", rows_before_cleaning)
    c4.metric("Rows after cleaning", rows_after_cleaning)
    st.write(f"Actual clean data range: **{min_date}** to **{max_date}**")

    st.subheader("2) Price series chart")
    st.line_chart(prices, use_container_width=True)

    st.subheader("3) Daily returns and excess returns")
    returns = prices.pct_change().dropna().copy()
    returns.columns = [f"{ticker}_ret", f"{benchmark}_ret"]

    daily_rf = (1 + annual_rf_percent / 100) ** (1 / 252) - 1
    returns["stock_excess"] = returns[f"{ticker}_ret"] - daily_rf
    returns["market_excess"] = returns[f"{benchmark}_ret"] - daily_rf

    st.write(f"Daily risk-free rate used: **{daily_rf:.8f}**")
    st.dataframe(returns.head(10), use_container_width=True)

    return_chart = returns[[f"{ticker}_ret", f"{benchmark}_ret"]].rename(
        columns={f"{ticker}_ret": f"{ticker} daily return", f"{benchmark}_ret": f"{benchmark} daily return"}
    )
    st.subheader("4) Return series chart")
    st.line_chart(return_chart, use_container_width=True)

    model_df = returns[["stock_excess", "market_excess"]].dropna().copy()
    if len(model_df) < 20:
        st.error("Not enough clean return data for regression. Try a longer date range.")
        st.stop()

    st.subheader("5) Train/test split (70% / 30% by time)")
    split_index = int(len(model_df) * 0.70)
    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()

    c5, c6, c7 = st.columns(3)
    c5.metric("Total return rows", len(model_df))
    c6.metric("Train rows (70%)", len(train_df))
    c7.metric("Test rows (30%)", len(test_df))

    x_train = sm.add_constant(train_df["market_excess"])
    y_train = train_df["stock_excess"]
    ols_model = sm.OLS(y_train, x_train).fit()

    alpha = ols_model.params.get("const", np.nan)
    beta = ols_model.params.get("market_excess", np.nan)
    r_squared = ols_model.rsquared
    pvals = ols_model.pvalues
    stderrs = ols_model.bse

    st.subheader("6) CAPM regression details (train set)")
    details = pd.DataFrame(
        {
            "Metric": [
                "Alpha (intercept)",
                "Beta (market sensitivity)",
                "R-squared",
                "P-value (alpha)",
                "P-value (beta)",
                "Std error (alpha)",
                "Std error (beta)",
            ],
            "Value": [
                alpha,
                beta,
                r_squared,
                pvals.get("const", np.nan),
                pvals.get("market_excess", np.nan),
                stderrs.get("const", np.nan),
                stderrs.get("market_excess", np.nan),
            ],
        }
    )
    st.dataframe(details, use_container_width=True)

    with st.expander("Full OLS summary table"):
        st.text(ols_model.summary().as_text())

    st.subheader("7) Scatter (train): market excess return vs stock excess return")
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
    ax_scatter.scatter(train_df["market_excess"], train_df["stock_excess"], alpha=0.5, label="Train points")
    x_line = np.linspace(train_df["market_excess"].min(), train_df["market_excess"].max(), 200)
    y_line = alpha + beta * x_line
    ax_scatter.plot(x_line, y_line, color="red", linewidth=2, label="Fitted line")
    ax_scatter.set_xlabel("Market excess return (QQQ)")
    ax_scatter.set_ylabel(f"{ticker} excess return")
    ax_scatter.set_title("Train scatter with fitted CAPM line")
    ax_scatter.legend()
    st.pyplot(fig_scatter)

    st.subheader("8) Test predictions vs actual")
    x_test = sm.add_constant(test_df["market_excess"])
    test_df["predicted_excess"] = ols_model.predict(x_test)
    test_df["residual"] = test_df["stock_excess"] - test_df["predicted_excess"]

    mae = mean_absolute_error(test_df["stock_excess"], test_df["predicted_excess"])
    rmse = np.sqrt(mean_squared_error(test_df["stock_excess"], test_df["predicted_excess"]))
    residual_mean = test_df["residual"].mean()
    residual_std = test_df["residual"].std()

    c8, c9 = st.columns(2)
    c8.metric("MAE (test)", f"{mae:.6f}")
    c9.metric("RMSE (test)", f"{rmse:.6f}")

    st.write("Prediction table (first 15 test rows):")
    st.dataframe(
        test_df[["stock_excess", "predicted_excess", "residual"]].head(15),
        use_container_width=True,
    )

    chart_df = test_df[["stock_excess", "predicted_excess"]].rename(
        columns={
            "stock_excess": "Actual stock excess return",
            "predicted_excess": "Predicted stock excess return",
        }
    )
    st.line_chart(chart_df, use_container_width=True)

    st.subheader("9) Residual diagnostics summary")
    rd1, rd2 = st.columns(2)
    rd1.metric("Residual mean (test)", f"{residual_mean:.8f}")
    rd2.metric("Residual std (test)", f"{residual_std:.8f}")

    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    ax_hist.hist(test_df["residual"], bins=30, edgecolor="black", alpha=0.7)
    ax_hist.set_title("Residual histogram (test)")
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Frequency")
    st.pyplot(fig_hist)

    st.subheader("Warnings and assumptions")
    st.warning(
        """
        - This model uses historical data and historical relationships can change.
        - CAPM is a simplified one-factor model and does not include all market drivers.
        - Results are for education and analysis, not financial advice.
        - A good historical fit does not guarantee future prediction accuracy.
        """
    )


if __name__ == "__main__":
    main()
