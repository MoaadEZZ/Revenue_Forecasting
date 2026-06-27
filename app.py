from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import re
import traceback

app = Flask(__name__)

def load_and_prepare(path: str) -> pd.DataFrame:
    if os.path.exists(path + "/data.xlsx"):
        df = pd.read_excel(path + "/data.xlsx")
    else:
        files = [f for f in os.listdir(path) if f.endswith('.xlsx')]
        df_list = []
        pattern = r"data_(.*?)_20\d{2}\.xlsx"
        for f in files:
            df_temp = pd.read_excel(os.path.join(path, f), skiprows=2)
            m = re.search(pattern, f)
            df_temp["customer_region"] = m.group(1) if m else "Unknown"
            df_list.append(df_temp)
        df = pd.concat(df_list, ignore_index=True)
        df.to_excel(path + "/data.xlsx", index=False)

    df.columns = df.columns.str.strip()
    df["Period"] = pd.to_datetime(df["Period"])
    # Clean revenue string
    if df["_S_Revenue_actual"].dtype == object:
        df["_S_Revenue_actual"] = df["_S_Revenue_actual"].str.replace(r"[,$]", "", regex=True).astype(float)
        
    return df

# ══════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── Model 1: XGBoost v3 (global model, ASINH transform, StandardScaler) ────
M1_PATH  = os.path.join(script_dir, "Models/revenue_v3_asinh.pkl")
m1       = joblib.load(M1_PATH)
m1_model    = m1["model"]
m1_scaler   = m1["scaler"]
m1_features = m1["features"]
m1_scope    = 6   # must match SCOPE used during training

# ── Model 2: XGBoost v2 (Rep_Code level, log-space features) ───────────────
M2_PATH  = os.path.join(script_dir, "Models/revenue_model_XGBoost_v2_rep.pkl")
m2       = joblib.load(M2_PATH)
m2_model       = m2["model"]
m2_model_lower = m2["model_lower"]
m2_model_upper = m2["model_upper"]
m2_features    = m2["features"]
m2_rep_encoder = m2["rep_encoder"]
m2_scope       = m2["scope"]

# ── Data ───────────────────────────────────────────────────────────────────
DATA_REVENUE_DIR = os.path.join(script_dir, "data_cagr_all")
DATA_PATH = os.path.join(script_dir, DATA_REVENUE_DIR+"/data.xlsx")
data = load_and_prepare(DATA_REVENUE_DIR)
data["Period"] = pd.to_datetime(data["Period"])
data = data.sort_values("Period")

# Pre-compute dataset year bounds (used by multiple routes)
DATA_MIN_YEAR = int(data["Period"].dt.year.min())
DATA_MAX_YEAR = int(data["Period"].dt.year.max())

# How many years beyond the dataset max the user may select
EXTRA_FORECAST_YEARS = 5

# ── Rep → (Domain, Sub-Business Line) lookup ───────────────────────────────
# Each rep_code is mapped to exactly one (Domain, Sub-Business Line) pair.
# If a rep appears under multiple combinations we take the most frequent one.
def _build_rep_meta(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Rep_Code", "Domain", "Sub-Business Line"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["Rep_Code", "Domain", "Sub-Business Line"]).set_index("Rep_Code")
    meta = (
        df.groupby(["Rep_Code", "Domain", "Sub-Business Line"])
          .size()
          .reset_index(name="_n")
          .sort_values("_n", ascending=False)
          .drop_duplicates(subset="Rep_Code")
          [["Rep_Code", "Domain", "Sub-Business Line"]]
          .set_index("Rep_Code")
    )
    return meta

REP_META = _build_rep_meta(data)   # DataFrame indexed by Rep_Code


# ══════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════

def transform_target(x):
    return np.arcsinh(x)

def inverse_transform_target(x):
    return np.sinh(x)

def log_sign(x):
    return np.sign(x) * np.log1p(np.abs(x))

def inverse_log_sign(x):
    clipped = np.clip(x, -700, 700)
    result  = np.sign(clipped) * np.expm1(np.abs(clipped))
    return np.where(np.isfinite(result), result, 0.0)

def next_month(period: pd.Timestamp) -> pd.Timestamp:
    if period.month == 12:
        return pd.Timestamp(period.year + 1, 1, 1)
    return pd.Timestamp(period.year, period.month + 1, 1)


# ══════════════════════════════════════════════════════════
#  MODEL 1 — feature builder + sequence predictor
# ══════════════════════════════════════════════════════════

def build_features_m1(history_df: pd.DataFrame, period: pd.Timestamp) -> dict:
    raw   = history_df["_S_Revenue_actual"].values.astype(float)
    trans = transform_target(raw)
    n     = len(trans)

    row = {"month": period.month, "quarter": (period.month - 1) // 3 + 1}
    for i in range(1, m1_scope + 1):
        row[f"lag_{i}"] = float(trans[-i]) if n >= i else 0.0

    window       = trans[-m1_scope:] if n >= m1_scope else trans
    rolling_mean = float(np.nanmean(window)) if len(window) > 0 else 0.0
    rolling_std  = float(np.nanstd(window))  if len(window) > 0 else 0.0
    row["rolling_mean"] = rolling_mean
    row["rolling_std"]  = rolling_std
    row["z_score"]      = (row["lag_1"] - rolling_mean) / (rolling_std + 1e-6)
    row["delta_1"]      = row["lag_1"] - row.get("lag_2", 0.0)
    return row


def predict_sequence_m1(
    history_df:   pd.DataFrame,
    rep_code:     str,
    start_period: pd.Timestamp,
    end_period:   pd.Timestamp,
) -> list:
    """
    Predicts month-by-month for rep_code from start_period to end_period
    inclusive, appending each prediction back into the rolling history.
    history_df must contain at minimum: Rep_Code, Period, _S_Revenue_actual.
    Returns list of {"period": "YYYY-MM", "value": float}.
    """
    results = []
    current_history = (
        history_df[history_df["Rep_Code"] == rep_code]
        [["Rep_Code", "Period", "_S_Revenue_actual"]]
        .sort_values("Period")
        .reset_index(drop=True)
        .copy()
    )
    period = start_period

    while period <= end_period:
        if len(current_history) < m1_scope:
            period = next_month(period)
            continue

        row      = build_features_m1(current_history, period)
        X_row    = pd.DataFrame([row])[m1_features].fillna(0).values
        X_scaled = m1_scaler.transform(X_row)
        pred_val = float(inverse_transform_target(
            np.array([m1_model.predict(X_scaled)[0]])
        )[0])

        results.append({"period": period.strftime("%Y-%m"), "value": pred_val})
        current_history = pd.concat(
            [current_history, pd.DataFrame([{
                "Rep_Code": rep_code,
                "Period":   period,
                "_S_Revenue_actual": pred_val,
            }])],
            ignore_index=True,
        )
        period = next_month(period)

    return results


# ══════════════════════════════════════════════════════════
#  MODEL 2 — feature builder + sequence predictor
# ══════════════════════════════════════════════════════════

def build_features_m2(
    history_df: pd.DataFrame,
    rep_code:   str,
    period:     pd.Timestamp,
    spike_threshold: float = None,
) -> dict:
    raw    = history_df["_S_Revenue_actual"].values
    log_s  = log_sign(raw)
    n      = len(raw)
    rep_id = m2_rep_encoder.get(rep_code, -1)

    row = {
        "rep_id":  rep_id / max(n, 1),
        "month":   period.month,
        "quarter": (period.month - 1) // 3 + 1,
    }
    for i in range(1, m2_scope):
        row[f"lag_{i}"]      = log_s[-i]        if n >= i else 0.0
        row[f"lag_sign_{i}"] = np.sign(raw[-i]) if n >= i else 0.0

    window_log = log_s[-m2_scope:] if n >= m2_scope else log_s
    row["rolling_mean"] = float(np.nanmean(window_log))
    row["rolling_std"]  = float(np.nanstd(window_log))
    row["rolling_min"]  = float(np.nanmin(window_log))
    row["rolling_max"]  = float(np.nanmax(window_log))
    row["neg_count"]    = float(np.sum(window_log < 0))

    denom = abs(row["rolling_mean"]) + 1e-3
    row["lag1_to_mean_ratio"] = np.clip(row["lag_1"] / denom, -5, 5)
    row["rolling_min_ratio"]  = np.clip(row["rolling_min"] / denom, -5, 5)
    row["mom_growth"]         = float(log_s[-1] - log_s[-2]) if n >= 2 else 0.0

    if spike_threshold is None:
        spike_threshold = float(np.quantile(raw, 0.95)) if n > 0 else 0.0
    window_raw = raw[-m2_scope:] if n >= m2_scope else raw
    row["spike_count"] = float(np.sum(window_raw > spike_threshold))
    return row


def predict_sequence_m2(
    history_df:   pd.DataFrame,
    rep_code:     str,
    start_period: pd.Timestamp,
    end_period:   pd.Timestamp,
) -> tuple:
    results_mid, results_lower, results_upper = [], [], []
    current_history = (
        history_df[history_df["Rep_Code"] == rep_code]
        [["Rep_Code", "Period", "_S_Revenue_actual"]]
        .sort_values("Period")
        .reset_index(drop=True)
        .copy()
    )
    seed_raw     = current_history["_S_Revenue_actual"].values
    frozen_spike = float(np.quantile(seed_raw, 0.95)) if len(seed_raw) > 0 else 0.0
    period       = start_period

    while period <= end_period:
        row   = build_features_m2(current_history, rep_code, period, frozen_spike)
        X_row = pd.DataFrame([row])[m2_features].fillna(0)

        pred_mid   = float(inverse_log_sign(m2_model.predict(X_row))[0])
        pred_lower = float(inverse_log_sign(m2_model_lower.predict(X_row))[0])
        pred_upper = float(inverse_log_sign(m2_model_upper.predict(X_row))[0])

        results_mid.append({"period": period.strftime("%Y-%m"), "value": pred_mid})
        results_lower.append({"period": period.strftime("%Y-%m"), "value": pred_lower})
        results_upper.append({"period": period.strftime("%Y-%m"), "value": pred_upper})

        current_history = pd.concat(
            [current_history, pd.DataFrame([{
                "Rep_Code": rep_code,
                "Period":   period,
                "_S_Revenue_actual": pred_mid,
            }])],
            ignore_index=True,
        )
        period = next_month(period)

    return results_mid, results_lower, results_upper


# ══════════════════════════════════════════════════════════
#  CAGR HELPERS
# ══════════════════════════════════════════════════════════

def compute_cagr(start_val: float, end_val: float, n_years: int):
    """Standard CAGR. Returns None when mathematically undefined."""
    if n_years <= 0 or start_val <= 0 or end_val <= 0:
        return None
    return (end_val / start_val) ** (1.0 / n_years) - 1.0


def get_annual_revenue_real(year: int, base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sums _S_Revenue_actual for all months of `year` from real data.
    Returns DataFrame [Rep_Code, annual_revenue].
    """
    mask = base_df["Period"].dt.year == year
    return (
        base_df[mask]
        .groupby("Rep_Code")["_S_Revenue_actual"]
        .sum()
        .reset_index()
        .rename(columns={"_S_Revenue_actual": "annual_revenue"})
    )


def get_annual_revenue_predicted(year: int, live_df: pd.DataFrame) -> tuple:
    """
    Predicts Jan–Dec of `year` for every rep in live_df using Model 1.
    Returns:
      - rep_annual: DataFrame [Rep_Code, annual_revenue]
      - monthly_rows: list of dicts {Rep_Code, Period, _S_Revenue_actual}
        (all 12 months per rep, needed to extend live_df for chained years)
    """
    start_p   = pd.Timestamp(year, 1, 1)
    end_p     = pd.Timestamp(year, 12, 1)
    rep_codes = live_df["Rep_Code"].unique()

    rep_annual   = []
    monthly_rows = []

    for rep in rep_codes:
        if len(live_df[live_df["Rep_Code"] == rep]) < m1_scope:
            continue
        monthly = predict_sequence_m1(live_df, rep, start_p, end_p)
        if not monthly:
            continue
        annual_sum = sum(m["value"] for m in monthly)
        rep_annual.append({"Rep_Code": rep, "annual_revenue": annual_sum})
        for m in monthly:
            monthly_rows.append({
                "Rep_Code":          rep,
                "Period":            pd.Timestamp(m["period"] + "-01"),
                "_S_Revenue_actual": m["value"],
            })

    return (
        pd.DataFrame(rep_annual, columns=["Rep_Code", "annual_revenue"]),
        monthly_rows,
    )


def build_domain_sbl_table(rev_df: pd.DataFrame, rep_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Joins rep-level annual revenues to Domain/SBL metadata and groups.
    Returns DataFrame [Sub-Business Line, Domain, revenue].
    """
    merged  = rev_df.join(rep_meta, on="Rep_Code", how="inner")
    grouped = (
        merged
        .groupby(["Sub-Business Line", "Domain"])["annual_revenue"]
        .sum()
        .reset_index()
        .rename(columns={"annual_revenue": "revenue"})
    )
    return grouped


def assemble_cagr_response(
    start_year:        int,
    end_year:          int,
    start_table:       pd.DataFrame,   # [Sub-Business Line, Domain, revenue]
    end_table:         pd.DataFrame,   # same shape
    end_year_predicted: bool = False,
) -> dict:
    """Merges start/end tables, computes CAGR per Domain, averages per SBL."""
    n_years = end_year - start_year

    merged = start_table.merge(
        end_table,
        on=["Sub-Business Line", "Domain"],
        how="outer",
        suffixes=("_start", "_end"),
    ).fillna(0.0)

    result_sbls = []
    for sbl_name, sbl_df in merged.groupby("Sub-Business Line"):
        domains_out = []
        cagr_values = []

        for _, row in sbl_df.iterrows():
            rev_s = float(row["revenue_start"])
            rev_e = float(row["revenue_end"])
            cagr  = compute_cagr(rev_s, rev_e, n_years)
            if cagr is not None:
                cagr_values.append(cagr)
            domains_out.append({
                "name":          row["Domain"],
                "revenue_start": rev_s,
                "revenue_end":   rev_e,
                "cagr":          round(cagr * 100, 4) if cagr is not None else None,
            })

        domains_out.sort(key=lambda d: d["name"])
        avg_cagr = float(np.mean(cagr_values)) if cagr_values else None

        result_sbls.append({
            "name":        sbl_name,
            "total_start": float(sbl_df["revenue_start"].sum()),
            "total_end":   float(sbl_df["revenue_end"].sum()),
            "avg_cagr":    round(avg_cagr * 100, 4) if avg_cagr is not None else None,
            "domains":     domains_out,
        })

    result_sbls.sort(key=lambda s: s["name"])
    return {
        "start_year":          start_year,
        "end_year":            end_year,
        "n_years":             n_years,
        "end_year_predicted":  end_year_predicted,
        "sub_business_lines":  result_sbls,
    }


# ══════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════

@app.route("/")
def index():
    products = sorted(data["Product"].dropna().unique().tolist()) if "Product" in data.columns else []
    return render_template("index.html", products=products)
    


@app.route("/cagr")
def cagr_page():
    return render_template("cagr.html")


@app.route("/get_products")
def get_product():
    if "Product" in data.columns:
        return jsonify(sorted(data["Product"].dropna().unique().tolist()))
    return jsonify([])

@app.route("/api/products")
def get_products():
    # Merge the unique reps with their Domain metadata
    df_products = data[["Rep_Code"]].drop_duplicates()
    # Join with REP_META to get the Domain column
    merged = df_products.join(REP_META, on="Rep_Code", how="left")
    
    # Return list of dicts: [{"rep_code": "...", "domain": "..."}, ...]
    res = merged.reset_index(drop=True).to_dict(orient="records")
    return jsonify(res)

@app.route("/get_reps_by_domain")
def get_reps_by_domain():
    """
    Returns all Rep_Codes that belong to the given Domain (case-sensitive).
    Used by the forecast page when deep-linking from the CAGR analysis.
    Query param: domain (required)
    """
    domain = request.args.get("domain", "").strip()
    if not domain or "Domain" not in data.columns:
        return jsonify([])
    reps = sorted(
        data[data["Domain"] == domain]["Rep_Code"].dropna().unique().tolist()
    )
    return jsonify(reps)


@app.route("/get_rep_codes")
def get_rep_codes():
    product_param = request.args.get("product", "").strip()
    if not product_param or "Product" not in data.columns:
        rep_codes = sorted(data["Rep_Code"].dropna().unique().tolist())
    else:
        products  = [p.strip() for p in product_param.split(",") if p.strip()]
        rep_codes = sorted(
            data[data["Product"].isin(products)]["Rep_Code"].dropna().unique().tolist()
        )
    return jsonify(rep_codes)


@app.route("/get_cagr_years")
def get_cagr_years():
    """
    Returns min_year (dataset start), max_year (dataset end), and
    max_select_year (max_year + EXTRA_FORECAST_YEARS) for the frontend
    to build year dropdowns and flag which options are forecast-only.
    """
    return jsonify({
        "min_year":        DATA_MIN_YEAR,
        "max_year":        DATA_MAX_YEAR,
        "max_select_year": DATA_MAX_YEAR + EXTRA_FORECAST_YEARS,
    })


@app.route("/compute_cagr", methods=["POST"])
def compute_cagr_route():
    """
    Body: { start_year: int, end_year: int }

    ┌─ end_year <= DATA_MAX_YEAR ──────────────────────────────────────────┐
    │  Pure real-data path: sum actual revenues per year, group by         │
    │  Domain → Sub-Business Line, compute CAGR.                           │
    └──────────────────────────────────────────────────────────────────────┘
    ┌─ end_year > DATA_MAX_YEAR ───────────────────────────────────────────┐
    │  Hybrid path:                                                         │
    │   • start_year revenue  → always from real data                      │
    │   • end_year revenue    → Model 1 predicts Jan–Dec for every         │
    │     rep_code (chaining through any intermediate future years so       │
    │     lag inputs remain valid), then sums 12 monthly predictions        │
    │     into an annual total per rep, then groups by Domain → SBL.        │
    └──────────────────────────────────────────────────────────────────────┘
    """
    try:
        body       = request.json
        start_year = int(body["start_year"])
        end_year   = int(body["end_year"])
        max_select = DATA_MAX_YEAR + EXTRA_FORECAST_YEARS

        # ── Validate ──────────────────────────────────────────────────────
        if start_year < DATA_MIN_YEAR:
            return jsonify({"error": f"Start year cannot be before {DATA_MIN_YEAR}."}), 400
        if start_year > DATA_MAX_YEAR:
            return jsonify({"error": f"Start year must be within the dataset (≤ {DATA_MAX_YEAR})."}), 400
        if end_year > max_select:
            return jsonify({"error": f"End year cannot exceed {max_select}."}), 400
        if start_year >= end_year:
            return jsonify({"error": "Start year must be strictly before end year."}), 400

        for col in ("Domain", "Sub-Business Line", "_S_Revenue_actual"):
            if col not in data.columns:
                return jsonify({"error": f"Column '{col}' not found in dataset."}), 500

        # Minimal working copy (only columns needed for prediction)
        base_df = data[["Rep_Code", "Period", "_S_Revenue_actual"]].copy()

        # ── Start-year revenue: always from real data ─────────────────────
        real_start  = get_annual_revenue_real(start_year, base_df)
        start_table = build_domain_sbl_table(real_start, REP_META)

        # ── End-year revenue ──────────────────────────────────────────────
        if end_year <= DATA_MAX_YEAR:
            # ── Case A: fully within dataset — use real data ──────────────
            real_end  = get_annual_revenue_real(end_year, base_df)
            end_table = build_domain_sbl_table(real_end, REP_META)
            end_predicted = False

        else:
            # ── Case B: end_year is in the future — predict with Model 1 ──
            #
            # live_df starts as the full real dataset.
            # For each intermediate year between DATA_MAX_YEAR+1 and end_year-1
            # we predict all 12 months and append them to live_df so that the
            # next year has valid lag inputs.
            # Finally we predict end_year and sum the 12 months per rep.

            live_df = base_df.copy()

            for interim_year in range(DATA_MAX_YEAR + 1, end_year):
                _, monthly_rows = get_annual_revenue_predicted(interim_year, live_df)
                if monthly_rows:
                    live_df = pd.concat(
                        [live_df, pd.DataFrame(monthly_rows)],
                        ignore_index=True,
                    ).sort_values("Period").reset_index(drop=True)

            # Predict the target end_year
            predicted_end, _ = get_annual_revenue_predicted(end_year, live_df)
            end_table         = build_domain_sbl_table(predicted_end, REP_META)
            end_predicted     = True

        # ── Assemble & return ─────────────────────────────────────────────
        result = assemble_cagr_response(
            start_year, end_year, start_table, end_table, end_predicted
        )
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body         = request.json
        rep_code     = body["rep_code"]
        start_year   = int(body["start_year"])
        start_month  = int(body["start_month"])
        end_year     = int(body["end_year"])
        end_month    = int(body["end_month"])
        model_choice = body.get("model", "m1")

        start_period = pd.Timestamp(start_year, start_month, 1)
        end_period   = pd.Timestamp(end_year,   end_month,   1)

        if end_period <= start_period:
            return jsonify({"error": "End date must be after start date"}), 400

        rep_data = (
            data[data["Rep_Code"] == rep_code]
            [["Rep_Code", "Period", "_S_Revenue_actual"]]
            .sort_values("Period")
            .reset_index(drop=True)
        )

        if len(rep_data) == 0:
            return jsonify({"error": f"No data found for rep {rep_code}"}), 400

        real_in_range = rep_data[
            (rep_data["Period"] >= start_period) & (rep_data["Period"] <= end_period)
        ]
        real_points = [
            {"period": r["Period"].strftime("%Y-%m"), "value": float(r["_S_Revenue_actual"])}
            for _, r in real_in_range.iterrows()
        ]

        history_seed = rep_data[rep_data["Period"] < start_period]
        if len(history_seed) == 0:
            return jsonify({
                "error": (
                    f"No historical data before {start_period.strftime('%Y-%m')} "
                    f"for rep {rep_code}. Move start date later."
                )
            }), 400

        if len(real_in_range) > 0:
            last_real                = real_in_range["Period"].max()
            pred_start               = next_month(last_real)
            history_for_continuation = rep_data[rep_data["Period"] <= last_real]
        else:
            pred_start               = start_period
            history_for_continuation = history_seed

        response = {"real_points": real_points, "rep_code": rep_code}

        if model_choice in ("m1", "both"):
            if len(history_seed) < m1_scope:
                response["m1_error"] = (
                    f"Not enough history for Model 1 (need {m1_scope} rows, "
                    f"got {len(history_seed)})"
                )
            else:
                m1_curve1 = predict_sequence_m1(history_seed, rep_code, start_period, end_period)
                m1_curve2_pred = []
                
                if pred_start <= end_period:
                    m1_curve2_pred = predict_sequence_m1(
                        history_for_continuation, rep_code, pred_start, end_period
                    )
                response["m1_curve1"] = m1_curve1
                response["m1_curve2"] = real_points + m1_curve2_pred

        if model_choice in ("m2", "both"):
            m2_mid, m2_lower, m2_upper = predict_sequence_m2(
                history_seed, rep_code, start_period, end_period
            )
            m2_cont_mid, m2_cont_lower, m2_cont_upper = [], [], []
            if pred_start <= end_period:
                m2_cont_mid, m2_cont_lower, m2_cont_upper = predict_sequence_m2(
                    history_for_continuation, rep_code, pred_start, end_period
                )

            response["m2_curve1"]       = m2_mid
            response["m2_curve1_lower"] = m2_lower
            response["m2_curve1_upper"] = m2_upper
            response["m2_curve2"]       = real_points + m2_cont_mid
            response["m2_curve2_lower"] = real_points + m2_cont_lower
            response["m2_curve2_upper"] = real_points + m2_cont_upper

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
