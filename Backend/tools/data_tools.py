"""Data Agent tools — data quality, profiling, cleaning, and analysis.

Tools for the Data Agent covering schema inspection, profiling, data quality
assessment, missing pattern analysis, class imbalance, drift detection,
outlier detection, and the full 6-step cleaning pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from strands import tool

from backend.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level callback reference (set by orchestrator before agent runs)
# ---------------------------------------------------------------------------
_callback_handler: Any = None


def set_callback_handler(handler: Any) -> None:
    """Set the module-level callback handler for emitting SSE events from tools."""
    global _callback_handler
    _callback_handler = handler


# ---------------------------------------------------------------------------
# SQL safety constants
# ---------------------------------------------------------------------------

ALLOWED_QUERY_PREFIXES = ("select", "with", "pragma", "explain")
DISALLOWED_SQL_PATTERNS = (
    "insert ", "update ", "delete ", "drop ", "alter ", "create ",
    "attach ", "detach ", "replace ", "truncate ", "vacuum", "reindex",
)

# Columns that leak post-origination information (must be dropped for PD)
LEAKAGE_COLUMNS = [
    "recoveries", "collection_recovery_fee",
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "last_pymnt_d", "last_pymnt_amnt",
    "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low",
    "out_prncp", "out_prncp_inv",
    "next_pymnt_d", "pymnt_plan",
]

# Columns not useful for modeling
DROP_COLUMNS = [
    "id", "member_id", "url", "desc", "emp_title", "title",
    "zip_code", "policy_code",
]

# Columns to Winsorize at 1st/99th percentile
WINSORIZE_COLS = [
    "annual_inc", "revol_bal", "loan_amnt", "funded_amnt", "dti", "open_acc",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(payload: Dict[str, Any]) -> dict:
    return {"status": "success", "content": [{"text": json.dumps(payload, default=str, indent=2)}]}


def _error(message: str) -> dict:
    return {"status": "error", "content": [{"text": message}]}


def _open_db(db_path: Path | None = None) -> sqlite3.Connection:
    p = db_path or get_settings().db_abs_path
    if not p.exists():
        raise FileNotFoundError(f"Database not found: {p}")
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def _validate_read_only_sql(query: str) -> str:
    candidate = (query or "").strip()
    if not candidate:
        raise ValueError("SQL query is required.")
    lowered = candidate.lower().lstrip("(").strip()
    if ";" in lowered:
        stripped = [part.strip() for part in lowered.split(";") if part.strip()]
        if len(stripped) > 1:
            raise ValueError("Only one SQL statement is allowed per call.")
        lowered = stripped[0]
    if not lowered.startswith(ALLOWED_QUERY_PREFIXES):
        raise ValueError("Only read-only SQL is allowed: SELECT, WITH, PRAGMA, EXPLAIN.")
    if any(pattern in lowered for pattern in DISALLOWED_SQL_PATTERNS):
        raise ValueError("Query includes a disallowed SQL keyword.")
    return candidate


def _fetch_dicts(cursor: sqlite3.Cursor) -> List[Dict[str, Any]]:
    rows = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description] if cursor.description else []
    return [dict(zip(cols, row)) for row in rows]


@lru_cache(maxsize=1)
def _load_lc_dictionary() -> Dict[str, str]:
    """Load LendingClub data dictionary from CSV."""
    settings = get_settings()
    path = settings.dictionary_abs_path
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    lookup: Dict[str, str] = {}
    for _, row in df.iterrows():
        field = str(row.get("LoanStatNew", "")).strip()
        desc = str(row.get("Description", "")).strip()
        if field and field.lower() != "nan":
            lookup[field.lower().strip()] = desc
    return lookup


def _classify_feature_type(series: pd.Series, col_name: str) -> str:
    """Classify a column into a feature type category."""
    name_lower = col_name.lower()
    # Identifier patterns
    if name_lower in ("id", "member_id") or name_lower.endswith("_id"):
        return "identifier"
    # Date patterns
    if "date" in name_lower or name_lower.endswith("_d") or name_lower in ("issue_d", "earliest_cr_line"):
        return "date"
    unique_count = int(series.nunique())
    if unique_count <= 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        return "ordinal" if unique_count <= 20 else "continuous"
    return "nominal"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def list_tables() -> dict:
    """List all non-system tables in the LendingClub SQLite database."""
    try:
        settings = get_settings()
        with _open_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
            return _ok({"database_path": str(settings.db_abs_path), "tables": [row[0] for row in cur.fetchall()]})
    except Exception as exc:
        return _error(f"Failed to list tables: {exc}")


@tool
def describe_table(table_name: str = "") -> dict:
    """Describe a table's columns and attach LendingClub dictionary definitions.

    Args:
        table_name: Table to inspect (defaults to configured table).
    """
    try:
        settings = get_settings()
        table_name = table_name or settings.db_table
        lookup = _load_lc_dictionary()

        with _open_db() as conn:
            cur = conn.cursor()
            cur.execute(f'PRAGMA table_info("{table_name}")')
            rows = cur.fetchall()
            if not rows:
                return _error(f"Table '{table_name}' not found or has no columns.")

            columns: List[Dict[str, Any]] = []
            matched = 0
            for row in rows:
                col_name = row[1]
                desc = lookup.get(col_name.lower().strip())
                if desc:
                    matched += 1
                columns.append({
                    "column_name": col_name,
                    "sqlite_type": row[2],
                    "not_null": bool(row[3]),
                    "primary_key": bool(row[5]),
                    "description": desc,
                })

            return _ok({
                "table_name": table_name,
                "column_count": len(columns),
                "dictionary_match_count": matched,
                "columns": columns,
            })
    except Exception as exc:
        return _error(f"Failed to describe table: {exc}")


@tool
def get_data_dictionary_summary() -> dict:
    """Summarize the LendingClub data dictionary and mapping to the database table."""
    try:
        settings = get_settings()
        lookup = _load_lc_dictionary()
        if not lookup:
            return _error(f"Dictionary not found at: {settings.dictionary_abs_path}")

        with _open_db() as conn:
            cur = conn.cursor()
            cur.execute(f'PRAGMA table_info("{settings.db_table}")')
            table_cols = [row[1] for row in cur.fetchall()]

        matched = [c for c in table_cols if c.lower().strip() in lookup]
        unmatched = [c for c in table_cols if c.lower().strip() not in lookup]

        return _ok({
            "dictionary_path": str(settings.dictionary_abs_path),
            "table_name": settings.db_table,
            "table_column_count": len(table_cols),
            "dictionary_field_count": len(lookup),
            "matched_count": len(matched),
            "unmatched_count": len(unmatched),
            "unmatched_columns": unmatched[:30],
        })
    except Exception as exc:
        return _error(f"Failed to summarize dictionary: {exc}")


@tool
def run_sql_query(query: str, limit: int = 200) -> dict:
    """Run a single read-only SQL query on the LendingClub database.

    Args:
        query: SQL (SELECT/WITH/PRAGMA/EXPLAIN only).
        limit: Maximum rows returned.
    """
    try:
        safe = _validate_read_only_sql(query)
        cap = max(1, min(limit, 2000))
        wrapped = f"SELECT * FROM ({safe}) q LIMIT {cap}"
        with _open_db() as conn:
            cur = conn.cursor()
            cur.execute(wrapped)
            data = _fetch_dicts(cur)
        return _ok({"query": safe, "limit": cap, "returned_rows": len(data), "rows": data})
    except Exception as exc:
        return _error(f"Failed to run query: {exc}")


# ---------------------------------------------------------------------------
# Improvement 5: Adaptive Schema-Aware Baseline Scan
# ---------------------------------------------------------------------------


@tool
def run_baseline_data_quality_scan(
    table_name: str = "",
    key_column: str = "id",
    date_column: str = "issue_d",
) -> dict:
    """Run an adaptive baseline data-quality scan. Generates queries based on
    the actual schema — works with any dataset, not just LendingClub.

    Args:
        table_name: Table to scan (defaults to configured table).
        key_column: Primary key column.
        date_column: Date/vintage column.
    """
    settings = get_settings()
    table_name = table_name or settings.db_table

    # --- Adaptive schema detection ---
    with _open_db() as conn:
        cur = conn.cursor()
        cur.execute(f'PRAGMA table_info("{table_name}")')
        schema_rows = cur.fetchall()
    col_names_lower = {r[1].lower() for r in schema_rows}

    # Core queries (always run)
    baseline_queries: Dict[str, str] = {
        "row_count": f'SELECT COUNT(*) AS row_count FROM "{table_name}"',
        "column_count": f'SELECT COUNT(*) AS column_count FROM pragma_table_info("{table_name}")',
        "key_nulls": (
            f'SELECT COUNT(*) AS null_key_count FROM "{table_name}" '
            f'WHERE "{key_column}" IS NULL OR TRIM(CAST("{key_column}" AS TEXT)) = ""'
        ),
        "duplicate_keys": (
            f'SELECT "{key_column}", COUNT(*) AS cnt FROM "{table_name}" '
            f'GROUP BY 1 HAVING COUNT(*) > 1 ORDER BY cnt DESC LIMIT 50'
        ),
    }

    # Date/vintage query
    if date_column.lower() in col_names_lower:
        baseline_queries["vintage_coverage"] = (
            f'SELECT "{date_column}", COUNT(*) AS cnt FROM "{table_name}" '
            f'GROUP BY 1 ORDER BY 1 DESC LIMIT 48'
        )

    # Adaptive numeric range detection
    known_numeric = ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_low"]
    range_cols = [c for c in known_numeric if c.lower() in col_names_lower]
    if not range_cols:
        # Fall back to first 5 numeric-type columns from schema
        for r in schema_rows:
            if (r[2] or "").upper() in ("INTEGER", "REAL", "NUMERIC", "FLOAT", "DOUBLE"):
                range_cols.append(r[1])
            if len(range_cols) >= 5:
                break
    if range_cols:
        min_max_parts = ", ".join(
            f'MIN("{c}") AS min_{c}, MAX("{c}") AS max_{c}' for c in range_cols
        )
        baseline_queries["core_metric_ranges"] = f'SELECT {min_max_parts} FROM "{table_name}"'

    # Adaptive categorical distributions
    for col_name, query_name in [
        ("loan_status", "loan_status_distribution"),
        ("grade", "grade_distribution"),
        ("purpose", "purpose_distribution"),
        ("home_ownership", "home_ownership_distribution"),
    ]:
        if col_name.lower() in col_names_lower:
            baseline_queries[query_name] = (
                f'SELECT "{col_name}", COUNT(*) AS cnt FROM "{table_name}" '
                f'GROUP BY 1 ORDER BY cnt DESC LIMIT 20'
            )

    # Vintage default rates (for sparkline/chart data)
    if "loan_status" in col_names_lower and date_column.lower() in col_names_lower:
        baseline_queries["vintage_default_rates"] = (
            f'SELECT "{date_column}", '
            f'COUNT(*) AS total, '
            f'SUM(CASE WHEN loan_status = "Charged Off" THEN 1 ELSE 0 END) AS defaults, '
            f'ROUND(1.0 * SUM(CASE WHEN loan_status = "Charged Off" THEN 1 ELSE 0 END) / COUNT(*), 4) AS default_rate '
            f'FROM "{table_name}" '
            f'WHERE loan_status IN ("Fully Paid", "Charged Off") '
            f'GROUP BY 1 ORDER BY 1'
        )

    try:
        output: Dict[str, Any] = {"table_name": table_name, "query_trace": [], "schema_adaptive": True}
        with _open_db() as conn:
            cur = conn.cursor()
            for name, sql in baseline_queries.items():
                safe_sql = _validate_read_only_sql(sql)
                cur.execute(safe_sql)
                rows = _fetch_dicts(cur)
                output[name] = rows
                output["query_trace"].append({"query_name": name, "sql": safe_sql, "returned_rows": len(rows)})

        # Emit structured events via callback
        if _callback_handler is not None:
            if "loan_status_distribution" in output:
                _callback_handler.on_table(
                    "Loan Status Distribution",
                    ["Status", "Count"],
                    [[r.get("loan_status", ""), r.get("cnt", 0)] for r in output["loan_status_distribution"]],
                )
            if "vintage_default_rates" in output:
                _callback_handler.on_chart_data("vintage_default_rates", output["vintage_default_rates"])

        output["next_step"] = (
            "Run additional targeted queries for any warning/fail finding "
            "to identify segment-level root causes."
        )
        return _ok(output)
    except Exception as exc:
        return _error(f"Failed to run baseline DQ scan: {exc}")


# ---------------------------------------------------------------------------
# Improvement 1: Bulk Column Profiler
# ---------------------------------------------------------------------------


@tool
def profile_all_columns(table_name: str = "", sample_size: int = 100000) -> dict:
    """Profile every column in one call: null rates, dtypes, cardinality,
    stats/top-values, and automatic feature type classification.

    Args:
        table_name: Table to profile (defaults to configured table).
        sample_size: Max rows to sample for profiling (0 = all rows).
    """
    settings = get_settings()
    table_name = table_name or settings.db_table
    try:
        with _open_db() as conn:
            if sample_size > 0:
                df = pd.read_sql_query(
                    f'SELECT * FROM "{table_name}" ORDER BY RANDOM() LIMIT {sample_size}', conn
                )
            else:
                df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)

        profiles: List[Dict[str, Any]] = []
        feature_type_map: Dict[str, str] = {}
        high_null_cols: List[str] = []
        zero_variance_cols: List[str] = []
        type_counts: Dict[str, int] = {}

        for col_name in df.columns:
            col = df[col_name]
            total = len(col)
            nulls = int(col.isna().sum())
            null_rate = round(nulls / total, 4) if total else 0
            unique = int(col.nunique())

            ftype = _classify_feature_type(col, col_name)
            feature_type_map[col_name] = ftype
            type_counts[ftype] = type_counts.get(ftype, 0) + 1

            if null_rate > 0.5:
                high_null_cols.append(col_name)
            if unique <= 1:
                zero_variance_cols.append(col_name)

            profile: Dict[str, Any] = {
                "column": col_name,
                "dtype": str(col.dtype),
                "null_count": nulls,
                "null_rate": null_rate,
                "unique_count": unique,
                "feature_type": ftype,
            }

            if pd.api.types.is_numeric_dtype(col) and unique > 2:
                desc = col.describe()
                profile.update({
                    "mean": round(float(desc.get("mean", 0)), 4),
                    "std": round(float(desc.get("std", 0)), 4),
                    "min": float(desc.get("min", 0)),
                    "p25": float(desc.get("25%", 0)),
                    "median": float(desc.get("50%", 0)),
                    "p75": float(desc.get("75%", 0)),
                    "max": float(desc.get("max", 0)),
                })
            elif ftype in ("nominal", "binary", "ordinal"):
                vc = col.value_counts().head(10)
                profile["top_values"] = {str(k): int(v) for k, v in vc.items()}

            profiles.append(profile)

        summary = {
            "total_columns": len(df.columns),
            "sampled_rows": len(df),
            "by_type": type_counts,
            "high_null_cols": high_null_cols[:20],
            "zero_variance_cols": zero_variance_cols,
        }

        # Emit summary table via callback
        if _callback_handler is not None:
            _callback_handler.on_table(
                "Column Profile Summary",
                ["Column", "Type", "Null %", "Unique", "Feature Type"],
                [
                    [p["column"], p["dtype"], f"{p['null_rate']*100:.1f}%", p["unique_count"], p["feature_type"]]
                    for p in profiles[:30]
                ],
            )

        return _ok({
            "profiles": profiles,
            "feature_type_map": feature_type_map,
            "summary": summary,
        })
    except Exception as exc:
        return _error(f"Failed to profile columns: {exc}")


# ---------------------------------------------------------------------------
# Original single-column profiler (kept for targeted follow-up)
# ---------------------------------------------------------------------------


@tool
def profile_column(column: str, table_name: str = "") -> dict:
    """Profile a single column: distribution stats, percentiles, null rate, cardinality.

    Args:
        column: Column name to profile.
        table_name: Table (defaults to configured table).
    """
    settings = get_settings()
    table_name = table_name or settings.db_table
    try:
        with _open_db() as conn:
            df = pd.read_sql_query(
                f'SELECT "{column}" FROM "{table_name}"', conn
            )
        col = df[column]
        total = len(col)
        nulls = int(col.isna().sum())
        result: Dict[str, Any] = {
            "column": column,
            "total_rows": total,
            "null_count": nulls,
            "null_rate": round(nulls / total, 4) if total else 0,
            "dtype": str(col.dtype),
            "unique_count": int(col.nunique()),
        }
        if pd.api.types.is_numeric_dtype(col):
            desc = col.describe()
            result.update({
                "mean": round(float(desc["mean"]), 4),
                "std": round(float(desc["std"]), 4),
                "min": float(desc["min"]),
                "p25": float(desc["25%"]),
                "median": float(desc["50%"]),
                "p75": float(desc["75%"]),
                "max": float(desc["max"]),
                "p01": round(float(col.quantile(0.01)), 4),
                "p99": round(float(col.quantile(0.99)), 4),
            })
        else:
            vc = col.value_counts().head(10)
            result["top_values"] = {str(k): int(v) for k, v in vc.items()}
        return _ok(result)
    except Exception as exc:
        return _error(f"Failed to profile column '{column}': {exc}")


@tool
def compute_psi(
    reference_column: str,
    comparison_column: str,
    buckets: int = 10,
    table_name: str = "",
) -> dict:
    """Calculate Population Stability Index between two subsets.

    Args:
        reference_column: SQL expression for reference subset (e.g., train).
        comparison_column: SQL expression for comparison subset (e.g., test).
        buckets: Number of bins.
        table_name: Table name.
    """
    settings = get_settings()
    table_name = table_name or settings.db_table
    try:
        with _open_db() as conn:
            ref = pd.read_sql_query(
                f'SELECT {reference_column} AS val FROM "{table_name}" '
                f'WHERE {reference_column} IS NOT NULL', conn
            )["val"]
            comp = pd.read_sql_query(
                f'SELECT {comparison_column} AS val FROM "{table_name}" '
                f'WHERE {comparison_column} IS NOT NULL', conn
            )["val"]

        breakpoints = np.linspace(0, 100, buckets + 1)
        edges = np.percentile(ref, breakpoints)
        edges[0] = -np.inf
        edges[-1] = np.inf

        ref_counts = np.histogram(ref, bins=edges)[0]
        comp_counts = np.histogram(comp, bins=edges)[0]

        ref_pct = ref_counts / ref_counts.sum()
        comp_pct = comp_counts / comp_counts.sum()

        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        comp_pct = np.where(comp_pct == 0, 0.0001, comp_pct)

        psi = float(np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)))

        status = "PASS" if psi < 0.10 else ("WARN" if psi < 0.25 else "FAIL")
        return _ok({"psi": round(psi, 6), "status": status, "buckets": buckets})
    except Exception as exc:
        return _error(f"Failed to compute PSI: {exc}")


@tool
def run_outlier_detection(column: str, method: str = "both", table_name: str = "") -> dict:
    """Detect outliers using Z-score and IQR methods.

    Args:
        column: Numeric column to check.
        method: 'zscore', 'iqr', or 'both'.
        table_name: Table name.
    """
    settings = get_settings()
    table_name = table_name or settings.db_table
    try:
        with _open_db() as conn:
            s = pd.read_sql_query(
                f'SELECT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL', conn
            )[column].astype(float)

        result: Dict[str, Any] = {"column": column, "total": len(s)}

        if method in ("zscore", "both"):
            z = np.abs(sp_stats.zscore(s))
            result["zscore_outliers_gt4"] = int((z > 4.0).sum())
            result["zscore_outlier_rate"] = round(float((z > 4.0).mean()), 6)

        if method in ("iqr", "both"):
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
            outliers = ((s < lower) | (s > upper)).sum()
            result["iqr_outliers_3x"] = int(outliers)
            result["iqr_outlier_rate"] = round(float(outliers / len(s)), 6)
            result["iqr_lower_bound"] = round(float(lower), 4)
            result["iqr_upper_bound"] = round(float(upper), 4)

        return _ok(result)
    except Exception as exc:
        return _error(f"Failed to detect outliers for '{column}': {exc}")


# ---------------------------------------------------------------------------
# Improvement 2: Missing Value Pattern Analyzer
# ---------------------------------------------------------------------------


@tool
def analyze_missing_patterns(
    table_name: str = "",
    group_by_columns: str = "grade,purpose",
    top_n_patterns: int = 20,
) -> dict:
    """Analyze missing value patterns: co-missingness, segmented null rates,
    and simplified MCAR testing.

    Args:
        table_name: Table to analyze (defaults to configured table).
        group_by_columns: Comma-separated columns to segment null rates by.
        top_n_patterns: Number of top missing patterns to report.
    """
    settings = get_settings()
    table_name = table_name or settings.db_table
    try:
        with _open_db() as conn:
            df = pd.read_sql_query(
                f'SELECT * FROM "{table_name}" ORDER BY RANDOM() LIMIT 100000', conn
            )

        total_rows = len(df)
        null_matrix = df.isnull()

        # --- Top missing patterns (combinations of null columns) ---
        # Create a string key for each row's missing pattern
        cols_with_nulls = [c for c in df.columns if null_matrix[c].any()]
        if cols_with_nulls:
            pattern_keys = null_matrix[cols_with_nulls].apply(
                lambda row: "|".join(sorted(c for c, v in row.items() if v)), axis=1
            )
            pattern_counts = pattern_keys.value_counts().head(top_n_patterns)
            top_patterns = [
                {
                    "columns_missing": p.split("|") if p else [],
                    "row_count": int(cnt),
                    "pct": round(cnt / total_rows * 100, 2),
                }
                for p, cnt in pattern_counts.items()
            ]
        else:
            top_patterns = [{"columns_missing": [], "row_count": total_rows, "pct": 100.0}]

        # --- Co-missingness (correlation of null indicators) ---
        co_missing_pairs: List[Dict[str, Any]] = []
        if len(cols_with_nulls) >= 2:
            null_indicators = null_matrix[cols_with_nulls].astype(int)
            corr = null_indicators.corr()
            for i, col_a in enumerate(cols_with_nulls):
                for col_b in cols_with_nulls[i + 1:]:
                    r = corr.loc[col_a, col_b]
                    if abs(r) > 0.3:
                        co_missing_pairs.append({
                            "col_a": col_a,
                            "col_b": col_b,
                            "correlation": round(float(r), 4),
                        })
            co_missing_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            co_missing_pairs = co_missing_pairs[:20]

        # --- Segmented null rates ---
        segmented: Dict[str, Any] = {}
        group_cols = [c.strip() for c in group_by_columns.split(",") if c.strip() in df.columns]
        for gcol in group_cols:
            seg_rates: Dict[str, Dict[str, float]] = {}
            for group_val, group_df in df.groupby(gcol):
                key = str(group_val)
                rates = {}
                for nc in cols_with_nulls[:15]:
                    rates[nc] = round(float(group_df[nc].isna().mean()), 4)
                seg_rates[key] = rates
            segmented[gcol] = seg_rates

        # --- Simplified MCAR test (t-test on observed values) ---
        mcar_results: List[Dict[str, Any]] = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        test_cols = [c for c in cols_with_nulls if c in numeric_cols][:10]
        for test_col in test_cols:
            mask = df[test_col].isna()
            if mask.sum() < 10 or (~mask).sum() < 10:
                continue
            # Test: does the mean of other numeric columns differ when test_col is missing?
            for other_col in numeric_cols[:5]:
                if other_col == test_col:
                    continue
                grp_present = df.loc[~mask, other_col].dropna()
                grp_missing = df.loc[mask, other_col].dropna()
                if len(grp_present) < 10 or len(grp_missing) < 10:
                    continue
                t_stat, p_val = sp_stats.ttest_ind(grp_present, grp_missing, equal_var=False)
                if p_val < 0.05:
                    mcar_results.append({
                        "null_column": test_col,
                        "test_column": other_col,
                        "t_statistic": round(float(t_stat), 4),
                        "p_value": round(float(p_val), 6),
                        "interpretation": "MAR likely" if p_val < 0.01 else "Possibly MAR",
                    })
        mcar_results.sort(key=lambda x: x["p_value"])
        mcar_results = mcar_results[:15]

        result = {
            "total_rows_sampled": total_rows,
            "columns_with_nulls": len(cols_with_nulls),
            "top_patterns": top_patterns,
            "co_missing_pairs": co_missing_pairs,
            "segmented_null_rates": segmented,
            "mcar_test_results": mcar_results,
            "summary": {
                "total_null_cols": len(cols_with_nulls),
                "mcar_violations": len(mcar_results),
                "high_correlation_pairs": len([p for p in co_missing_pairs if abs(p["correlation"]) > 0.7]),
            },
        }

        # Emit co-missing pairs as table
        if _callback_handler is not None and co_missing_pairs:
            _callback_handler.on_table(
                "Co-Missing Column Pairs",
                ["Column A", "Column B", "Correlation"],
                [[p["col_a"], p["col_b"], f"{p['correlation']:.3f}"] for p in co_missing_pairs[:10]],
            )

        return _ok(result)
    except Exception as exc:
        return _error(f"Failed to analyze missing patterns: {exc}")


# ---------------------------------------------------------------------------
# Improvement 3: Class Imbalance Assessment + SMOTE Integration
# ---------------------------------------------------------------------------


@tool
def assess_class_imbalance(
    table_name: str = "",
    target_expression: str = "CASE WHEN loan_status = 'Charged Off' THEN 1 ELSE 0 END",
    group_by_vintage: bool = True,
) -> dict:
    """Assess class imbalance for the target variable and generate a SMOTE
    recommendation. Segments default rate by vintage year.

    Args:
        table_name: Table to analyze (defaults to configured table).
        target_expression: SQL expression for the binary target.
        group_by_vintage: Whether to compute vintage-level default rates.
    """
    settings = get_settings()
    table_name = table_name or settings.db_table
    try:
        # Overall class distribution
        with _open_db() as conn:
            class_df = pd.read_sql_query(
                f'SELECT {target_expression} AS target, COUNT(*) AS cnt '
                f'FROM "{table_name}" '
                f'WHERE loan_status IN ("Fully Paid", "Charged Off") '
                f'GROUP BY 1',
                conn,
            )

        class_dist = {int(row["target"]): int(row["cnt"]) for _, row in class_df.iterrows()}
        total = sum(class_dist.values())
        minority_count = min(class_dist.values()) if class_dist else 0
        majority_count = max(class_dist.values()) if class_dist else 0
        minority_ratio = minority_count / total if total else 0

        # Severity classification
        if minority_ratio < 0.05:
            severity = "severe"
        elif minority_ratio < 0.15:
            severity = "moderate"
        elif minority_ratio < 0.30:
            severity = "mild"
        else:
            severity = "none"

        # SMOTE recommendation
        from backend.enhancements.smote_handler import SMOTEConfig, calculate_smote_samples_needed, get_class_weights
        config = SMOTEConfig()
        smote_needed = minority_ratio < config.minority_threshold
        smote_rec = {
            "enabled": smote_needed,
            "reason": f"Minority ratio {minority_ratio:.3f} {'<' if smote_needed else '>='} threshold {config.minority_threshold}",
            "samples_needed": calculate_smote_samples_needed(
                np.array([0] * majority_count + [1] * minority_count)
            ) if smote_needed else 0,
        }

        # Class weights (alternative to SMOTE)
        y_arr = np.array([0] * class_dist.get(0, 0) + [1] * class_dist.get(1, 0))
        class_weights = get_class_weights(y_arr) if len(y_arr) > 0 else {}

        # Vintage-level default rates
        vintage_rates: List[Dict[str, Any]] = []
        if group_by_vintage:
            with _open_db() as conn:
                vdf = pd.read_sql_query(
                    f'SELECT issue_d, '
                    f'COUNT(*) AS total, '
                    f'SUM({target_expression}) AS defaults '
                    f'FROM "{table_name}" '
                    f'WHERE loan_status IN ("Fully Paid", "Charged Off") '
                    f'GROUP BY issue_d ORDER BY issue_d',
                    conn,
                )
            for _, row in vdf.iterrows():
                rate = row["defaults"] / row["total"] if row["total"] > 0 else 0
                vintage_rates.append({
                    "vintage": str(row["issue_d"]),
                    "total": int(row["total"]),
                    "defaults": int(row["defaults"]),
                    "default_rate": round(float(rate), 4),
                })

        result = {
            "class_distribution": class_dist,
            "total_resolved": total,
            "minority_ratio": round(minority_ratio, 4),
            "imbalance_severity": severity,
            "smote_recommendation": smote_rec,
            "class_weights": {str(k): round(v, 4) for k, v in class_weights.items()},
            "vintage_default_rates": vintage_rates[-48:],  # Last 48 vintages
        }

        # Emit class distribution as table
        if _callback_handler is not None:
            _callback_handler.on_table(
                "Class Distribution",
                ["Class", "Count", "Pct"],
                [[str(k), v, f"{v/total*100:.1f}%"] for k, v in class_dist.items()],
            )

        return _ok(result)
    except Exception as exc:
        return _error(f"Failed to assess class imbalance: {exc}")


# ---------------------------------------------------------------------------
# Improvement 6: Vintage-Stratified PSI + Multivariate Drift
# ---------------------------------------------------------------------------


def _compute_psi_arrays(ref: np.ndarray, comp: np.ndarray, buckets: int = 10) -> float:
    """Compute PSI between two numeric arrays."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    edges = np.percentile(ref, breakpoints)
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts = np.histogram(ref, bins=edges)[0]
    comp_counts = np.histogram(comp, bins=edges)[0]
    ref_pct = np.where(ref_counts == 0, 0.0001, ref_counts / ref_counts.sum())
    comp_pct = np.where(comp_counts == 0, 0.0001, comp_counts / comp_counts.sum())
    return float(np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)))


@tool
def run_vintage_drift_analysis(
    table_name: str = "",
    date_column: str = "issue_d",
    feature_columns: str = "loan_amnt,int_rate,annual_inc,dti,fico_range_low,revol_util,open_acc",
    train_cutoff_year: int = 2015,
    test_cutoff_year: int = 2017,
) -> dict:
    """Run vintage-stratified PSI for multiple features and compute a multivariate
    drift score across train/validation/test splits.

    Args:
        table_name: Table to analyze.
        date_column: Date column for vintage extraction.
        feature_columns: Comma-separated feature columns to analyze.
        train_cutoff_year: Last year of training period (inclusive).
        test_cutoff_year: First year of test period (inclusive).
    """
    settings = get_settings()
    table_name = table_name or settings.db_table
    features = [f.strip() for f in feature_columns.split(",") if f.strip()]
    try:
        cols_sql = ", ".join(f'"{c}"' for c in features + [date_column])
        with _open_db() as conn:
            df = pd.read_sql_query(
                f'SELECT {cols_sql} FROM "{table_name}" '
                f'WHERE loan_status IN ("Fully Paid", "Charged Off")',
                conn,
            )

        # Parse vintage year
        def _parse_year(val):
            try:
                return pd.to_datetime(val, format="%b-%Y").year
            except Exception:
                try:
                    return int(val)
                except Exception:
                    return None

        df["_year"] = df[date_column].apply(_parse_year)
        df = df.dropna(subset=["_year"])
        df["_year"] = df["_year"].astype(int)

        train = df[df["_year"] <= train_cutoff_year]
        val = df[df["_year"] == train_cutoff_year + 1]
        test = df[df["_year"] >= test_cutoff_year]

        # Per-feature PSI
        per_feature_psi: List[Dict[str, Any]] = []
        for feat in features:
            if feat not in df.columns:
                continue
            ref = train[feat].dropna().astype(float).values
            if len(ref) < 100:
                continue
            psi_val = psi_test = None
            status_val = status_test = "N/A"
            if len(val[feat].dropna()) >= 50:
                psi_val = round(_compute_psi_arrays(ref, val[feat].dropna().astype(float).values), 6)
                status_val = "PASS" if psi_val < 0.10 else ("WARN" if psi_val < 0.25 else "FAIL")
            if len(test[feat].dropna()) >= 50:
                psi_test = round(_compute_psi_arrays(ref, test[feat].dropna().astype(float).values), 6)
                status_test = "PASS" if psi_test < 0.10 else ("WARN" if psi_test < 0.25 else "FAIL")
            per_feature_psi.append({
                "feature": feat,
                "psi_train_val": psi_val,
                "status_train_val": status_val,
                "psi_train_test": psi_test,
                "status_train_test": status_test,
            })

        # Multivariate drift (simplified MMD via mean-embedding distance)
        numeric_features = [f for f in features if f in train.columns and pd.api.types.is_numeric_dtype(train[f])]
        drift_score = None
        drift_pvalue = None
        if numeric_features and len(train) >= 100 and len(test) >= 50:
            train_mat = train[numeric_features].dropna().values
            test_mat = test[numeric_features].dropna().values
            # Normalize
            mu = train_mat.mean(axis=0)
            sigma = train_mat.std(axis=0)
            sigma[sigma == 0] = 1
            train_norm = (train_mat - mu) / sigma
            test_norm = (test_mat - mu) / sigma
            # Observed statistic: L2 distance between means
            observed = float(np.linalg.norm(train_norm.mean(axis=0) - test_norm.mean(axis=0)))
            # Permutation test (100 permutations)
            combined = np.vstack([train_norm, test_norm])
            n_train = len(train_norm)
            perm_stats = []
            rng = np.random.RandomState(42)
            for _ in range(100):
                perm = rng.permutation(len(combined))
                perm_a = combined[perm[:n_train]]
                perm_b = combined[perm[n_train:]]
                perm_stats.append(float(np.linalg.norm(perm_a.mean(axis=0) - perm_b.mean(axis=0))))
            drift_score = round(observed, 6)
            drift_pvalue = round(float(np.mean(np.array(perm_stats) >= observed)), 4)

        interpretation = "No significant multivariate drift"
        if drift_pvalue is not None and drift_pvalue < 0.05:
            interpretation = "Significant multivariate drift detected (p < 0.05)"

        result = {
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            "per_feature_psi": per_feature_psi,
            "multivariate_drift_score": drift_score,
            "multivariate_drift_pvalue": drift_pvalue,
            "interpretation": interpretation,
        }

        # Emit PSI results as table
        if _callback_handler is not None and per_feature_psi:
            _callback_handler.on_table(
                "Vintage PSI Analysis",
                ["Feature", "PSI (Train→Val)", "Status", "PSI (Train→Test)", "Status"],
                [
                    [p["feature"], p["psi_train_val"], p["status_train_val"], p["psi_train_test"], p["status_train_test"]]
                    for p in per_feature_psi
                ],
            )

        return _ok(result)
    except Exception as exc:
        return _error(f"Failed to run vintage drift analysis: {exc}")


# ---------------------------------------------------------------------------
# Improvement 7: DQ Scorecard Emitter
# ---------------------------------------------------------------------------


@tool
def emit_dq_result(
    test_id: str,
    test_name: str,
    status: str,
    value: str,
    threshold: str,
    evidence: str = "",
) -> dict:
    """Emit a data quality test result to the live DQ scorecard dashboard.

    Call this after evaluating each DQ test (DQ-01 through DQ-10).

    Args:
        test_id: Test identifier (e.g., 'DQ-01').
        test_name: Human-readable test name (e.g., 'Completeness').
        status: Test result: 'PASS', 'WARN', or 'FAIL'.
        value: Measured value as string (e.g., '0.023').
        threshold: Threshold as string (e.g., '< 0.05').
        evidence: Brief evidence/explanation.
    """
    if _callback_handler is not None:
        _callback_handler.on_dq_test(
            test_id=test_id,
            test_name=test_name,
            status=status.upper(),
            value=value,
            threshold=threshold,
            evidence=evidence,
        )
    return _ok({
        "test_id": test_id,
        "test_name": test_name,
        "status": status.upper(),
        "emitted": _callback_handler is not None,
    })


# ---------------------------------------------------------------------------
# Improvement 4: Data Lineage Tracker (augmented write_cleaned_dataset)
# ---------------------------------------------------------------------------


@tool
def write_cleaned_dataset(output_dir: str, run_id: str = "") -> dict:
    """Apply the full 6-step cleaning pipeline with lineage tracking.

    Produces:
    - cleaned_features.parquet (features only, no leakage)
    - targets.parquet (default_flag, lgd, ead, issue_year)
    - leakage_cols.parquet (preserved for LGD/EAD target construction)
    - data_lineage.json (detailed audit trail per step)
    - handoff.json

    Args:
        output_dir: Directory to write output files.
        run_id: Pipeline run identifier.
    """
    settings = get_settings()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lineage_steps: List[Dict[str, Any]] = []

    try:
        # --- Load full dataset ---
        with _open_db() as conn:
            df = pd.read_sql_query(f'SELECT * FROM "{settings.db_table}"', conn)

        initial_rows = len(df)
        initial_cols = len(df.columns)

        # --- Step 0: Preserve leakage columns for LGD/EAD ---
        leakage_cols_present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
        leakage_df = df[["id"] + leakage_cols_present].copy() if leakage_cols_present else pd.DataFrame()
        lineage_steps.append({
            "step": 0, "name": "Preserve Leakage Columns",
            "columns_preserved": leakage_cols_present,
            "count": len(leakage_cols_present),
        })

        # --- Step 1: Drop leakage + non-useful columns ---
        cols_before_drop = list(df.columns)
        cols_to_drop = [c for c in LEAKAGE_COLUMNS + DROP_COLUMNS if c in df.columns]
        cols_to_drop += [c for c in df.columns if c.startswith("settlement_") or c.startswith("hardship_")]
        cols_to_drop = list(set(cols_to_drop))
        df = df.drop(columns=cols_to_drop, errors="ignore")
        lineage_steps.append({
            "step": 1, "name": "Drop Leakage & Non-Useful Columns",
            "rows_before": initial_rows, "rows_after": len(df),
            "columns_before": len(cols_before_drop), "columns_after": len(df.columns),
            "columns_dropped": cols_to_drop,
        })

        # --- Step 2: Filter to resolved loans ---
        rows_before_filter = len(df)
        df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
        resolved_rows = len(df)
        lineage_steps.append({
            "step": 2, "name": "Filter to Resolved Loans",
            "rows_before": rows_before_filter, "rows_after": resolved_rows,
            "rows_dropped": rows_before_filter - resolved_rows,
            "kept_statuses": ["Fully Paid", "Charged Off"],
        })

        # --- Construct targets BEFORE cleaning features ---
        targets = pd.DataFrame(index=df.index)
        targets["default_flag"] = (df["loan_status"] == "Charged Off").astype(int)

        def _parse_issue_d(val):
            try:
                return pd.to_datetime(val, format="%b-%Y")
            except Exception:
                return pd.NaT

        issue_dates = df["issue_d"].apply(_parse_issue_d)
        targets["issue_year"] = issue_dates.dt.year
        targets["issue_d"] = issue_dates

        if "recoveries" in leakage_df.columns and "collection_recovery_fee" in leakage_df.columns:
            leakage_aligned = leakage_df.loc[df.index]
            net_rec = leakage_aligned["recoveries"].fillna(0) - leakage_aligned["collection_recovery_fee"].fillna(0)
            funded = df["funded_amnt"].replace(0, np.nan)
            targets["lgd"] = (1 - net_rec / funded).clip(0, 1)
        else:
            targets["lgd"] = np.nan

        if "out_prncp" in leakage_df.columns:
            targets["ead"] = leakage_df.loc[df.index, "out_prncp"].fillna(0)
        else:
            targets["ead"] = df["funded_amnt"]

        targets["ccf"] = targets["ead"] / df["funded_amnt"].replace(0, np.nan)

        # --- Step 3: Type coercion ---
        type_coercions: List[Dict[str, str]] = []
        if "int_rate" in df.columns:
            df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "", regex=False).str.strip()
            df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")
            type_coercions.append({"column": "int_rate", "action": "strip %, to float"})

        if "term" in df.columns:
            df["term"] = df["term"].astype(str).str.replace("months", "", regex=False).str.strip()
            df["term"] = pd.to_numeric(df["term"], errors="coerce")
            type_coercions.append({"column": "term", "action": "strip months, to int"})

        if "emp_length" in df.columns:
            def _parse_emp(val):
                s = str(val).strip().lower()
                if s in ("nan", "", "none"):
                    return np.nan
                if "< 1" in s:
                    return 0
                if "10+" in s:
                    return 10
                m = re.search(r"(\d+)", s)
                return int(m.group(1)) if m else np.nan
            df["emp_length"] = df["emp_length"].apply(_parse_emp)
            type_coercions.append({"column": "emp_length", "action": "parse text to numeric"})

        if "revol_util" in df.columns:
            df["revol_util"] = df["revol_util"].astype(str).str.replace("%", "", regex=False).str.strip()
            df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")
            type_coercions.append({"column": "revol_util", "action": "strip %, to float"})

        df = df.drop(columns=["loan_status", "issue_d"], errors="ignore")
        lineage_steps.append({
            "step": 3, "name": "Type Coercion",
            "coercions": type_coercions,
        })

        # --- Step 4: Missing value imputation ---
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        imputation_details: List[Dict[str, Any]] = []

        if "grade" in df.columns:
            for col in numeric_cols:
                if df[col].isna().any():
                    null_before = int(df[col].isna().sum())
                    medians = df.groupby("grade")[col].transform("median")
                    df[col] = df[col].fillna(medians)
                    df[col] = df[col].fillna(df[col].median())
                    imputation_details.append({
                        "column": col, "method": "grade-grouped median",
                        "nulls_before": null_before, "nulls_after": int(df[col].isna().sum()),
                    })
        else:
            for col in numeric_cols:
                if df[col].isna().any():
                    null_before = int(df[col].isna().sum())
                    med = df[col].median()
                    df[col] = df[col].fillna(med)
                    imputation_details.append({
                        "column": col, "method": "global median",
                        "nulls_before": null_before, "nulls_after": int(df[col].isna().sum()),
                        "imputed_value": round(float(med), 4) if not pd.isna(med) else None,
                    })

        for col in cat_cols:
            if df[col].isna().any():
                null_before = int(df[col].isna().sum())
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])
                    imputation_details.append({
                        "column": col, "method": "mode",
                        "nulls_before": null_before, "nulls_after": int(df[col].isna().sum()),
                        "imputed_value": str(mode_val.iloc[0]),
                    })

        lineage_steps.append({
            "step": 4, "name": "Missing Value Imputation",
            "columns_imputed": len(imputation_details),
            "details": imputation_details[:30],
        })

        # --- Step 5: Winsorization (using credit risk config) ---
        try:
            from backend.enhancements.winsorization_config import (
                apply_winsorization, create_credit_risk_winsorize_config,
            )
            win_config = create_credit_risk_winsorize_config()
            winsorize_targets = [c for c in WINSORIZE_COLS if c in df.columns]
            df, win_info = apply_winsorization(df, win_config, columns=winsorize_targets)
            lineage_steps.append({
                "step": 5, "name": "Winsorization",
                "method": "credit_risk_config",
                "columns_winsorized": len(winsorize_targets),
                "values_clipped": int(win_info.get("values_clipped", 0)),
                "thresholds": win_info.get("thresholds", {}),
                "clipping_rates": {k: round(v, 4) for k, v in win_info.get("clipping_rates", {}).items()},
            })
        except Exception:
            # Fallback to simple percentile clipping
            win_details: Dict[str, Any] = {}
            for col in WINSORIZE_COLS:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    p01 = df[col].quantile(0.01)
                    p99 = df[col].quantile(0.99)
                    clipped = int(((df[col] < p01) | (df[col] > p99)).sum())
                    df[col] = df[col].clip(p01, p99)
                    win_details[col] = {"lower": round(float(p01), 4), "upper": round(float(p99), 4), "clipped": clipped}
            lineage_steps.append({
                "step": 5, "name": "Winsorization",
                "method": "percentile_fallback",
                "details": win_details,
            })

        # --- Step 6: Categorical encoding ---
        encoding_details: List[Dict[str, Any]] = []
        grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        if "grade" in df.columns:
            df["grade_ord"] = df["grade"].map(grade_map)
            df = df.drop(columns=["grade"])
            encoding_details.append({"column": "grade", "method": "ordinal", "mapping": grade_map})

        if "sub_grade" in df.columns:
            unique_sg = sorted(df["sub_grade"].dropna().unique())
            sg_map = {sg: i + 1 for i, sg in enumerate(unique_sg)}
            df["sub_grade_ord"] = df["sub_grade"].map(sg_map)
            df = df.drop(columns=["sub_grade"])
            encoding_details.append({"column": "sub_grade", "method": "ordinal", "unique_values": len(sg_map)})

        if "purpose" in df.columns:
            purpose_risk = targets.groupby(df["purpose"])["default_flag"].mean()
            df["purpose_risk"] = df["purpose"].map(purpose_risk)
            df = df.drop(columns=["purpose"])
            encoding_details.append({"column": "purpose", "method": "risk_encoding", "unique_values": len(purpose_risk)})

        ohe_cols = [c for c in ["home_ownership", "verification_status",
                                "initial_list_status", "application_type"]
                    if c in df.columns]
        if ohe_cols:
            df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)
            encoding_details.append({"columns": ohe_cols, "method": "one_hot", "drop_first": True})

        remaining_obj = df.select_dtypes(include=["object"]).columns.tolist()
        if remaining_obj:
            df = df.drop(columns=remaining_obj)
            encoding_details.append({"columns_dropped": remaining_obj, "method": "drop_remaining_object"})

        lineage_steps.append({
            "step": 6, "name": "Categorical Encoding",
            "encodings": encoding_details,
            "final_feature_count": len(df.columns),
        })

        # --- Write outputs ---
        df.to_parquet(out / "cleaned_features.parquet", index=False)
        targets.to_parquet(out / "targets.parquet", index=False)
        if not leakage_df.empty:
            leakage_df.loc[df.index].to_parquet(out / "leakage_cols.parquet", index=False)

        default_rate = float(targets["default_flag"].mean())

        # Write data lineage
        lineage = {
            "run_id": run_id,
            "initial_rows": initial_rows,
            "initial_cols": initial_cols,
            "final_rows": len(df),
            "final_cols": len(df.columns),
            "steps": lineage_steps,
        }
        (out / "data_lineage.json").write_text(json.dumps(lineage, indent=2, default=str))

        handoff = {
            "agent": "Data_Agent",
            "status": "success",
            "output_files": {
                "cleaned_features": str(out / "cleaned_features.parquet"),
                "targets": str(out / "targets.parquet"),
                "leakage_cols": str(out / "leakage_cols.parquet"),
                "data_lineage": str(out / "data_lineage.json"),
            },
            "metrics": {
                "initial_rows": initial_rows,
                "initial_cols": initial_cols,
                "resolved_rows": resolved_rows,
                "feature_count": len(df.columns),
                "default_rate": round(default_rate, 4),
                "default_count": int(targets["default_flag"].sum()),
                "fully_paid_count": int((targets["default_flag"] == 0).sum()),
                "leakage_cols_preserved": len(leakage_cols_present),
                "cols_dropped": len(cols_to_drop),
                "winsorized_cols": len([c for c in WINSORIZE_COLS if c in df.columns]),
            },
        }
        (out / "handoff.json").write_text(json.dumps(handoff, indent=2, default=str))

        return _ok(handoff)
    except Exception as exc:
        return _error(f"Failed to write cleaned dataset: {exc}")


# --- Collect all tools for agent registration ---
ALL_DATA_TOOLS = [
    list_tables,
    describe_table,
    get_data_dictionary_summary,
    run_sql_query,
    run_baseline_data_quality_scan,
    profile_all_columns,
    profile_column,
    compute_psi,
    run_outlier_detection,
    analyze_missing_patterns,
    assess_class_imbalance,
    run_vintage_drift_analysis,
    emit_dq_result,
    write_cleaned_dataset,
]
