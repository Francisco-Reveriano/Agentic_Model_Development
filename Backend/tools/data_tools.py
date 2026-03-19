"""Data Agent tools — refactored from Backend/Agents/Data_Agent.py for LendingClub.

Reuses the helper patterns (_ok, _error, _validate_read_only_sql, _fetch_dicts,
_open_db) but adapts all schema references from Trepp CMBS to LendingClub and
adds the new tools required by the PRD (DQ tests, PSI, outlier detection,
cleaning pipeline).
"""

from __future__ import annotations

import json
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

# ---------------------------------------------------------------------------
# SQL safety constants (unchanged from reference)
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
# Helpers (from reference Data_Agent.py with parameterized paths)
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


@tool
def run_baseline_data_quality_scan(
    table_name: str = "",
    key_column: str = "id",
    date_column: str = "issue_d",
) -> dict:
    """Run a baseline data-quality scan on the LendingClub dataset.

    Args:
        table_name: Table to scan (defaults to configured table).
        key_column: Primary key column.
        date_column: Date/vintage column.
    """
    settings = get_settings()
    table_name = table_name or settings.db_table

    baseline_queries = {
        "row_count": f'SELECT COUNT(*) AS row_count FROM "{table_name}"',
        "column_count": f'SELECT COUNT(*) AS column_count FROM pragma_table_info("{table_name}")',
        "loan_status_distribution": (
            f'SELECT loan_status, COUNT(*) AS cnt FROM "{table_name}" '
            f'GROUP BY 1 ORDER BY cnt DESC LIMIT 20'
        ),
        "vintage_coverage": (
            f'SELECT "{date_column}", COUNT(*) AS cnt FROM "{table_name}" '
            f'GROUP BY 1 ORDER BY 1 DESC LIMIT 48'
        ),
        "key_nulls": (
            f'SELECT COUNT(*) AS null_key_count FROM "{table_name}" '
            f'WHERE "{key_column}" IS NULL OR TRIM(CAST("{key_column}" AS TEXT)) = ""'
        ),
        "duplicate_keys": (
            f'SELECT "{key_column}", COUNT(*) AS cnt FROM "{table_name}" '
            f'GROUP BY 1 HAVING COUNT(*) > 1 ORDER BY cnt DESC LIMIT 50'
        ),
        "core_metric_ranges": (
            f'SELECT '
            f'MIN(loan_amnt) AS min_loan, MAX(loan_amnt) AS max_loan, '
            f'MIN(int_rate) AS min_rate, MAX(int_rate) AS max_rate, '
            f'MIN(annual_inc) AS min_inc, MAX(annual_inc) AS max_inc, '
            f'MIN(dti) AS min_dti, MAX(dti) AS max_dti, '
            f'MIN(fico_range_low) AS min_fico, MAX(fico_range_low) AS max_fico '
            f'FROM "{table_name}"'
        ),
        "grade_distribution": (
            f'SELECT grade, COUNT(*) AS cnt FROM "{table_name}" '
            f'GROUP BY 1 ORDER BY 1'
        ),
        "purpose_distribution": (
            f'SELECT purpose, COUNT(*) AS cnt FROM "{table_name}" '
            f'GROUP BY 1 ORDER BY cnt DESC LIMIT 20'
        ),
        "home_ownership_distribution": (
            f'SELECT home_ownership, COUNT(*) AS cnt FROM "{table_name}" '
            f'GROUP BY 1 ORDER BY cnt DESC'
        ),
    }

    try:
        output: Dict[str, Any] = {"table_name": table_name, "query_trace": []}
        with _open_db() as conn:
            cur = conn.cursor()
            for name, sql in baseline_queries.items():
                safe_sql = _validate_read_only_sql(sql)
                cur.execute(safe_sql)
                rows = _fetch_dicts(cur)
                output[name] = rows
                output["query_trace"].append({"query_name": name, "sql": safe_sql, "returned_rows": len(rows)})

        output["next_step"] = (
            "Run additional targeted queries for any warning/fail finding "
            "to identify segment-level root causes."
        )
        return _ok(output)
    except Exception as exc:
        return _error(f"Failed to run baseline DQ scan: {exc}")


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

        # Avoid division by zero
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


@tool
def write_cleaned_dataset(output_dir: str, run_id: str = "") -> dict:
    """Apply the full 6-step cleaning pipeline and write cleaned dataset.

    Produces:
    - cleaned_features.parquet (features only, no leakage)
    - targets.parquet (default_flag, lgd, ead, issue_year)
    - leakage_cols.parquet (preserved for LGD/EAD target construction)
    - handoff.json

    Args:
        output_dir: Directory to write output files.
        run_id: Pipeline run identifier.
    """
    settings = get_settings()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        # --- Load full dataset ---
        with _open_db() as conn:
            df = pd.read_sql_query(f'SELECT * FROM "{settings.db_table}"', conn)

        initial_rows = len(df)
        initial_cols = len(df.columns)

        # --- Step 0: Preserve leakage columns for LGD/EAD ---
        leakage_cols_present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
        leakage_df = df[["id"] + leakage_cols_present].copy() if leakage_cols_present else pd.DataFrame()

        # --- Step 1: Drop leakage + non-useful columns ---
        cols_to_drop = [c for c in LEAKAGE_COLUMNS + DROP_COLUMNS if c in df.columns]
        # Also drop settlement_* and hardship_* columns
        cols_to_drop += [c for c in df.columns if c.startswith("settlement_") or c.startswith("hardship_")]
        cols_to_drop = list(set(cols_to_drop))
        df = df.drop(columns=cols_to_drop, errors="ignore")

        # --- Step 2: Filter to resolved loans ---
        df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
        resolved_rows = len(df)

        # --- Construct targets BEFORE cleaning features ---
        targets = pd.DataFrame(index=df.index)
        targets["default_flag"] = (df["loan_status"] == "Charged Off").astype(int)

        # Parse issue_d to extract issue_year for vintage split
        def _parse_issue_d(val):
            try:
                return pd.to_datetime(val, format="%b-%Y")
            except Exception:
                return pd.NaT

        issue_dates = df["issue_d"].apply(_parse_issue_d)
        targets["issue_year"] = issue_dates.dt.year
        targets["issue_d"] = issue_dates

        # LGD target (from leakage_df — only for defaults)
        if "recoveries" in leakage_df.columns and "collection_recovery_fee" in leakage_df.columns:
            leakage_aligned = leakage_df.loc[df.index]
            net_rec = leakage_aligned["recoveries"].fillna(0) - leakage_aligned["collection_recovery_fee"].fillna(0)
            funded = df["funded_amnt"].replace(0, np.nan)
            targets["lgd"] = (1 - net_rec / funded).clip(0, 1)
        else:
            targets["lgd"] = np.nan

        # EAD target
        if "out_prncp" in leakage_df.columns:
            targets["ead"] = leakage_df.loc[df.index, "out_prncp"].fillna(0)
        else:
            targets["ead"] = df["funded_amnt"]

        targets["ccf"] = targets["ead"] / df["funded_amnt"].replace(0, np.nan)

        # --- Step 3: Type coercion ---
        if "int_rate" in df.columns:
            df["int_rate"] = (
                df["int_rate"]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")

        if "term" in df.columns:
            df["term"] = (
                df["term"]
                .astype(str)
                .str.replace("months", "", regex=False)
                .str.strip()
            )
            df["term"] = pd.to_numeric(df["term"], errors="coerce")

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

        if "revol_util" in df.columns:
            df["revol_util"] = (
                df["revol_util"]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")

        # Drop loan_status (it IS the target, shouldn't be a feature)
        df = df.drop(columns=["loan_status", "issue_d"], errors="ignore")

        # --- Step 4: Missing value imputation ---
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        if "grade" in df.columns:
            for col in numeric_cols:
                if df[col].isna().any():
                    medians = df.groupby("grade")[col].transform("median")
                    df[col] = df[col].fillna(medians)
                    # Fill remaining with global median
                    df[col] = df[col].fillna(df[col].median())
        else:
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())

        for col in cat_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])

        # --- Step 5: Winsorization ---
        for col in WINSORIZE_COLS:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                p01 = df[col].quantile(0.01)
                p99 = df[col].quantile(0.99)
                df[col] = df[col].clip(p01, p99)

        # --- Step 6: Categorical encoding ---
        grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        if "grade" in df.columns:
            df["grade_ord"] = df["grade"].map(grade_map)
            df = df.drop(columns=["grade"])

        if "sub_grade" in df.columns:
            # Ordinal: A1=1, A2=2, ..., G5=35
            unique_sg = sorted(df["sub_grade"].dropna().unique())
            sg_map = {sg: i + 1 for i, sg in enumerate(unique_sg)}
            df["sub_grade_ord"] = df["sub_grade"].map(sg_map)
            df = df.drop(columns=["sub_grade"])

        # Risk-based encoding for purpose (default rate per category)
        if "purpose" in df.columns:
            purpose_risk = targets.groupby(df["purpose"])["default_flag"].mean()
            df["purpose_risk"] = df["purpose"].map(purpose_risk)
            df = df.drop(columns=["purpose"])

        # One-hot encode remaining categoricals
        ohe_cols = [c for c in ["home_ownership", "verification_status",
                                "initial_list_status", "application_type"]
                    if c in df.columns]
        if ohe_cols:
            df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)

        # Drop any remaining object columns
        remaining_obj = df.select_dtypes(include=["object"]).columns.tolist()
        if remaining_obj:
            df = df.drop(columns=remaining_obj)

        # --- Write outputs ---
        df.to_parquet(out / "cleaned_features.parquet", index=False)
        targets.to_parquet(out / "targets.parquet", index=False)
        if not leakage_df.empty:
            leakage_df.loc[df.index].to_parquet(out / "leakage_cols.parquet", index=False)

        default_rate = float(targets["default_flag"].mean())

        handoff = {
            "agent": "Data_Agent",
            "status": "success",
            "output_files": {
                "cleaned_features": str(out / "cleaned_features.parquet"),
                "targets": str(out / "targets.parquet"),
                "leakage_cols": str(out / "leakage_cols.parquet"),
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
    profile_column,
    compute_psi,
    run_outlier_detection,
    write_cleaned_dataset,
]
