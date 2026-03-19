import json
import os
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from strands import Agent, tool
from strands.models.anthropic import AnthropicModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "Data" / "Raw" / "Trepp_Data.db"
DEFAULT_DICTIONARY_PATH = PROJECT_ROOT / "Data" / "Raw" / "Trepp_Tables_Data_Dictionary (1).xlsx"
DEFAULT_TABLE = "my_table"

ALLOWED_QUERY_PREFIXES = ("select", "with", "pragma", "explain")
DISALLOWED_SQL_PATTERNS = (
    "insert ",
    "update ",
    "delete ",
    "drop ",
    "alter ",
    "create ",
    "attach ",
    "detach ",
    "replace ",
    "truncate ",
    "vacuum",
    "reindex",
)


def _ok(payload: Dict[str, Any]) -> dict:
    return {"status": "success", "content": [{"text": json.dumps(payload, default=str, indent=2)}]}


def _error(message: str) -> dict:
    return {"status": "error", "content": [{"text": message}]}


def _normalize_field_name(value: str) -> str:
    cleaned = str(value or "").strip().lower()
    cleaned = re.sub(r"\.\d+$", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]+", "", cleaned)
    return cleaned


@lru_cache(maxsize=1)
def _dictionary_lookup() -> Dict[str, Dict[str, Any]]:
    if not DEFAULT_DICTIONARY_PATH.exists():
        return {}

    workbook = pd.ExcelFile(DEFAULT_DICTIONARY_PATH)
    dictionary_sheets = ("TREPP_Loan1", "TREPP_Loan2", "TREPP_Proforma", "TREPP_Control")
    lookup: Dict[str, Dict[str, Any]] = {}

    for sheet_name in dictionary_sheets:
        if sheet_name not in workbook.sheet_names:
            continue
        df = pd.read_excel(DEFAULT_DICTIONARY_PATH, sheet_name=sheet_name)
        expected_cols = {"FIELD NAME", "DESCRIPTION", "FIELD TYPE"}
        if not expected_cols.issubset(df.columns):
            continue

        for _, row in df.iterrows():
            field_name = str(row.get("FIELD NAME", "")).strip()
            if not field_name or field_name.lower() == "nan":
                continue
            normalized = _normalize_field_name(field_name)
            if not normalized:
                continue
            lookup[normalized] = {
                "field_name": field_name,
                "description": str(row.get("DESCRIPTION", "")).strip(),
                "field_type": str(row.get("FIELD TYPE", "")).strip(),
                "sheet": sheet_name,
            }
    return lookup


def _open_db() -> sqlite3.Connection:
    if not DEFAULT_DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DEFAULT_DB_PATH}")
    return sqlite3.connect(f"file:{DEFAULT_DB_PATH}?mode=ro", uri=True)


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


@tool
def list_tables() -> dict:
    """
    List all non-system tables in the Trepp SQLite database.
    """
    try:
        with _open_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
            return _ok({"database_path": str(DEFAULT_DB_PATH), "tables": [row[0] for row in cur.fetchall()]})
    except Exception as exc:
        return _error(f"Failed to list tables: {exc}")


@tool
def describe_table(table_name: str = DEFAULT_TABLE) -> dict:
    """
    Describe a table's columns and attach dictionary definitions when available.

    Args:
        table_name: Table to inspect.
    """
    try:
        lookup = _dictionary_lookup()
        with _open_db() as conn:
            cur = conn.cursor()
            cur.execute(f'PRAGMA table_info("{table_name}")')
            rows = cur.fetchall()
            if not rows:
                return _error(f"Table '{table_name}' not found or has no columns.")

            columns: List[Dict[str, Any]] = []
            matched_count = 0
            for row in rows:
                column_name = row[1]
                normalized = _normalize_field_name(column_name)
                dict_item = lookup.get(normalized)
                if dict_item:
                    matched_count += 1
                columns.append(
                    {
                        "column_name": column_name,
                        "sqlite_type": row[2],
                        "not_null": bool(row[3]),
                        "primary_key": bool(row[5]),
                        "dictionary_match": dict_item,
                    }
                )

            return _ok(
                {
                    "table_name": table_name,
                    "column_count": len(columns),
                    "dictionary_match_count": matched_count,
                    "dictionary_unmatched_count": len(columns) - matched_count,
                    "columns": columns,
                }
            )
    except Exception as exc:
        return _error(f"Failed to describe table '{table_name}': {exc}")


@tool
def get_data_dictionary_summary() -> dict:
    """
    Summarize the Trepp data dictionary workbook and mappings to the current table.
    """
    try:
        lookup = _dictionary_lookup()
        if not lookup:
            return _error(f"Could not load dictionary workbook at: {DEFAULT_DICTIONARY_PATH}")

        with _open_db() as conn:
            cur = conn.cursor()
            cur.execute(f'PRAGMA table_info("{DEFAULT_TABLE}")')
            rows = cur.fetchall()
            table_cols = [row[1] for row in rows]
            normalized_table_cols = {_normalize_field_name(col): col for col in table_cols}

        matched = []
        unmatched = []
        for normalized, original in normalized_table_cols.items():
            if normalized in lookup:
                matched.append({"column_name": original, "dictionary": lookup[normalized]})
            else:
                unmatched.append(original)

        return _ok(
            {
                "dictionary_path": str(DEFAULT_DICTIONARY_PATH),
                "table_name": DEFAULT_TABLE,
                "table_column_count": len(table_cols),
                "dictionary_field_count": len(lookup),
                "matched_columns_count": len(matched),
                "unmatched_columns_count": len(unmatched),
                "unmatched_columns": unmatched,
                "matched_columns_sample": matched[:30],
            }
        )
    except Exception as exc:
        return _error(f"Failed to summarize data dictionary: {exc}")


@tool
def run_sql_query(query: str, limit: int = 200) -> dict:
    """
    Run a single read-only SQL query on Trepp data.

    Args:
        query: SQL (SELECT/WITH/PRAGMA/EXPLAIN only).
        limit: Maximum rows returned in tool output.
    """
    try:
        safe_query = _validate_read_only_sql(query)
        capped_limit = max(1, min(limit, 2000))
        wrapped = f"SELECT * FROM ({safe_query}) q LIMIT {capped_limit}"

        with _open_db() as conn:
            cur = conn.cursor()
            cur.execute(wrapped)
            data = _fetch_dicts(cur)

        return _ok(
            {
                "query": safe_query,
                "limit": capped_limit,
                "returned_rows": len(data),
                "rows": data,
            }
        )
    except Exception as exc:
        return _error(f"Failed to run query: {exc}")


@tool
def run_baseline_data_quality_scan(
    table_name: str = DEFAULT_TABLE,
    key_column: str = "masterloanidtrepp",
    date_column: str = "distdate",
) -> dict:
    """
    Run a baseline data-quality scan for model development.
    The agent should still add adaptive follow-up queries afterwards.
    """
    baseline_queries = {
        "row_count": f'SELECT COUNT(*) AS row_count FROM "{table_name}"',
        "column_count": f'SELECT COUNT(*) AS column_count FROM pragma_table_info("{table_name}")',
        "date_coverage": (
            f'SELECT MIN("{date_column}") AS min_distdate, '
            f'MAX("{date_column}") AS max_distdate, '
            f'COUNT(DISTINCT "{date_column}") AS distinct_distdates '
            f'FROM "{table_name}"'
        ),
        "monthly_volume": (
            f'SELECT SUBSTR(CAST("{date_column}" AS TEXT), 1, 6) AS yyyymm, COUNT(*) AS record_count '
            f'FROM "{table_name}" GROUP BY 1 ORDER BY 1 DESC LIMIT 36'
        ),
        "key_nulls": (
            f'SELECT COUNT(*) AS null_or_blank_key_count FROM "{table_name}" '
            f'WHERE "{key_column}" IS NULL OR TRIM(CAST("{key_column}" AS TEXT)) = ""'
        ),
        "duplicate_key_date": (
            f'SELECT "{key_column}", "{date_column}", COUNT(*) AS cnt FROM "{table_name}" '
            f'GROUP BY 1,2 HAVING COUNT(*) > 1 ORDER BY cnt DESC LIMIT 100'
        ),
        "core_metric_ranges": (
            f'SELECT MIN(ltv) AS min_ltv, MAX(ltv) AS max_ltv, '
            f'MIN(dscrnoi) AS min_dscrnoi, MAX(dscrnoi) AS max_dscrnoi, '
            f'MIN(beginbal) AS min_beginbal, MAX(beginbal) AS max_beginbal '
            f'FROM "{table_name}"'
        ),
        "status_distribution": (
            f'SELECT dlqderivedcd, COUNT(*) AS cnt FROM "{table_name}" '
            f'GROUP BY 1 ORDER BY cnt DESC LIMIT 50'
        ),
        "state_distribution": (
            f'SELECT state, COUNT(*) AS cnt FROM "{table_name}" '
            f'GROUP BY 1 ORDER BY cnt DESC LIMIT 100'
        ),
        "property_type_distribution": (
            f'SELECT proptype, COUNT(*) AS cnt FROM "{table_name}" '
            f'GROUP BY 1 ORDER BY cnt DESC LIMIT 100'
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
            "Run additional targeted queries for any warning/fail finding to identify segment-level root causes."
        )
        return _ok(output)
    except Exception as exc:
        return _error(f"Failed to run baseline data-quality scan: {exc}")


anthropic_model = AnthropicModel(
    client_args={
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
    },
    max_tokens=128000,
    model_id="claude-opus-4-6",
    params={
        "temperature": 1,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "max"},
    },
)

DATA_AGENT_PROMPT = f"""
You are Data_Agent, a model-development data quality specialist.

Data sources:
- SQLite DB: {DEFAULT_DB_PATH}
- Dictionary workbook: {DEFAULT_DICTIONARY_PATH}
- Primary table: {DEFAULT_TABLE}

Hard requirements:
1) First understand schema and dictionary mapping before conclusions.
2) Use SQL evidence for every claim.
3) Run multiple SQL queries, then generate additional adaptive follow-up queries whenever findings are inconclusive.
4) Never use non-read-only SQL.

Required workflow:
1. Call `list_tables`.
2. Call `describe_table` for target table.
3. Call `get_data_dictionary_summary`.
4. Call `run_baseline_data_quality_scan`.
5. For each key question below, generate and run additional SQL queries until evidence is sufficient:
   - Completeness (null rates overall and segmented by time/category).
   - Validity/ranges (out-of-range, impossible values, invalid codes).
   - Consistency (cross-field contradictions, e.g., balances/status relationships).
   - Uniqueness and grain integrity (`masterloanidtrepp` + `distdate`).
   - Temporal stability (distribution shifts and discontinuities over time).
   - Outliers (robust, segment-aware checks).

Adaptive query rule:
- Start with baseline.
- If any warning/fail appears, run follow-up drill-down queries by segments (`distdate`, `state`, `proptype`, status fields).
- Continue generating follow-up SQL while root cause is unclear.
- Stop only when each material issue has a supported explanation or is marked unresolved with low confidence.

Output format (always):
1.1 Data quality
1.1.1 Data Overview
1.2 Data assumptions and treatments
Data Quality Scorecard
Query Trace
Fit for Model Development Decision

Output content constraints:
- Rank top 10 issues by model risk.
- For each issue provide: evidence, likely impact, treatment recommendation (impute/winsorize/exclude/recode/manual review), confidence.
- In Query Trace, include baseline and additional queries plus why each additional query was created.
"""

Data_Agent = Agent(
    name="Data_Agent",
    system_prompt=DATA_AGENT_PROMPT,
    model=anthropic_model,
    tools=[
        list_tables,
        describe_table,
        get_data_dictionary_summary,
        run_sql_query,
        run_baseline_data_quality_scan,
    ],
)