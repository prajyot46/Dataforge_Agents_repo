from google.adk.agents import Agent
from google.cloud import storage
from google.adk.tools import load_artifacts
from google.genai import types
from io import StringIO
import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for Cloud Shell / Cloud Run
import matplotlib.pyplot as plt
import base64
import io


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BUCKET_NAME = 'hackathon-agent-data'
ORDERS_FOLDER = 'data/orders/'        # Fulfilment system orders
CT_ORDERS_FOLDER = 'data/orders_CT/'  # Website / CT orders


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_project_id() -> str:
    """Get project ID from environment (Cloud Shell sets these automatically)."""
    return (
        os.environ.get('PROJECT_ID') or
        os.environ.get('GOOGLE_CLOUD_PROJECT') or
        os.environ.get('DEVSHELL_PROJECT_ID') or
        ''
    )


def _load_csv_from_gcs(folder: str, label: str = 'orders') -> pd.DataFrame:
    """Generic helper: load the first CSV found in a GCS folder.

    Args:
        folder: GCS prefix/folder path (e.g. 'data/orders/')
        label: Human-readable name for error messages

    Raises:
        EnvironmentError: PROJECT_ID not available
        FileNotFoundError: No CSV found at the expected GCS path
        RuntimeError: CSV could not be parsed
    """
    project_id = get_project_id()
    if not project_id:
        raise EnvironmentError(
            "PROJECT_ID not set. In Cloud Shell run: export PROJECT_ID='your-project-id'"
        )

    client = storage.Client(project=project_id)
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=folder))
    csv_blob = next((b for b in blobs if b.name.endswith('.csv')), None)

    if not csv_blob:
        raise FileNotFoundError(
            f"No CSV found at gs://{BUCKET_NAME}/{folder}. "
            f"Check that {label} data has been uploaded."
        )

    try:
        return pd.read_csv(StringIO(csv_blob.download_as_text()))
    except Exception as e:
        raise RuntimeError(f"Failed to parse {label} CSV: {e}")


def _load_orders() -> pd.DataFrame:
    """Load fulfilment system orders from GCS."""
    return _load_csv_from_gcs(ORDERS_FOLDER, 'fulfilment orders')


def _load_ct_orders() -> pd.DataFrame:
    """Load website/CT orders from GCS."""
    return _load_csv_from_gcs(CT_ORDERS_FOLDER, 'website/CT orders')


def _df_to_response(df: pd.DataFrame, description: str) -> dict:
    """Convert a DataFrame result to a clean JSON-serialisable response."""
    return {
        "description": description,
        "rows": len(df),
        "columns": list(df.columns),
        "data": df.to_dict('records')
    }


def _error(msg: str, exception: Exception = None) -> dict:
    """Consistent error response."""
    result = {"error": msg}
    if exception:
        result["detail"] = str(exception)
        result["type"] = type(exception).__name__
    return result


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Convert a DataFrame to a Markdown table string."""
    if len(df) == 0:
        return "*No data*"
    show_df = df.head(max_rows)
    headers = list(show_df.columns)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in show_df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    result = "\n".join(lines)
    if len(df) > max_rows:
        result += f"\n\n*Showing {max_rows} of {len(df)} rows*"
    return result


# ---------------------------------------------------------------------------
# Tool 1 — Sample / Preview
# ---------------------------------------------------------------------------

def sample_orders(rows: int = 10) -> str:
    """Preview the first N rows of the orders table.

    Args:
        rows: Number of rows to return (default 10, max 100)
    """
    try:
        df = _load_orders()
        rows = min(rows, 100)
        text = f"📋 **Orders Table Preview** — {len(df)} total rows, {len(df.columns)} columns\n\n"
        text += f"**Columns:** {', '.join(df.columns)}\n\n"
        text += _df_to_markdown(df.head(rows), max_rows=rows)
        return text
    except Exception as e:
        return f"❌ Failed to sample orders: {e}"


# ---------------------------------------------------------------------------
# Tool 2 — Filter / Select
# ---------------------------------------------------------------------------

def filter_orders(column: str, value: str, rows: int = 50) -> str:
    """Filter orders where a column matches a value.

    Args:
        column: Column name to filter on (e.g., 'status', 'customer_id')
        value: Value to match (case-insensitive for strings)
        rows: Max rows to return (default 50, max 500)
    """
    try:
        df = _load_orders()

        if column not in df.columns:
            return f"❌ Column '{column}' not found. Available columns: {', '.join(df.columns)}"

        col = df[column]
        if col.dtype == object:
            mask = col.str.lower() == value.lower()
        else:
            mask = col == type(col.iloc[0])(value)

        total_matches = int(mask.sum())
        result = df[mask].head(min(rows, 500))
        text = f"🔍 **Orders where {column} = '{value}'** — {total_matches} match(es)\n\n"
        text += _df_to_markdown(result)
        return text
    except Exception as e:
        return f"❌ Failed to filter orders: {e}"


# ---------------------------------------------------------------------------
# Tool 3 — Aggregations
# ---------------------------------------------------------------------------

def _aggregate_internal(
    group_by: str = '',
    metric_column: str = None,
    aggregation: str = 'count'
) -> dict:
    """Internal aggregation returning dict (used by plot_orders).

    Args:
        group_by: Column to group by. Leave empty for overall total.
        metric_column: Column to aggregate (required for sum/avg/min/max)
        aggregation: One of 'count', 'sum', 'avg', 'min', 'max'
    """
    try:
        df = _load_orders()

        agg_map = {'count': 'count', 'sum': 'sum', 'avg': 'mean', 'min': 'min', 'max': 'max'}
        agg_fn = agg_map.get(aggregation.lower())
        if not agg_fn:
            return {"error": f"Unknown aggregation '{aggregation}'. Use: count, sum, avg, min, max"}

        # --- Overall aggregate (no grouping) ---
        if not group_by or group_by.strip() == '':
            if aggregation == 'count':
                return {
                    "description": "Total order count",
                    "total_count": len(df),
                    "rows": 1,
                    "data": [{"total_count": len(df)}]
                }
            if not metric_column:
                return {"error": f"'metric_column' is required for aggregation '{aggregation}'"}
            if metric_column not in df.columns:
                return {
                    "error": f"Column '{metric_column}' not found.",
                    "available_columns": list(df.columns)
                }
            value = float(df[metric_column].agg(agg_fn))
            label = f"{aggregation}_{metric_column}"
            return {
                "description": f"Overall {aggregation.upper()} of '{metric_column}'",
                "rows": 1,
                "columns": [label],
                "data": [{label: round(value, 2)}]
            }

        # --- Grouped aggregate ---
        if group_by not in df.columns:
            return {
                "error": f"Column '{group_by}' not found.",
                "available_columns": list(df.columns)
            }

        if aggregation == 'count':
            result = df.groupby(group_by).size().reset_index(name='count')
            result = result.sort_values('count', ascending=False)
        else:
            if not metric_column:
                return {"error": f"'metric_column' is required for aggregation '{aggregation}'"}
            if metric_column not in df.columns:
                return {
                    "error": f"Column '{metric_column}' not found.",
                    "available_columns": list(df.columns)
                }
            result = df.groupby(group_by)[metric_column].agg(agg_fn).reset_index()
            result.columns = [group_by, f"{aggregation}_{metric_column}"]
            result = result.sort_values(f"{aggregation}_{metric_column}", ascending=False)

        return _df_to_response(result, f"{aggregation.upper()} of orders grouped by '{group_by}'")
    except Exception as e:
        return _error("Failed to aggregate orders", e)


def aggregate_orders(
    group_by: str = '',
    metric_column: str = None,
    aggregation: str = 'count'
) -> str:
    """Group orders by a column and apply an aggregation.

    Args:
        group_by: Column to group by (e.g., 'status', 'customer_id').
                  Leave empty or pass '' for an overall total/average/etc.
        metric_column: Column to aggregate (required for sum/avg/min/max)
        aggregation: One of 'count', 'sum', 'avg', 'min', 'max' (default: 'count')
    """
    result = _aggregate_internal(group_by, metric_column, aggregation)
    if "error" in result:
        avail = result.get("available_columns")
        msg = f"❌ {result['error']}"
        if avail:
            msg += f"\nAvailable columns: {', '.join(avail)}"
        return msg
    text = f"📊 **{result['description']}**\n\n"
    if result.get('data'):
        tbl = pd.DataFrame(result['data'])
        text += _df_to_markdown(tbl, max_rows=30)
    return text


# ---------------------------------------------------------------------------
# Tool 4 — Plot
# ---------------------------------------------------------------------------

async def plot_orders(
    group_by: str,
    metric_column: str = None,
    aggregation: str = 'count',
    chart_type: str = 'bar',
    top_n: int = 15,
    tool_context = None
) -> dict:
    """Aggregate orders and return a chart as a base64-encoded PNG.

    Args:
        group_by: Column to group by (e.g., 'status', 'customer_id')
        metric_column: Column to aggregate (required for sum/avg/min/max)
        aggregation: One of 'count', 'sum', 'avg', 'min', 'max' (default: 'count')
        chart_type: 'bar', 'pie', or 'line' (default: 'bar')
        top_n: Only plot top N groups to keep chart readable (default: 15)
    """
    try:
        agg_result = _aggregate_internal(group_by, metric_column, aggregation)
        if "error" in agg_result:
            return agg_result

        rows = agg_result["data"][:top_n]
        labels = [str(r[group_by]) for r in rows]
        value_key = [k for k in rows[0].keys() if k != group_by][0]
        values = [r[value_key] for r in rows]

        fig, ax = plt.subplots(figsize=(10, 5))

        if chart_type == 'bar':
            ax.bar(labels, values, color='steelblue')
            ax.set_xlabel(group_by)
            ax.set_ylabel(value_key)
            plt.xticks(rotation=45, ha='right')
        elif chart_type == 'pie':
            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
        elif chart_type == 'line':
            ax.plot(labels, values, marker='o', color='steelblue')
            ax.set_xlabel(group_by)
            ax.set_ylabel(value_key)
            plt.xticks(rotation=45, ha='right')
        else:
            return {"error": f"Unknown chart_type '{chart_type}'. Use: bar, pie, line"}

        title = f"{aggregation.upper()} of orders by {group_by}"
        ax.set_title(title)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        plt.close(fig)
        buf.seek(0)
        image_bytes = buf.read()

        # Save as artifact following ADK pattern
        if tool_context:
            from google.genai import types
            artifact_filename = f"chart_{group_by}_{chart_type}.png"
            await tool_context.save_artifact(
                artifact_filename,
                types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            )
            return {
                "status": "success",
                "message": f"{title} - Chart saved as {artifact_filename}",
                "artifact_filename": artifact_filename,
                "chart_type": chart_type,
                "groups_shown": len(rows)
            }
        else:
            # Fallback if no tool_context (shouldn't happen in ADK)
            return {
                "status": "error",
                "message": "tool_context not available"
            }
    except Exception as e:
        return _error("Failed to plot orders", e)


# ---------------------------------------------------------------------------
# Tool 5 — Smart Suggestions / Proactive Insights
# ---------------------------------------------------------------------------

def suggest_analysis(context: str = '') -> str:
    """Analyze the orders dataset and suggest interesting follow-up analyses.

    Inspects column types, value distributions, date ranges, and data shape
    to recommend the most valuable next queries the user could ask.

    Args:
        context: Optional description of what the user just asked or explored,
                 so suggestions are contextually relevant.
    """
    try:
        df = _load_orders()
        suggestions = []
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        date_cols = []

        # Detect date columns
        for col in categorical_cols[:]:
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                if parsed.notna().sum() > len(df) * 0.5:
                    date_cols.append(col)
                    categorical_cols.remove(col)
            except Exception:
                pass

        # --- Date-based suggestions ---
        for col in date_cols:
            parsed = pd.to_datetime(df[col], errors='coerce')
            date_range = parsed.max() - parsed.min()
            suggestions.append({
                "suggestion": f"Trend analysis: plot order volume over time using '{col}'",
                "query_example": f"Show me a line chart of orders over time by {col}",
                "reason": f"'{col}' spans {date_range.days} days — great for spotting trends and seasonality"
            })
            if date_range.days > 30:
                suggestions.append({
                    "suggestion": f"Period comparison: compare recent vs earlier orders using '{col}'",
                    "query_example": f"How do orders this month compare to last month?",
                    "reason": "Period-over-period comparison reveals growth or decline"
                })

        # --- Categorical suggestions ---
        for col in categorical_cols:
            nunique = df[col].nunique()
            if 2 <= nunique <= 20:
                suggestions.append({
                    "suggestion": f"Distribution breakdown by '{col}' ({nunique} categories)",
                    "query_example": f"Show me a pie chart of orders by {col}",
                    "reason": f"Low-cardinality column — ideal for grouping and visualization"
                })
            elif nunique > 20:
                suggestions.append({
                    "suggestion": f"Top-N analysis on '{col}' ({nunique} unique values)",
                    "query_example": f"Who are the top 10 {col}s by order count?",
                    "reason": f"High-cardinality column — a Top-N view highlights key players"
                })

        # --- Numeric suggestions ---
        for col in numeric_cols:
            stats = df[col].describe()
            if stats['std'] > 0:
                skew = df[col].skew()
                suggestions.append({
                    "suggestion": f"Statistical deep-dive on '{col}' (mean={stats['mean']:.2f}, std={stats['std']:.2f})",
                    "query_example": f"What is the average {col} by status?" if 'status' in df.columns else f"What is the average {col}?",
                    "reason": f"{'Highly skewed' if abs(skew) > 1 else 'Normally distributed'} — {'outliers likely exist' if abs(skew) > 1 else 'consistent values'}"
                })

        # --- Cross-column suggestions ---
        if categorical_cols and numeric_cols:
            cat = categorical_cols[0]
            num = numeric_cols[0]
            suggestions.append({
                "suggestion": f"Compare '{num}' across '{cat}' groups",
                "query_example": f"Show me average {num} by {cat} as a bar chart",
                "reason": "Combining a measure with a dimension reveals actionable patterns"
            })

        # --- Data quality suggestion ---
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if len(cols_with_nulls) > 0:
            suggestions.append({
                "suggestion": "Data quality check — some columns have missing values",
                "query_example": "Are there any data quality issues in the orders?",
                "reason": f"{len(cols_with_nulls)} column(s) have nulls: {', '.join(cols_with_nulls.index.tolist()[:5])}"
            })

        # Format as readable text
        top_suggestions = suggestions[:8]
        text = "💡 **Smart Analysis Suggestions**\n\n"
        text += f"**Dataset:** {len(df)} rows × {len(df.columns)} columns\n"
        if numeric_cols:
            text += f"**Numeric columns:** {', '.join(numeric_cols)}\n"
        if categorical_cols:
            text += f"**Categorical columns:** {', '.join(categorical_cols)}\n"
        if date_cols:
            text += f"**Date columns:** {', '.join(date_cols)}\n"
        if context:
            text += f"**Context:** {context}\n"
        text += "\n---\n\n"
        for i, s in enumerate(top_suggestions, 1):
            text += f"**{i}. {s['suggestion']}**\n"
            text += f"   → _{s['query_example']}_\n"
            text += f"   ℹ️ {s['reason']}\n\n"
        if not top_suggestions:
            text += "_No specific suggestions — try asking about the data first._\n"
        return text
    except Exception as e:
        return f"❌ Failed to generate suggestions: {e}"


# ---------------------------------------------------------------------------
# Tool 6 — Executive Summary / Business Intelligence Briefing
# ---------------------------------------------------------------------------

def executive_summary() -> str:
    """Generate a comprehensive executive business intelligence summary of
    the orders data in a single call. Covers KPIs, trends, customer
    segments, status breakdown, and actionable insights.

    No parameters required — call this tool when the user asks for a
    summary, briefing, overview, dashboard, or business intelligence report.
    """
    try:
        df = _load_orders()
        text = "📊 **Executive Summary — Orders Intelligence Briefing**\n\n"
        text += "---\n\n"

        # ---- 1. Key Performance Indicators ----
        total_orders = len(df)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        # Try to find the main amount column
        amount_col = None
        for candidate in ['amount', 'total_amount', 'order_amount', 'revenue']:
            if candidate in df.columns:
                amount_col = candidate
                break
        if amount_col is None and numeric_cols:
            amount_col = numeric_cols[0]

        text += "### 🎯 Key Performance Indicators\n"
        text += f"- **Total Orders:** {total_orders:,}\n"

        if amount_col:
            total_rev = df[amount_col].sum()
            avg_val = df[amount_col].mean()
            median_val = df[amount_col].median()
            min_val = df[amount_col].min()
            max_val = df[amount_col].max()
            text += f"- **Total Revenue ({amount_col}):** ${total_rev:,.2f}\n"
            text += f"- **Average Order Value:** ${avg_val:,.2f}\n"
            text += f"- **Median Order Value:** ${median_val:,.2f}\n"
            text += f"- **Order Range:** ${min_val:,.2f} — ${max_val:,.2f}\n"

        # Unique customers
        cust_col = None
        for candidate in ['customer_id', 'cust_id', 'customer']:
            if candidate in df.columns:
                cust_col = candidate
                break
        if cust_col:
            text += f"- **Unique Customers:** {df[cust_col].nunique():,}\n"
            text += f"- **Avg Orders per Customer:** {total_orders / df[cust_col].nunique():.1f}\n"
        text += "\n"

        # ---- 2. Order Status Breakdown ----
        status_col = None
        for candidate in ['status', 'order_status']:
            if candidate in df.columns:
                status_col = candidate
                break
        if status_col:
            status_counts = df[status_col].value_counts()
            text += "### 📋 Order Status Breakdown\n"
            text += "| Status | Count | % of Total |\n"
            text += "| --- | --- | --- |\n"
            for status, count in status_counts.items():
                pct = count / total_orders * 100
                text += f"| {status} | {count} | {pct:.1f}% |\n"
            text += "\n"

        # ---- 3. Revenue by Status ----
        if status_col and amount_col:
            rev_by_status = df.groupby(status_col)[amount_col].agg(['sum', 'mean', 'count'])
            rev_by_status = rev_by_status.sort_values('sum', ascending=False)
            text += "### 💰 Revenue by Status\n"
            text += "| Status | Total Revenue | Avg Order | Orders |\n"
            text += "| --- | --- | --- | --- |\n"
            for status, row in rev_by_status.iterrows():
                text += f"| {status} | ${row['sum']:,.2f} | ${row['mean']:,.2f} | {int(row['count'])} |\n"
            text += "\n"

        # ---- 4. Top 5 Customers ----
        if cust_col and amount_col:
            top_cust = df.groupby(cust_col)[amount_col].agg(['sum', 'count']).sort_values('sum', ascending=False).head(5)
            text += "### 👑 Top 5 Customers by Revenue\n"
            text += "| Customer | Total Spend | Orders |\n"
            text += "| --- | --- | --- |\n"
            for cust, row in top_cust.iterrows():
                text += f"| {cust} | ${row['sum']:,.2f} | {int(row['count'])} |\n"
            text += "\n"

        # ---- 5. Payment Method Mix ----
        pay_col = None
        for candidate in ['payment_method', 'payment_type', 'pay_method']:
            if candidate in df.columns:
                pay_col = candidate
                break
        if pay_col:
            pay_counts = df[pay_col].value_counts()
            text += "### 💳 Payment Method Distribution\n"
            text += "| Method | Orders | % |\n"
            text += "| --- | --- | --- |\n"
            for method, count in pay_counts.items():
                text += f"| {method} | {count} | {count / total_orders * 100:.1f}% |\n"
            text += "\n"

        # ---- 6. Monthly Trend (if date column exists) ----
        date_col = None
        for candidate in ['order_date', 'date', 'created_at', 'created_date']:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col:
            try:
                df['_parsed_date'] = pd.to_datetime(df[date_col], errors='coerce')
                valid = df.dropna(subset=['_parsed_date'])
                if len(valid) > 0:
                    min_d = valid['_parsed_date'].min().strftime('%Y-%m-%d')
                    max_d = valid['_parsed_date'].max().strftime('%Y-%m-%d')
                    text += f"### 📅 Time Span\n"
                    text += f"- **Date Range:** {min_d} → {max_d}\n\n"

                    monthly = valid.set_index('_parsed_date').resample('M').agg(
                        orders=(amount_col, 'count') if amount_col else (date_col, 'count'),
                        revenue=(amount_col, 'sum') if amount_col else (date_col, 'count')
                    ).tail(12)
                    if len(monthly) > 1:
                        text += "**Monthly Trend (last 12 months):**\n"
                        text += "| Month | Orders | Revenue |\n"
                        text += "| --- | --- | --- |\n"
                        for idx, row in monthly.iterrows():
                            text += f"| {idx.strftime('%Y-%m')} | {int(row['orders'])} | ${row['revenue']:,.2f} |\n"
                        text += "\n"
                df.drop(columns=['_parsed_date'], inplace=True, errors='ignore')
            except Exception:
                pass

        # ---- 7. Shipping Breakdown ----
        ship_col = None
        for candidate in ['shipping_method', 'ship_method', 'delivery_method']:
            if candidate in df.columns:
                ship_col = candidate
                break
        if ship_col:
            ship_counts = df[ship_col].value_counts()
            text += "### 🚚 Shipping Method Breakdown\n"
            text += "| Method | Orders | % |\n"
            text += "| --- | --- | --- |\n"
            for method, count in ship_counts.items():
                text += f"| {method} | {count} | {count / total_orders * 100:.1f}% |\n"
            text += "\n"

        # ---- 8. AI Insights & Observations ----
        text += "### 🧠 AI-Generated Insights\n\n"
        insights = []

        if amount_col:
            # Skewness insight
            skew = df[amount_col].skew()
            if abs(skew) > 1:
                insights.append(f"⚠️ **Revenue distribution is highly skewed** (skew={skew:.2f}) — a few large orders drive a disproportionate share of revenue.")
            else:
                insights.append(f"✅ **Revenue distribution is balanced** (skew={skew:.2f}) — order values are fairly consistent.")

            # Concentration insight
            if cust_col:
                top1_rev = df.groupby(cust_col)[amount_col].sum().max()
                top1_pct = top1_rev / df[amount_col].sum() * 100
                if top1_pct > 15:
                    insights.append(f"⚠️ **Customer concentration risk:** Top customer accounts for {top1_pct:.1f}% of total revenue.")

        if status_col:
            cancelled_rate = (df[status_col].str.lower() == 'cancelled').sum() / total_orders * 100
            pending_rate = (df[status_col].str.lower() == 'pending').sum() / total_orders * 100
            if cancelled_rate > 10:
                insights.append(f"🔴 **High cancellation rate:** {cancelled_rate:.1f}% of orders are cancelled — investigate root cause.")
            elif cancelled_rate > 5:
                insights.append(f"🟡 **Notable cancellations:** {cancelled_rate:.1f}% of orders cancelled.")
            if pending_rate > 10:
                insights.append(f"🟡 **Pending backlog:** {pending_rate:.1f}% of orders are still pending.")

        # Data quality insight
        null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        if null_pct > 5:
            insights.append(f"⚠️ **Data quality concern:** {null_pct:.1f}% of all cells are null.")
        elif null_pct == 0:
            insights.append("✅ **Data quality is excellent** — zero missing values detected.")

        if not insights:
            insights.append("✅ No significant issues detected in the dataset.")

        for insight in insights:
            text += f"- {insight}\n"

        text += "\n---\n"
        text += "_Generated by OrdersAnalyticsAgent — ask follow-up questions to drill deeper._\n"

        return text
    except Exception as e:
        return f"❌ Failed to generate executive summary: {e}"


# ---------------------------------------------------------------------------
# Tool 7 — Anomaly Detection (Cross-Source Reconciliation)
# ---------------------------------------------------------------------------

def detect_anomalies(
    join_column: str = 'order_id',
    compare_columns: str = '',
    threshold_pct: float = 5.0
) -> str:
    """Cross-reference fulfilment orders vs website/CT orders to find anomalies.

    Compares two data sources — the fulfilment system (data/orders/) and the
    website/CT system (data/orders_CT/) — to detect mismatches, missing records,
    and value discrepancies.

    Args:
        join_column: The column to match records across both sources
                     (default: 'order_id'). Must exist in both datasets.
        compare_columns: Comma-separated list of columns to compare values
                         (e.g. 'amount,status'). If empty, auto-detects
                         columns common to both sources.
        threshold_pct: Percentage difference threshold for flagging numeric
                       discrepancies (default: 5.0 means flag if >5% off).
    """
    try:
        df_ful = _load_orders()
        df_ct = _load_ct_orders()

        # Validate join column exists in both
        if join_column not in df_ful.columns:
            return f"❌ Join column '{join_column}' not found in fulfilment data. Available: {', '.join(df_ful.columns)}"
        if join_column not in df_ct.columns:
            return f"❌ Join column '{join_column}' not found in website/CT data. Available: {', '.join(df_ct.columns)}"

        anomalies = []

        # ---- 1. Aggregate Overview ----
        overview = {
            "fulfilment_total_rows": len(df_ful),
            "website_ct_total_rows": len(df_ct),
            "fulfilment_unique_keys": int(df_ful[join_column].nunique()),
            "website_ct_unique_keys": int(df_ct[join_column].nunique()),
        }

        # ---- 2. Missing Records ----
        ful_keys = set(df_ful[join_column].dropna().astype(str))
        ct_keys = set(df_ct[join_column].dropna().astype(str))

        only_in_fulfilment = ful_keys - ct_keys
        only_in_website = ct_keys - ful_keys
        common_keys = ful_keys & ct_keys

        missing_records = {
            "orders_only_in_fulfilment": len(only_in_fulfilment),
            "orders_only_in_website_ct": len(only_in_website),
            "orders_in_both": len(common_keys),
        }

        if only_in_fulfilment:
            sample_ful_only = list(only_in_fulfilment)[:10]
            missing_records["sample_fulfilment_only"] = sample_ful_only
            if len(only_in_fulfilment) > 0:
                anomalies.append({
                    "type": "MISSING_FROM_WEBSITE",
                    "severity": "HIGH" if len(only_in_fulfilment) > 10 else "MEDIUM",
                    "count": len(only_in_fulfilment),
                    "description": f"{len(only_in_fulfilment)} order(s) exist in fulfilment but NOT on the website/CT system",
                    "sample_keys": sample_ful_only
                })

        if only_in_website:
            sample_ct_only = list(only_in_website)[:10]
            missing_records["sample_website_ct_only"] = sample_ct_only
            if len(only_in_website) > 0:
                anomalies.append({
                    "type": "MISSING_FROM_FULFILMENT",
                    "severity": "HIGH" if len(only_in_website) > 10 else "MEDIUM",
                    "count": len(only_in_website),
                    "description": f"{len(only_in_website)} order(s) exist on the website but NOT in the fulfilment system",
                    "sample_keys": sample_ct_only
                })

        # ---- 3. Value Discrepancies (on matched records) ----
        value_discrepancies = []
        if common_keys:
            # Determine columns to compare
            if compare_columns:
                cols_to_compare = [c.strip() for c in compare_columns.split(',')]
            else:
                # Auto-detect: columns present in both (excluding the join column)
                common_cols = set(df_ful.columns) & set(df_ct.columns) - {join_column}
                cols_to_compare = list(common_cols)

            if cols_to_compare:
                # Merge on join column for matched records
                df_ful_str = df_ful.copy()
                df_ct_str = df_ct.copy()
                df_ful_str[join_column] = df_ful_str[join_column].astype(str)
                df_ct_str[join_column] = df_ct_str[join_column].astype(str)

                merged = df_ful_str.merge(
                    df_ct_str,
                    on=join_column,
                    how='inner',
                    suffixes=('_fulfilment', '_website')
                )

                for col in cols_to_compare:
                    col_ful = f"{col}_fulfilment" if f"{col}_fulfilment" in merged.columns else col
                    col_ct = f"{col}_website" if f"{col}_website" in merged.columns else col

                    if col_ful not in merged.columns or col_ct not in merged.columns:
                        continue

                    # Numeric comparison
                    if pd.api.types.is_numeric_dtype(merged[col_ful]) and pd.api.types.is_numeric_dtype(merged[col_ct]):
                        merged['_diff'] = (merged[col_ful] - merged[col_ct]).abs()
                        merged['_pct_diff'] = merged.apply(
                            lambda r: (r['_diff'] / r[col_ful] * 100) if r[col_ful] != 0 else (100.0 if r['_diff'] != 0 else 0.0),
                            axis=1
                        )
                        mismatches = merged[merged['_pct_diff'] > threshold_pct]

                        if len(mismatches) > 0:
                            sample_rows = mismatches.head(5)[[join_column, col_ful, col_ct, '_pct_diff']].to_dict('records')
                            anomalies.append({
                                "type": "VALUE_DISCREPANCY",
                                "column": col,
                                "severity": "HIGH" if len(mismatches) > len(merged) * 0.1 else "MEDIUM",
                                "count": len(mismatches),
                                "total_matched": len(merged),
                                "description": f"{len(mismatches)}/{len(merged)} matched orders have '{col}' differing by >{threshold_pct}%",
                                "sample": sample_rows
                            })

                        # Aggregate drift for this numeric column
                        sum_ful = float(merged[col_ful].sum())
                        sum_ct = float(merged[col_ct].sum())
                        if sum_ful != 0:
                            agg_drift_pct = abs(sum_ful - sum_ct) / abs(sum_ful) * 100
                        else:
                            agg_drift_pct = 0.0 if sum_ct == 0 else 100.0

                        value_discrepancies.append({
                            "column": col,
                            "fulfilment_total": round(sum_ful, 2),
                            "website_ct_total": round(sum_ct, 2),
                            "difference": round(sum_ful - sum_ct, 2),
                            "drift_pct": round(agg_drift_pct, 2)
                        })

                        if agg_drift_pct > threshold_pct:
                            anomalies.append({
                                "type": "AGGREGATE_DRIFT",
                                "column": col,
                                "severity": "CRITICAL" if agg_drift_pct > 20 else "HIGH",
                                "description": f"Total '{col}' differs by {agg_drift_pct:.1f}% between systems (fulfilment: {sum_ful:,.2f}, website: {sum_ct:,.2f})",
                            })

                    else:
                        # Categorical comparison
                        mismatches = merged[merged[col_ful].astype(str).str.lower() != merged[col_ct].astype(str).str.lower()]
                        if len(mismatches) > 0:
                            sample_rows = mismatches.head(5)[[join_column, col_ful, col_ct]].to_dict('records')
                            anomalies.append({
                                "type": "VALUE_MISMATCH",
                                "column": col,
                                "severity": "MEDIUM",
                                "count": len(mismatches),
                                "total_matched": len(merged),
                                "description": f"{len(mismatches)}/{len(merged)} matched orders have different '{col}' values",
                                "sample": sample_rows
                            })

        # ---- 4. Duplicate Detection ----
        ful_dupes = int(df_ful[join_column].duplicated().sum())
        ct_dupes = int(df_ct[join_column].duplicated().sum())
        if ful_dupes > 0:
            anomalies.append({
                "type": "DUPLICATES",
                "source": "fulfilment",
                "severity": "MEDIUM",
                "count": ful_dupes,
                "description": f"{ful_dupes} duplicate '{join_column}' values in fulfilment data"
            })
        if ct_dupes > 0:
            anomalies.append({
                "type": "DUPLICATES",
                "source": "website_ct",
                "severity": "MEDIUM",
                "count": ct_dupes,
                "description": f"{ct_dupes} duplicate '{join_column}' values in website/CT data"
            })

        # ---- Sort anomalies by severity ----
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        anomalies.sort(key=lambda a: severity_order.get(a.get("severity", "LOW"), 3))

        health = ("✅ HEALTHY" if len(anomalies) == 0 else
                  "🔴 CRITICAL" if any(a["severity"] == "CRITICAL" for a in anomalies) else
                  "🟠 ISSUES FOUND" if any(a["severity"] == "HIGH" for a in anomalies) else
                  "🟡 MINOR ISSUES")

        # ---- Format as readable report ----
        text = "🔎 **Cross-Source Anomaly Report**\n\n"
        text += f"**Health Status:** {health}\n"
        text += f"**Sources:** Fulfilment (`{ORDERS_FOLDER}`) vs Website/CT (`{CT_ORDERS_FOLDER}`)\n\n"
        text += "---\n\n"

        text += "### Overview\n"
        text += "| Metric | Fulfilment | Website/CT |\n"
        text += "| --- | --- | --- |\n"
        text += f"| Total rows | {overview['fulfilment_total_rows']} | {overview['website_ct_total_rows']} |\n"
        text += f"| Unique keys | {overview['fulfilment_unique_keys']} | {overview['website_ct_unique_keys']} |\n\n"

        text += "### Missing Records\n"
        text += f"- Orders in **both** systems: **{missing_records['orders_in_both']}**\n"
        if missing_records['orders_only_in_fulfilment'] > 0:
            samples = ', '.join(missing_records.get('sample_fulfilment_only', [])[:5])
            text += f"- Only in fulfilment: **{missing_records['orders_only_in_fulfilment']}** ({samples})\n"
        if missing_records['orders_only_in_website_ct'] > 0:
            samples = ', '.join(missing_records.get('sample_website_ct_only', [])[:5])
            text += f"- Only in website/CT: **{missing_records['orders_only_in_website_ct']}** ({samples})\n"
        text += "\n"

        if value_discrepancies:
            text += "### Aggregate Comparison\n"
            text += "| Column | Fulfilment Total | Website Total | Difference | Drift % |\n"
            text += "| --- | --- | --- | --- | --- |\n"
            for vd in value_discrepancies:
                text += f"| {vd['column']} | {vd['fulfilment_total']:,.2f} | {vd['website_ct_total']:,.2f} | {vd['difference']:,.2f} | {vd['drift_pct']:.1f}% |\n"
            text += "\n"

        if anomalies:
            sev_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "⚪"}
            text += f"### Anomalies Found ({len(anomalies)})\n\n"
            for a in anomalies:
                icon = sev_icons.get(a['severity'], "⚪")
                text += f"{icon} **{a['severity']} — {a['type']}**"
                if 'column' in a:
                    text += f" (`{a['column']}`)"
                text += f"\n{a['description']}\n"
                if 'sample' in a:
                    for s in a['sample'][:3]:
                        text += f"   • {s}\n"
                if 'sample_keys' in a:
                    text += f"   Keys: {', '.join(str(k) for k in a['sample_keys'][:5])}\n"
                text += "\n"
        else:
            text += "### No anomalies detected ✅\n"

        return text
    except Exception as e:
        return f"❌ Failed to detect anomalies: {e}"


# ---------------------------------------------------------------------------
# Root Agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    model="gemini-2.0-flash",
    name="OrdersAnalyticsAgent",
    description="Query, filter, aggregate, and plot data from the orders table in GCS.",
    instruction="""
        You are an orders data analyst. You have full access to the orders table
        stored in Google Cloud Storage and can query, filter, aggregate, and
        visualize it.

        TOOLS AVAILABLE:
        1. sample_orders(rows)
           Use for: "show me some orders", "what does the data look like?",
                    "what columns are available?"

        2. filter_orders(column, value, rows)
           Use for: "show orders with status = 'shipped'", "orders for customer X"

        3. aggregate_orders(group_by, metric_column, aggregation)
           aggregation options: count, sum, avg, min, max
           group_by is OPTIONAL — pass '' or omit it for overall totals.
           Use for: "how many orders by status?", "total revenue by customer",
                    "average order amount by region"
           Also for: "what is the total revenue?", "what is the average order amount?"
                     (no group_by needed — just set metric_column and aggregation)

        4. plot_orders(group_by, metric_column, aggregation, chart_type, top_n)
           chart_type options: bar, pie, line
           Use for: "plot orders by status", "pie chart of revenue by region"
           This tool saves the chart as an artifact that will be displayed in the ADK web UI.
           After calling the tool, simply confirm to the user that the chart was generated successfully.

        5. suggest_analysis(context)
           Use for: proactively suggesting follow-up analyses to the user.
           Pass a brief summary of what the user just explored as 'context'.
           The tool inspects column types, distributions, dates, and data quality
           to return smart, contextual suggestions.

        6. executive_summary()
           Generates a full business intelligence briefing in one call.
           Covers: KPIs, status breakdown, revenue analysis, top customers,
           payment methods, monthly trends, shipping mix, and AI-generated
           insights with reasoning.
           Use for: "give me a summary", "executive briefing", "overview",
                    "what's the state of the business?", "dashboard",
                    "business intelligence report"
           No parameters needed — just call it.

        7. detect_anomalies(join_column, compare_columns, threshold_pct)
           Cross-references TWO order sources to find discrepancies:
           - Source A: Fulfilment system (data/orders/)
           - Source B: Website/CT system (data/orders_CT/)
           Detects:
           • Missing records — orders in one system but not the other
           • Value discrepancies — same order, different amounts or statuses
           • Aggregate drift — total revenue/count doesn't match between systems
           • Duplicates — repeated order IDs within a source
           Use for: "are there any anomalies?", "reconcile orders",
                    "compare fulfilment vs website", "find mismatches",
                    "data reconciliation", "validate order data"
           Returns a health status: ✅ HEALTHY, 🟡 MINOR, 🟠 ISSUES, 🔴 CRITICAL

        WORKFLOW:
        - If the user seems unsure about column names, call sample_orders(rows=1) first.
        - Summary / overview / briefing → executive_summary
        - Distribution questions   → aggregate_orders with count
        - Revenue/amount questions → aggregate_orders with sum or avg on the amount column
        - Any "plot/chart/graph"   → plot_orders
        - Anomaly / reconciliation / mismatch questions → detect_anomalies

        PROACTIVE SUGGESTIONS:
        After answering ANY user question, always end your response with
        2-3 suggested follow-up questions under a "💡 You might also want to explore:" heading.
        Make these specific and relevant to what the user just asked.
        If the user says "hello", "hi", or asks a general question like "what can you do?",
        call suggest_analysis() to provide data-driven starting points.

        RESPONSE FORMAT:
        Tool outputs are pre-formatted as readable Markdown with tables,
        headers, and emoji icons. Present the tool output directly in your
        response — do NOT reformat it as JSON or omit the content.
        Add a brief plain-English interpretation after the tool output.
    """,
    tools=[sample_orders, filter_orders, aggregate_orders, plot_orders, suggest_analysis, executive_summary, detect_anomalies, load_artifacts],
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)

# Export for ADK
__all__ = ['root_agent']


if __name__ == "__main__":
    print("✅ OrdersAnalyticsAgent initialized")
    print(f"   Model : {root_agent.model}")
    print(f"   Tools : {[t.__name__ for t in root_agent.tools]}")
    project_id = get_project_id()
    print(f"   PROJECT_ID : {project_id or '⚠️  Not set'}")