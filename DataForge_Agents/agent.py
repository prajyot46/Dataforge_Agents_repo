from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.cloud import storage
import json, pandas as pd
import os
from io import StringIO


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_project_id():
    """Get project ID from environment variable or gcloud config."""
    project_id = os.environ.get('PROJECT_ID')
    if not project_id:
        try:
            import subprocess
            result = subprocess.run(
                ['gcloud', 'config', 'get-value', 'project'],
                capture_output=True, text=True, timeout=5
            )
            project_id = result.stdout.strip()
        except:
            pass
    return project_id


def _normalize_table_name(table_name: str) -> str:
    """Normalize table name to a consistent format."""
    name = table_name.lower().strip().replace(' table', '').replace('the ', '').strip()
    aliases = {'order': 'orders', 'customer': 'customers', 'product': 'products'}
    return aliases.get(name, name)


# ---------------------------------------------------------------------------
# Agent 1 — DataDiscoveryAgent
# Responsible for: listing tables, reading metadata, parsing schemas
# ---------------------------------------------------------------------------

def list_available_tables() -> dict:
    """List all available tables. CALL THIS FIRST to see what tables exist."""
    project_id = get_project_id()
    if not project_id:
        return {"error": "PROJECT_ID not set. Set environment variable: export PROJECT_ID='your-project-id'"}

    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket('hackathon-agent-data')
        blobs = bucket.list_blobs(prefix='metadata/tables/')

        tables = []
        for blob in blobs:
            if blob.name.endswith('_schema.json'):
                table_name = blob.name.split('/')[-1].replace('_schema.json', '')
                tables.append(table_name)

        return {
            "available_tables": tables,
            "count": len(tables),
            "instructions": "Use these EXACT names (e.g., 'orders', not 'order table') when calling read_metadata()"
        }
    except Exception as e:
        return {"error": f"Failed to list tables: {str(e)}"}


def read_metadata(table_name: str) -> dict:
    """Read table metadata schema from GCS bucket.

    Args:
        table_name: EXACT table name (e.g., 'orders', 'customers', 'products')
    """
    project_id = get_project_id()
    if not project_id:
        return {"error": "PROJECT_ID not set"}

    original_name = table_name
    table_name = _normalize_table_name(table_name)

    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket('hackathon-agent-data')
        blob = bucket.blob(f'metadata/tables/{table_name}_schema.json')

        if not blob.exists():
            blobs = list(bucket.list_blobs(prefix='metadata/tables/'))
            available = [b.name.split('/')[-1].replace('_schema.json', '')
                         for b in blobs if '_schema.json' in b.name]
            return {
                "error": f"Table '{original_name}' (normalized to '{table_name}') not found",
                "available_tables": available
            }

        return json.loads(blob.download_as_text())
    except Exception as e:
        return {"error": f"Failed to read metadata: {str(e)}"}


def parse_schema(schema_json: dict) -> dict:
    """Parse and analyze table schema metadata.

    Args:
        schema_json: Schema dictionary returned from read_metadata()
    """
    if "error" in schema_json:
        return schema_json

    columns = schema_json.get('columns', [])
    return {
        'table_name': schema_json.get('table_name', 'unknown'),
        'description': schema_json.get('description', ''),
        'column_count': len(columns),
        'column_names': [col['name'] for col in columns],
        'nullable_columns': [col['name'] for col in columns if col.get('nullable', True)],
        'primary_keys': [col['name'] for col in columns if col.get('nullable') == False],
        'column_types': {col['name']: col['type'] for col in columns},
        'foreign_keys': schema_json.get('foreign_keys', [])
    }


discovery_agent = Agent(
    model="gemini-2.0-flash",
    name="DataDiscoveryAgent",
    description="Lists available tables and reads/parses their schemas from GCS.",
    instruction="""
        You are a data discovery specialist. You help users understand what data is
        available and its structure.

        WORKFLOW:
        1. "What tables exist?" → Call list_available_tables()
        2. "Show me the schema for X" → Call read_metadata(table_name=X)
        3. "Parse/analyze the schema for X" → Call read_metadata() then parse_schema()

        Always use EXACT table names: "orders", "customers", "products".
        Explain results in clear, natural language.
    """,
    tools=[list_available_tables, read_metadata, parse_schema],
)


# ---------------------------------------------------------------------------
# Agent 2 — DataSamplingAgent
# Responsible for: reading raw CSV data samples from GCS
# ---------------------------------------------------------------------------

def read_data_sample(table_name: str, rows: int = 10) -> dict:
    """Sample rows from actual data file in GCS.

    Args:
        table_name: Table name (e.g., 'orders', 'customers')
        rows: Number of rows to sample (default 10, max 100)
    """
    project_id = get_project_id()
    if not project_id:
        return {"error": "PROJECT_ID not set", "fix": "Run: export PROJECT_ID='your-project-id'"}

    table_name = _normalize_table_name(table_name)
    rows = min(rows, 100)

    bucket_name = 'hackathon-agent-data'
    data_path = f'data/{table_name}/'

    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=data_path))

        if not blobs:
            return {
                "error": f"No data folder found for table '{table_name}'",
                "searched_path": f"gs://{bucket_name}/{data_path}",
                "suggestion": "Check that data has been uploaded to GCS"
            }

        csv_blob = next((b for b in blobs if b.name.endswith('.csv')), None)
        if not csv_blob:
            return {
                "error": f"No CSV files found in {data_path}",
                "files_found": [b.name for b in blobs]
            }

        # METHOD 1: Download blob and read from memory
        try:
            csv_content = csv_blob.download_as_text()
            df = pd.read_csv(StringIO(csv_content), nrows=rows)
            return {
                "table": table_name,
                "file_read": f"gs://{bucket_name}/{csv_blob.name}",
                "method": "storage_client",
                "columns": list(df.columns),
                "sample": df.to_dict('records'),
                "rows_returned": len(df),
                "total_columns": len(df.columns)
            }
        except Exception as e1:
            # METHOD 2: Direct pandas read
            try:
                full_path = f'gs://{bucket_name}/{csv_blob.name}'
                df = pd.read_csv(full_path, nrows=rows)
                return {
                    "table": table_name,
                    "file_read": full_path,
                    "method": "pandas_direct",
                    "columns": list(df.columns),
                    "sample": df.to_dict('records'),
                    "rows_returned": len(df),
                    "total_columns": len(df.columns)
                }
            except Exception as e2:
                return {
                    "error": "Failed to read CSV",
                    "file": f"gs://{bucket_name}/{csv_blob.name}",
                    "storage_client_error": str(e1),
                    "pandas_direct_error": str(e2),
                    "suggestion": "Check file format and permissions"
                }

    except Exception as e:
        return {"error": f"Failed to access GCS: {str(e)}", "bucket": bucket_name, "path": data_path}


sampling_agent = Agent(
    model="gemini-2.0-flash",
    name="DataSamplingAgent",
    description="Reads and returns raw data samples from GCS CSV files.",
    instruction="""
        You are a data sampling specialist. You fetch and display raw data rows
        from GCS tables for inspection.

        WORKFLOW:
        - "Show me N rows from X" → Call read_data_sample(table_name=X, rows=N)

        Always print the returned sample clearly so users can inspect the data.
        Use EXACT table names: "orders", "customers", "products".
    """,
    tools=[read_data_sample],
)


# ---------------------------------------------------------------------------
# Agent 3 — DataAnalyticsAgent
# Responsible for: aggregations, counts, cross-table joins
# ---------------------------------------------------------------------------

def count_by_column(table_name: str, column_name: str) -> dict:
    """Count occurrences grouped by a column value.

    Args:
        table_name: Table to analyze (e.g., 'orders', 'customers')
        column_name: Column to group by (e.g., 'status', 'customer_segment')
    """
    project_id = get_project_id()
    if not project_id:
        return {"error": "PROJECT_ID not set"}

    import gcsfs
    table_name = _normalize_table_name(table_name)

    try:
        fs = gcsfs.GCSFileSystem(project=project_id)
        path = f'hackathon-agent-data/data/{table_name}/'
        files = fs.ls(path)

        if not files:
            return {"error": f"No data files found for {table_name}"}

        df = pd.read_csv(f'gs://{files[0]}')

        if column_name not in df.columns:
            return {
                "error": f"Column '{column_name}' not found in {table_name}",
                "available_columns": list(df.columns)
            }

        counts = df[column_name].value_counts().to_dict()
        return {
            "table": table_name,
            "column": column_name,
            "counts": counts,
            "total_unique_values": len(counts),
            "total_rows": len(df)
        }
    except Exception as e:
        return {"error": f"Failed to count: {str(e)}"}


def join_tables_by_customer(limit: int = 10) -> dict:
    """Join orders and customers tables to show which customers have the most orders.

    Args:
        limit: Number of top customers to return (default 10)
    """
    project_id = get_project_id()
    if not project_id:
        return {"error": "PROJECT_ID not set"}

    import gcsfs

    try:
        fs = gcsfs.GCSFileSystem(project=project_id)
        orders_files = fs.ls('hackathon-agent-data/data/orders/')
        customers_files = fs.ls('hackathon-agent-data/data/customers/')

        if not orders_files or not customers_files:
            return {"error": "Missing data files for orders or customers"}

        orders_df = pd.read_csv(f'gs://{orders_files[0]}')
        customers_df = pd.read_csv(f'gs://{customers_files[0]}')

        order_counts = orders_df.groupby('customer_id').size().reset_index(name='order_count')

        result = order_counts.merge(
            customers_df[['customer_id', 'first_name', 'last_name', 'customer_segment', 'status']],
            on='customer_id',
            how='left'
        )
        result = result.sort_values('order_count', ascending=False).head(limit)

        return {
            "top_customers": result.to_dict('records'),
            "total_customers_analyzed": len(order_counts),
            "total_orders": len(orders_df)
        }
    except Exception as e:
        return {"error": f"Failed to join tables: {str(e)}"}


analytics_agent = Agent(
    model="gemini-2.0-flash",
    name="DataAnalyticsAgent",
    description="Runs aggregations, counts, and cross-table joins on GCS data.",
    instruction="""
        You are a data analytics specialist. You answer quantitative questions
        about the data using aggregation and joins.

        WORKFLOW:
        - "How many orders by status?" → Call count_by_column(table_name="orders", column_name="status")
        - "Which customers have the most orders?" → Call join_tables_by_customer(limit=10)

        Always explain results with clear summaries and highlight key insights.
    """,
    tools=[count_by_column, join_tables_by_customer],
)


# ---------------------------------------------------------------------------
# Root Orchestrator — routes to the three sub-agents
# ---------------------------------------------------------------------------

root_agent = Agent(
    model="gemini-2.0-flash",
    name="DataOrchestratorAgent",
    description="Orchestrates data discovery, sampling, and analytics across GCS buckets.",
    instruction="""
        You are a data intelligence orchestrator. Route user requests to the
        correct specialist sub-agent by calling them as tools:

        - DataDiscoveryAgent  -> questions about what tables exist, schemas, metadata
        - DataSamplingAgent   -> requests to view/preview raw data rows
        - DataAnalyticsAgent  -> counting, grouping, aggregations, joins

        ROUTING EXAMPLES:
        "What tables are available?"              -> call DataDiscoveryAgent
        "Show me the orders schema"               -> call DataDiscoveryAgent
        "Sample 5 rows from customers"            -> call DataSamplingAgent
        "How many orders by status?"              -> call DataAnalyticsAgent
        "Which customers placed the most orders?" -> call DataAnalyticsAgent

        Always explain what the sub-agent returned in natural language.
        If a request spans multiple agents (e.g., "show schema then sample data"),
        call them in sequence and combine the results.
    """,
    tools=[
        AgentTool(agent=discovery_agent),
        AgentTool(agent=sampling_agent),
        AgentTool(agent=analytics_agent),
    ],
)


# Export for ADK
__all__ = ['root_agent']


if __name__ == "__main__":
    print("✅ Multi-agent system initialized")
    print(f"   Orchestrator : {root_agent.name}")
    print(f"   Sub-agents   : DataDiscoveryAgent, DataSamplingAgent, DataAnalyticsAgent")

    project_id = get_project_id()
    if project_id:
        print(f"   PROJECT_ID   : {project_id}")
    else:
        print("   ⚠️  PROJECT_ID not set — export PROJECT_ID='your-project-id'")