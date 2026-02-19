"""
ADK Agent Configuration for DataForge Multi-Agent System
This file must be named 'agent.py' and contain 'root_agent' variable
"""

from google.adk.agents import Agent
from google.cloud import storage
import pandas as pd
import json


# ============================================================================
# TOOL FUNCTIONS
# ============================================================================

def list_available_tables(project_id: str) -> dict:
    """List all available tables in the metadata folder."""
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
        "message": f"Found {len(tables)} tables. Use exact names when calling other tools."
    }


def read_metadata(table_name: str, project_id: str) -> dict:
    """Read table schema from GCS."""
    original_name = table_name
    table_name = table_name.lower().strip().replace(' table', '').replace('the ', '').strip()
    aliases = {'order': 'orders', 'customer': 'customers'}
    table_name = aliases.get(table_name, table_name)
    
    client = storage.Client(project=project_id)
    bucket = client.bucket('hackathon-agent-data')
    blob = bucket.blob(f'metadata/tables/{table_name}_schema.json')
    
    if not blob.exists():
        blobs = list(bucket.list_blobs(prefix='metadata/tables/'))
        available = [b.name.split('/')[-1].replace('_schema.json', '') 
                    for b in blobs if '_schema.json' in b.name]
        return {"error": f"Table '{original_name}' not found", "available_tables": available}
    
    return json.loads(blob.download_as_text())


def parse_schema(schema_json: dict) -> dict:
    """Parse schema structure."""
    if "error" in schema_json:
        return schema_json
    
    columns = schema_json.get('columns', [])
    return {
        'table_name': schema_json.get('table_name', 'unknown'),
        'description': schema_json.get('description', ''),
        'column_count': len(columns),
        'column_names': [col['name'] for col in columns],
        'column_types': {col['name']: col['type'] for col in columns},
        'primary_key': schema_json.get('primary_key', None),
        'foreign_keys': schema_json.get('foreign_keys', [])
    }


def get_table_relationships(project_id: str) -> dict:
    """Read table relationships from ERD config."""
    client = storage.Client(project=project_id)
    bucket = client.bucket('hackathon-agent-data')
    
    try:
        blob = bucket.blob('metadata/relationships/erd_config.json')
        return json.loads(blob.download_as_text())
    except Exception as e:
        return {"error": str(e), "relationships": []}


def read_full_table(table_name: str, project_id: str) -> pd.DataFrame:
    """Internal helper to read complete table."""
    import gcsfs
    
    table_name = table_name.lower().strip().replace(' table', '')
    aliases = {'order': 'orders', 'customer': 'customers'}
    table_name = aliases.get(table_name, table_name)
    
    fs = gcsfs.GCSFileSystem(project=project_id)
    path = f'hackathon-agent-data/data/{table_name}/'
    files = fs.ls(path)
    
    if not files:
        raise FileNotFoundError(f"No data files for {table_name}")
    
    return pd.read_csv(f'gs://{files[0]}')


def sample_table_data(table_name: str, project_id: str, rows: int = 10) -> dict:
    """Sample rows from table."""
    try:
        df = read_full_table(table_name, project_id)
        return {
            "table": table_name,
            "total_rows": len(df),
            "columns": list(df.columns),
            "sample": df.head(rows).to_dict('records'),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
    except Exception as e:
        return {"error": str(e)}


def aggregate_data(
    table_name: str,
    project_id: str,
    group_by: str,
    agg_column: str,
    agg_func: str = 'count'
) -> dict:
    """Aggregate data with GROUP BY."""
    try:
        df = read_full_table(table_name, project_id)
        
        if group_by not in df.columns:
            return {"error": f"Column '{group_by}' not found in {table_name}"}
        if agg_column not in df.columns:
            return {"error": f"Column '{agg_column}' not found in {table_name}"}
        
        func_map = {'count': 'count', 'sum': 'sum', 'mean': 'mean', 
                   'min': 'min', 'max': 'max', 'average': 'mean', 'avg': 'mean'}
        func = func_map.get(agg_func.lower(), 'count')
        
        result = df.groupby(group_by)[agg_column].agg(func)
        result_sorted = result.sort_values(ascending=False)
        
        return {
            'table': table_name,
            'aggregation': f'{func}({agg_column}) GROUP BY {group_by}',
            'total_groups': len(result),
            'top_10': result_sorted.head(10).to_dict(),
            'all_results': result_sorted.to_dict()
        }
    except Exception as e:
        return {"error": str(e)}


def join_and_analyze(
    left_table: str,
    right_table: str,
    join_column: str,
    project_id: str,
    group_by: str = None,
    agg_column: str = None,
    agg_func: str = 'count'
) -> dict:
    """Join tables and optionally aggregate."""
    try:
        left_df = read_full_table(left_table, project_id)
        right_df = read_full_table(right_table, project_id)
        
        if join_column not in left_df.columns or join_column not in right_df.columns:
            return {"error": f"Join column '{join_column}' not found in both tables"}
        
        joined = left_df.merge(right_df, on=join_column, how='left', suffixes=('', '_right'))
        
        result = {
            'join': f'{left_table} ⋈ {right_table} ON {join_column}',
            'total_rows': len(joined),
            'columns': list(joined.columns)
        }
        
        if group_by and agg_column:
            if group_by not in joined.columns:
                return {"error": f"Column '{group_by}' not found after join"}
            if agg_column not in joined.columns:
                return {"error": f"Column '{agg_column}' not found after join"}
            
            func_map = {'count': 'count', 'sum': 'sum', 'mean': 'mean', 
                       'min': 'min', 'max': 'max', 'average': 'mean', 'avg': 'mean'}
            func = func_map.get(agg_func.lower(), 'count')
            
            agg_result = joined.groupby(group_by)[agg_column].agg(func)
            agg_sorted = agg_result.sort_values(ascending=False)
            
            # Get customer names if available
            results_list = []
            for key, value in agg_sorted.head(10).items():
                row_data = {'group_value': key, 'aggregated_value': value}
                
                # Try to enrich with names if customer_id
                if group_by == 'customer_id' and 'first_name' in joined.columns:
                    customer_row = joined[joined[group_by] == key].iloc[0]
                    row_data['first_name'] = customer_row.get('first_name', '')
                    row_data['last_name'] = customer_row.get('last_name', '')
                
                results_list.append(row_data)
            
            result['aggregation'] = {
                'operation': f'{func}({agg_column}) GROUP BY {group_by}',
                'top_10': results_list,
                'all_results': agg_sorted.to_dict()
            }
        else:
            result['sample'] = joined.head(10).to_dict('records')
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# SUB-AGENTS
# ============================================================================

schema_agent = Agent(
    model="gemini-2.0-flash-exp",
    name="SchemaAgent",
    description="Schema discovery and metadata specialist",
    instruction="""
        You discover and analyze table schemas.
        
        Tools:
        - list_available_tables(): See all tables
        - read_metadata(table_name): Get schema for a table
        - parse_schema(schema_json): Analyze schema structure
        - get_table_relationships(): Get FK relationships
        
        Always provide clear schema information.
    """,
    tools=[list_available_tables, read_metadata, parse_schema, get_table_relationships]
)


data_agent = Agent(
    model="gemini-2.0-flash-exp",
    name="DataAgent",
    description="Data sampling and preview specialist",
    instruction="""
        You sample and preview data.
        
        Tools:
        - sample_table_data(table_name, project_id, rows): Get data sample
        
        Show data concisely with column names and types.
    """,
    tools=[sample_table_data]
)


analytics_agent = Agent(
    model="gemini-2.0-flash-exp",
    name="AnalyticsAgent",
    description="Single-table analytics specialist",
    instruction="""
        You perform aggregations on single tables.
        
        Tools:
        - aggregate_data(table_name, project_id, group_by, agg_column, agg_func)
        
        Example: "Count orders by status"
        → aggregate_data("orders", project_id, "status", "order_id", "count")
        
        Always sort results descending (top first).
    """,
    tools=[aggregate_data]
)


join_agent = Agent(
    model="gemini-2.0-flash-exp",
    name="JoinAgent",
    description="Multi-table join specialist",
    instruction="""
        You join tables and perform cross-table analytics.
        
        Tools:
        - join_and_analyze(left_table, right_table, join_column, project_id, 
                          group_by=None, agg_column=None, agg_func="count")
        
        Example: "Which customers have most orders?"
        → join_and_analyze("orders", "customers", "customer_id", project_id,
                          group_by="customer_id", agg_column="order_id", agg_func="count")
        
        Always include customer names in results when available.
    """,
    tools=[join_and_analyze]
)


# ============================================================================
# ROOT AGENT (REQUIRED BY ADK)
# ============================================================================

root_agent = Agent(
    model="gemini-2.0-flash-exp",
    name="OrchestratorAgent",
    description="Multi-agent orchestrator for data analytics",
    instruction="""
        You orchestrate specialized sub-agents to answer data queries.
        
        SUB-AGENTS:
        1. SchemaAgent - Schema/metadata questions
        2. DataAgent - Data sampling/preview
        3. AnalyticsAgent - Single-table aggregations
        4. JoinAgent - Multi-table joins/analytics
        
        ROUTING:
        - "What tables exist?" → SchemaAgent
        - "Show schema" → SchemaAgent
        - "Sample data" → DataAgent
        - "Count/Sum/Avg by X" (single table) → AnalyticsAgent
        - "Which customers..." (multi-table) → JoinAgent
        
        You can use multiple agents for complex queries.
        Always synthesize results clearly and cite which agent(s) provided data.
    """,
    sub_agents=[schema_agent, data_agent, analytics_agent, join_agent]
)


# Export for ADK CLI
__all__ = ['root_agent']


if __name__ == "__main__":
    print("✅ DataForge Multi-Agent System")
    print(f"   Root Agent: {root_agent.name}")
    print(f"   Sub-Agents: {len(root_agent.sub_agents)}")
    for agent in root_agent.sub_agents:
        print(f"     • {agent.name}")