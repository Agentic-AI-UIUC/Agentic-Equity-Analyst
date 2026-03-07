---
name: add-equity-tool
description: Scaffold a new LangChain tool for the reporting agent. Use when the user wants to add a new tool, capability, or data source to the equity analyst.
---

# Add Equity Tool

Scaffold a new LangChain tool for the reporting agent.

## Steps

1. Ask the user for:
   - Tool name (e.g., `analyze_macro`)
   - What the tool should do (brief description)
   - Which file it belongs in (new file or existing, e.g., `analyst.py`)

2. Create or edit the target file and add the tool using the `@tool` decorator:

```python
from langchain.tools import tool

@tool
def your_tool_name(query: str) -> str:
    """
    One-line description of what this tool does.
    The reporting agent uses this description to decide when to call the tool.
    """
    # Implementation here
    return result
```

**Important:** LangChain tools must only accept simple Python types (`str`, `int`, `float`, `bool`) — no complex objects.

3. Register the tool in `reporting_pipeline.py` by adding it to the `reporting_tools` list:

```python
from your_module import your_tool_name

reporting_tools = [
    analyze_filings,
    analyze_parser,
    analyze_financials,
    analyze_news,
    find_dcf_tool,
    calculate_moving_average_tool,
    valuation_tool,
    your_tool_name,   # <- add here
]
```

4. If the tool uses a new ChromaDB collection, document it in `CLAUDE.md` under the Chroma Collections table.

5. Test by running a full analysis: `/run-equity-analysis`

## Notes
- Chroma connections must use Cloud config: pass `database`, `chroma_cloud_api_key`, `tenant`.
- Embedding model is `text-embedding-3-small` (hardcoded as `EMBEDDING_MODEL`).
- The `reporting_agent` is re-initialized automatically on next import after adding the tool.
