
# Unity Catalog Bedrock Integration

Integration package for using Unity Catalog functions with AWS Bedrock.

## Installation

```bash
pip install -e .
```

## Usage

```python
from unitycatalog.ai.bedrock import UCFunctionToolkit

# Create toolkit with function names
toolkit = UCFunctionToolkit(function_names=["your_function_name"])

# Create a session with your Bedrock agent
session = toolkit.create_session(agent_id="your_agent_id", 
                               agent_alias_id="your_alias_id")

# Invoke the agent
response = session.invoke_agent("your input text")
```
