import time
import pytest
from threading import Thread
from typing import Any, Dict, List

# Import the functions and classes under test (adjust the import paths as needed)
from unitycatalog.ai.bedrock.toolkit import (
    execute_tool_calls,
    extract_tool_calls,
    generate_tool_call_session_state,
    BedrockToolResponse
)

# Create a fake Bedrock client that simulates get_function and execute_function.
class FakeBedrockClient:
    def get_function(self, full_function_name: str) -> Dict[str, Any]:
        # Simulated response with function info.
        return {"full_name": full_function_name, "routine_body": "EXTERNAL"}

    class FakeExecutionResult:
        def __init__(self, value):
            self.value = value

    def execute_function(self, full_function_name: str, params: Dict[str, Any]) -> Any:
        # Always return "23" as the result.
        return FakeBedrockClient.FakeExecutionResult("23")

# Sample tool call data as produced by extract_tool_calls.
tool_calls_sample: List[Dict[str, Any]] = [{
    'function_name': 'tbd-bda-action-group-name__location_weather_in_c',
    'parameters': {
        'fetch_date': '2024-11-19',
        'location_id': '1234'
    },
    'invocation_id': 'test-invocation-id'
}]

# Dummy parameters for dynamic name building.
CATALOG_NAME = "AICatalog"
SCHEMA_NAME = "AISchema"
FUNCTION_NAME = "location_weather_in_c"

def test_execute_tool_calls_success():
    fake_client = FakeBedrockClient()
    # Call execute_tool_calls passing the dynamic values. Note that our function now splits the tool call function name,
    # and builds the full_function_name as: AICatalog.AISchema.location_weather_in_c.
    results = execute_tool_calls(
        tool_calls_sample,
        fake_client,
        catalog_name=CATALOG_NAME,
        schema_name=SCHEMA_NAME,
        function_name=FUNCTION_NAME
    )
    # Verify that we get one result with the expected invocation id and result value "23".
    assert isinstance(results, list)
    assert len(results) == 1
    result = results[0]
    assert result.get("invocation_id") == "test-invocation-id"
    assert result.get("result") == "23"

def test_full_function_name_building():
    fake_client = FakeBedrockClient()
    results = execute_tool_calls(
        tool_calls_sample,
        fake_client,
        catalog_name=CATALOG_NAME,
        schema_name=SCHEMA_NAME,
        function_name=FUNCTION_NAME
    )
    expected_full_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{FUNCTION_NAME}"
    function_info = fake_client.get_function(expected_full_name)
    assert function_info["full_name"] == expected_full_name
