
import pytest
import boto3
import json
from moto import mock_bedrock
from unittest.mock import patch, MagicMock
from unitycatalog.ai.bedrock.toolkit import BedrockSession, BedrockToolResponse, UCFunctionToolkit
from unitycatalog.ai.core.client import UnitycatalogFunctionClient

# Mock response for invoke_agent
MOCK_INVOKE_RESPONSE = {
    "completion": [
        {
            "returnControl": {
                "invocationId": "test-invocation-id",
                "invocationInputs": [
                    {
                        "functionInvocationInput": {
                            "actionGroup": "weather",
                            "function": "get_weather",
                            "parameters": [
                                {"name": "location", "value": "1234"},
                                {"name": "date", "value": "2024-01-01"}
                            ]
                        }
                    }
                ]
            }
        }
    ]
}

@pytest.fixture
def mock_uc_client():
    client = MagicMock(spec=UnitycatalogFunctionClient)
    client.get_function.return_value = {
        "full_name": "test_catalog.test_schema.test_function",
        "comment": "Test function",
        "routine_body": "EXTERNAL"
    }
    client.execute_function.return_value = MagicMock(value="23")
    return client

@pytest.fixture
def mock_bedrock_client():
    with mock_bedrock():
        client = boto3.client('bedrock-agent-runtime')
        return client

@pytest.fixture
def bedrock_session(mock_bedrock_client):
    return BedrockSession(
        agent_id="test-agent",
        agent_alias_id="test-alias",
        catalog_name="test_catalog",
        schema_name="test_schema",
        function_name="test_function"
    )

def test_toolkit_initialization():
    toolkit = UCFunctionToolkit(function_names=["test_function"])
    assert isinstance(toolkit, UCFunctionToolkit)
    assert toolkit.function_names == ["test_function"]

def test_create_session(mock_bedrock_client):
    toolkit = UCFunctionToolkit(function_names=["test_function"])
    session = toolkit.create_session(
        agent_id="test-agent",
        agent_alias_id="test-alias",
        catalog_name="test_catalog",
        schema_name="test_schema",
        function_name="test_function"
    )
    assert isinstance(session, BedrockSession)
    assert session.agent_id == "test-agent"
    assert session.agent_alias_id == "test-alias"

@patch('boto3.client')
def test_invoke_agent_with_tool_calls(mock_boto3_client, mock_uc_client):
    # Setup mock response
    mock_client = MagicMock()
    mock_client.invoke_agent.return_value = MOCK_INVOKE_RESPONSE
    mock_boto3_client.return_value = mock_client

    session = BedrockSession(
        agent_id="test-agent",
        agent_alias_id="test-alias",
        catalog_name="test_catalog",
        schema_name="test_schema",
        function_name="test_function"
    )

    response = session.invoke_agent(
        input_text="test input",
        enable_trace=True,
        session_id="test-session",
        uc_client=mock_uc_client
    )

    assert isinstance(response, BedrockToolResponse)
    assert len(response.tool_calls) > 0
    mock_client.invoke_agent.assert_called_once()

def test_bedrock_tool_response_properties():
    response = BedrockToolResponse(
        raw_response={
            "completion": [
                {"chunk": {"bytes": b"test response"}}
            ]
        }
    )
    
    assert response.final_response == "test response"
    assert not response.requires_tool_execution
    assert response.is_streaming

def test_bedrock_tool_response_streaming():
    response = BedrockToolResponse(
        raw_response={
            "completion": [
                {"chunk": {"bytes": b"chunk1"}},
                {"chunk": {"bytes": b"chunk2"}}
            ]
        }
    )
    
    chunks = list(response.get_stream())
    assert len(chunks) == 2
    assert chunks == ["chunk1", "chunk2"]

@patch('boto3.client')
def test_invoke_agent_error_handling(mock_boto3_client):
    mock_client = MagicMock()
    mock_client.invoke_agent.side_effect = Exception("Test error")
    mock_boto3_client.return_value = mock_client

    session = BedrockSession(
        agent_id="test-agent",
        agent_alias_id="test-alias",
        catalog_name="test_catalog",
        schema_name="test_schema",
        function_name="test_function"
    )

    with pytest.raises(Exception) as exc_info:
        session.invoke_agent(input_text="test input")
    
    assert str(exc_info.value) == "Test error"
