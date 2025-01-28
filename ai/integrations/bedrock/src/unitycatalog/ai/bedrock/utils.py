
from typing import Any, Dict, List, Optional, Union, Iterator
from pydantic import BaseModel, Field
from unitycatalog.ai.core.base import BaseFunctionClient
from unitycatalog.ai.core.utils.client_utils import validate_or_set_default_client
from unitycatalog.ai.core.utils.function_processing_utils import construct_original_function_name


class BedrockToolResponse(BaseModel):
    """Class to handle Bedrock agent responses and tool calls."""
    raw_response: Dict[str, Any]
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)

    @property
    def requires_tool_execution(self) -> bool:
        """Returns True if the response requires tool execution."""
        return any('returnControl' in event
                   for event in self.raw_response.get('completion', []))

    @property
    def final_response(self) -> Optional[str]:
        """Returns the final text response if available."""
        if not self.requires_tool_execution:
            for event in self.raw_response.get('completion', []):
                if 'chunk' in event:
                    return event['chunk'].get('bytes', b'').decode('utf-8')
        return None

    @property
    def is_streaming(self) -> bool:
        """Returns True if the response is a streaming response."""
        return 'chunk' in str(self.raw_response)

    def get_stream(self) -> Iterator[str]:
        """Yields chunks from a streaming response."""
        if not self.is_streaming:
            return

        for event in self.raw_response.get('completion', []):
            if 'chunk' in event:
                chunk = event['chunk'].get('bytes', b'').decode('utf-8')
                if chunk:
                    yield chunk


def extract_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extracts tool calls from Bedrock response."""
    tool_calls = []

    for event in response.get('completion', []):
        if 'returnControl' in event:
            control_data = event['returnControl']
            for invocation in control_data.get('invocationInputs', []):
                if 'functionInvocationInput' in invocation:
                    func_input = invocation['functionInvocationInput']
                    func_name = construct_original_function_name(
                        f"{func_input['actionGroup']}__{func_input['function']}"
                    )
                    tool_calls.append({
                        'function_name':
                        func_name,
                        'parameters': {
                            p['name']: p['value']
                            for p in func_input['parameters']
                        },
                        'invocation_id':
                        control_data['invocationId']
                    })
    return tool_calls


def execute_tool_calls(
        tool_calls: List[Dict[str, Any]],
        client: Optional[BaseFunctionClient] = None) -> List[Dict[str, Any]]:
    """Execute tool calls using Unity Catalog client."""
    client = validate_or_set_default_client(client)
    results = []

    for tool_call in tool_calls:
        result = client.execute_function(tool_call['function_name'],
                                         tool_call['parameters'])
        results.append({
            'invocation_id': tool_call['invocation_id'],
            'result': str(result.value)
        })

    return results


def generate_tool_call_session_state(
        tool_result: Dict[str, Any], tool_call: Dict[str,
                                                     Any]) -> Dict[str, Any]:
    """Generate session state for tool call results."""
    return {
        'invocationId':
        tool_result['invocation_id'],
        'returnControlInvocationResults': [{
            'functionResult': {
                'actionGroup': tool_call['function_name'].split('__')[0],
                'function': tool_call['function_name'].split('__')[1],
                'confirmationState': 'CONFIRM',
                'responseBody': {
                    'TEXT': {
                        'body': tool_result['result']
                    }
                }
            }
        }]
    }
