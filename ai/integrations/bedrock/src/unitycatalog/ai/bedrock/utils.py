
from typing import Dict, Any, List
import time

def extract_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extracts tool calls from Bedrock response."""
    tool_calls = []
    for event in response.get('completion', []):
        if 'returnControl' in event:
            control_data = event['returnControl']
            for invocation in control_data.get('invocationInputs', []):
                if 'functionInvocationInput' in invocation:
                    func_input = invocation['functionInvocationInput']
                    func_name = f"{func_input['actionGroup']}__{func_input['function']}"
                    tool_calls.append({
                        'function_name': func_name,
                        'parameters': {
                            p['name']: p['value']
                            for p in func_input['parameters']
                        },
                        'invocation_id': control_data['invocationId']
                    })
    return tool_calls

def execute_tool_calls(tool_calls: List[Dict[str, Any]], 
                      client: Any,
                      catalog_name: str,
                      schema_name: str,
                      function_name: str) -> List[Dict[str, Any]]:
    """Execute tool calls and return results."""
    results = []
    for tool_call in tool_calls:
        try:
            full_function_name = f"{catalog_name}.{schema_name}.{function_name}"
            print(f"Full Function Name: {full_function_name}") #Debugging
            function_info = client.get_function(full_function_name)
            print(f"Retrieved function info Override: {function_info}")
            
            result = client.execute_function(
                full_function_name,
                tool_call['parameters']
            )
            results.append({
                'invocation_id': tool_call['invocation_id'],
                'result': str(result.value)
            })
        except Exception as e:
            print(f"Error executing tool call for {tool_call}: {e}") #Debugging
            results.append({
                'invocation_id': tool_call['invocation_id'],
                'error': str(e)
            })
    return results

def generate_tool_call_session_state(tool_result: Dict[str, Any], 
                                   tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Generate session state for tool call results."""
    action_group, function = tool_call['function_name'].split('__')
    return {
        'invocationId': tool_result['invocation_id'],
        'returnControlInvocationResults': [{
            'functionResult': {
                'actionGroup': action_group,
                'function': function,
                'confirmationState': 'CONFIRM',
                'responseBody': {
                    'TEXT': {
                        'body': tool_result['result']
                    }
                }
            }
        }]
    }
