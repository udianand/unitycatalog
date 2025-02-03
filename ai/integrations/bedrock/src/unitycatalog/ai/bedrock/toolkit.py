from typing import Any, Dict, List, Optional, Union, Iterator
from pprint import pprint
import boto3
from pydantic import BaseModel, ConfigDict, Field, model_validator

from unitycatalog.ai.core.client import UnitycatalogFunctionClient
from unitycatalog.ai.core.utils.client_utils import validate_or_set_default_client
from unitycatalog.ai.core.utils.function_processing_utils import (
    generate_function_input_params_schema,
    get_tool_name,
    process_function_names,
)

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
                      client: Optional[UnitycatalogFunctionClient] = None) -> List[Dict[str, Any]]:
    """Execute tool calls using Unity Catalog client."""
    client = validate_or_set_default_client(client)
    results = []
    for tool_call in tool_calls:
        function_name = "location_weather_in_c"
        print(f"Tool Call Function Name Local: {function_name}") #Debugging
        result = client.execute_function(tool_call['function_name'],
                                       tool_call['parameters'])
        results.append({
            'invocation_id': tool_call['invocation_id'],
            'result': str(result.value)
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

class BedrockSession:
    """Manages a session with AWS Bedrock agent runtime."""

    def __init__(self, agent_id: str, agent_alias_id: str):
        """Initialize a Bedrock session."""
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        self.client = boto3.client('bedrock-agent-runtime')

    def invoke_agent(
            self,
            input_text: str,
            enable_trace: bool = None,
            session_id: str = None,
            session_state: dict = None,
            uc_client: Optional[UnitycatalogFunctionClient] = None
    ) -> BedrockToolResponse:
        """Invoke the Bedrock agent with the given input text."""
        params = {
            'agentId': self.agent_id,
            'agentAliasId': self.agent_alias_id,
            'inputText': input_text,
        }

        if enable_trace is not None:
            params['enableTrace'] = enable_trace
        if session_id is not None:
            params['sessionId'] = session_id
        if session_state is not None:
            params['sessionState'] = session_state

        response = self.client.invoke_agent(**params)
        tool_calls = extract_tool_calls(response)

        if tool_calls and uc_client:
            print(f"Tool Results Enter:") #Debugging
            tool_results = execute_tool_calls(tool_calls, uc_client)
            print(f"ToolResults: {tool_results}") #Debugging
            if tool_results:
                session_state = generate_tool_call_session_state(
                    tool_results[0], tool_calls[0])
                print(f"SessionState: {session_state}") #Debugging
                return self.invoke_agent(input_text="",
                                       session_id=session_id,
                                       enable_trace=enable_trace,
                                       session_state=session_state)

        return BedrockToolResponse(raw_response=response, tool_calls=tool_calls)

class BedrockTool(BaseModel):
    """Model representing a Unity Catalog function as a Bedrock tool."""
    name: str = Field(description="The name of the function.")
    description: str = Field(description="A brief description of the function's purpose.")
    parameters: Dict[str, Any] = Field(description="The parameters schema required by the function.")
    requireConfirmation: str = Field(default="ENABLED", 
                                   description="Whether confirmation is required before executing the function.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Union[str, Dict[str, Any]]]:
        """Convert the tool to a dictionary format for Bedrock."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requireConfirmation": self.requireConfirmation
        }

class UCFunctionToolkit(BaseModel):
    """A toolkit for managing Unity Catalog functions and converting them into Bedrock tools."""
    function_names: List[str] = Field(default_factory=list)
    tools_dict: Dict[str, BedrockTool] = Field(default_factory=dict)
    client: Optional[UnitycatalogFunctionClient] = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def create_session(self, agent_id: str, agent_alias_id: str) -> BedrockSession:
        """Creates a new Bedrock session for interacting with an agent."""
        return BedrockSession(agent_id=agent_id, agent_alias_id=agent_alias_id)

    @model_validator(mode="after")
    def validate_toolkit(self) -> "UCFunctionToolkit":
        """Validates and processes the toolkit configuration."""
        self.client = validate_or_set_default_client(self.client)
        self.tools_dict = process_function_names(
            function_names=self.function_names,
            tools_dict=self.tools_dict,
            client=self.client,
            uc_function_to_tool_func=self.uc_function_to_bedrock_tool,
        )
        return self

    @staticmethod
    def uc_function_to_bedrock_tool(
        *,
        client: Optional[UnitycatalogFunctionClient] = None,
        function_name: Optional[str] = None,
        function_info: Optional[Any] = None,
    ) -> BedrockTool:
        """Converts a Unity Catalog function to a Bedrock tool."""
        if function_name and function_info:
            raise ValueError(
                "Only one of function_name or function_info should be provided."
            )

        client = validate_or_set_default_client(client)
        if function_name:
            function_info = client.get_function(function_name)
        elif function_info:
            function_name = function_info.full_name
        else:
            raise ValueError(
                "Either function_name or function_info should be provided.")

        fn_schema = generate_function_input_params_schema(function_info)
        parameters = {
            "type": "object",
            "properties": fn_schema.pydantic_model.model_json_schema().get("properties", {}),
            "required": fn_schema.pydantic_model.model_json_schema().get("required", []),
        }

        return BedrockTool(
            name=get_tool_name(function_name),
            description=function_info.comment or "",
            parameters=parameters,
        )

    @property
    def tools(self) -> List[BedrockTool]:
        """Gets all available tools."""
        return list(self.tools_dict.values())

    def get_tool(self, name: str) -> Optional[BedrockTool]:
        """Gets a specific tool by name."""
        return self.tools_dict.get(name)