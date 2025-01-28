from typing import Any, Dict, List, Optional, Union
import boto3
from pydantic import BaseModel, ConfigDict, Field, model_validator

from unitycatalog.ai.core.base import BaseFunctionClient
from unitycatalog.ai.core.utils.client_utils import validate_or_set_default_client
from unitycatalog.ai.core.utils.function_processing_utils import (
    generate_function_input_params_schema,
    get_tool_name,
    process_function_names,
)


class BedrockSession:
    """Manages a session with AWS Bedrock agent runtime."""

    def __init__(self, agent_id: str, agent_alias_id: str):
        """
        Initialize a Bedrock session.

        Args:
            agent_id: The ID of the Bedrock agent
            agent_alias_id: The alias ID of the Bedrock agent
        """
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        self.client = boto3.client('bedrock-agent-runtime')

    def invoke_agent(
            self,
            input_text: str,
            enable_trace: bool = None,
            session_id: str = None,
            session_state: dict = None,
            uc_client: Optional[BaseFunctionClient] = None,
            stream: bool = False
    ) -> BedrockToolResponse:
        """
        Invoke the Bedrock agent with the given input text.

        Args:
            input_text: The text input to send to the agent
            enable_trace: Enable detailed trace information about request handling
            session_id: Unique ID for the session
            session_state: Optional session state for the agent
            uc_client: Optional Unity Catalog client for executing functions
            stream: Whether to stream the response

        Returns:
            BedrockToolResponse containing the agent's response and handled tool calls
        """
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
        if stream:
            params['streaming'] = True

        response = self.client.invoke_agent(**params)
        tool_calls = extract_tool_calls(response)

        if tool_calls and uc_client:
            tool_results = execute_tool_calls(tool_calls, uc_client)
            if tool_results:
                # Make follow-up call with tool results
                session_state = generate_tool_call_session_state(
                    tool_results[0], tool_calls[0])
                return self.invoke_agent(input_text="",
                                         session_id=session_id,
                                         enable_trace=enable_trace,
                                         session_state=session_state,
                                         stream=stream)

        return BedrockToolResponse(raw_response=response,
                                   tool_calls=tool_calls)

    def start_session(self, alias_id: str = None, session_ttl: int = None):
        """
        Start a new session with the Bedrock agent.

        Args:
            alias_id: The alias ID to use. Defaults to the one set in constructor
            session_ttl: Time-to-live duration for the session in seconds

        Returns:
            Response from the start session API
        """
        params = {
            'agentId': self.agent_id,
            'agentAliasId': alias_id or self.agent_alias_id
        }

        if session_ttl is not None:
            params['sessionTtl'] = session_ttl

        return self.client.start_agent_session(**params)

    def end_session(self, session_id: str):
        """
        End an existing session with the Bedrock agent.

        Args:
            session_id: ID of the session to end

        Returns:
            Response from the end session API
        """
        return self.client.delete_agent_session(
            agentId=self.agent_id,
            agentAliasId=self.agent_alias_id,
            sessionId=session_id)


class BedrockTool(BaseModel):
    """Model representing a Unity Catalog function as a Bedrock tool."""
    name: str = Field(description="The name of the function.")
    description: str = Field(
        description="A brief description of the function's purpose.")
    parameters: Dict[str, Any] = Field(
        description="The parameters schema required by the function.")
    requireConfirmation: str = Field(
        default="ENABLED",
        description=
        "Whether confirmation is required before executing the function.")

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

    function_names: List[str] = Field(
        default_factory=list,
        description=
        "List of function names in 'catalog.schema.function' format.")
    tools_dict: Dict[str, BedrockTool] = Field(
        default_factory=dict,
        description=
        "Dictionary mapping function names to their corresponding Bedrock tools."
    )
    client: Optional[BaseFunctionClient] = Field(
        default=None, description="The client for managing functions.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def create_session(self, agent_id: str,
                       agent_alias_id: str) -> BedrockSession:
        """
        Creates a new Bedrock session for interacting with an agent.

        Args:
            agent_id: The ID of the Bedrock agent
            agent_alias_id: The alias ID of the Bedrock agent

        Returns:
            BedrockSession: A new session object
        """
        return BedrockSession(agent_id=agent_id, agent_alias_id=agent_alias_id)

    @model_validator(mode="after")
    def validate_toolkit(self) -> "UCFunctionToolkit":
        """
        Validates and processes the toolkit configuration after initialization.
        Converts UC functions to Bedrock tools.
        """
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
        client: Optional[BaseFunctionClient] = None,
        function_name: Optional[str] = None,
        function_info: Optional[Any] = None,
    ) -> BedrockTool:
        """
        Converts a Unity Catalog function to a Bedrock tool.

        Args:
            client: The Unity Catalog function client
            function_name: The name of the function to convert
            function_info: Optional pre-fetched function information

        Returns:
            BedrockTool: The converted tool
        """
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
            "type":
            "object",
            "properties":
            fn_schema.pydantic_model.model_json_schema().get("properties", {}),
            "required":
            fn_schema.pydantic_model.model_json_schema().get("required", []),
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
        """
        Gets a specific tool by name.

        Args:
            name: The name of the tool to retrieve

        Returns:
            Optional[BedrockTool]: The tool if found, None otherwise
        """
        return self.tools_dict.get(name)
