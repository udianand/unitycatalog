
from typing import Any, Dict, List, Optional
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
    def __init__(self, agent_id: str, agent_alias_id: str):
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        self.client = boto3.client('bedrock-agent-runtime')
        
    def invoke_agent(self, input_text: str):
        return self.client.invoke_agent(
            agentId=self.agent_id,
            agentAliasId=self.agent_alias_id,
            inputText=input_text
        )

class BedrockTool(BaseModel):
    """Model representing a Bedrock tool."""
    name: str = Field(description="The name of the function.")
    description: str = Field(description="A brief description of the function's purpose.")
    parameters: Dict[str, Any] = Field(description="The parameters schema required by the function.")
    requireConfirmation: str = Field(
        default="ENABLED",
        description="Whether confirmation is required before executing the function."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
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
        description="List of function names in 'catalog.schema.function' format."
    )
    tools_dict: Dict[str, BedrockTool] = Field(
        default_factory=dict,
        description="Dictionary mapping function names to their corresponding Bedrock tools."
    )
    client: Optional[BaseFunctionClient] = Field(
        default=None, description="The client for managing functions."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def create_session(self, agent_id: str, agent_alias_id: str) -> BedrockSession:
        """Creates a new Bedrock session."""
        return BedrockSession(agent_id=agent_id, agent_alias_id=agent_alias_id)

    @model_validator(mode="after")
    def validate_toolkit(self) -> "UCFunctionToolkit":
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
        if function_name and function_info:
            raise ValueError("Only one of function_name or function_info should be provided.")
        
        client = validate_or_set_default_client(client)
        if function_name:
            function_info = client.get_function(function_name)
        elif function_info:
            function_name = function_info.full_name
        else:
            raise ValueError("Either function_name or function_info should be provided.")

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
