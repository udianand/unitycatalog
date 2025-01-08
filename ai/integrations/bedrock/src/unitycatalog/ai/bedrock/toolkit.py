
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unitycatalog.ai.core.base import BaseFunctionClient
from unitycatalog.ai.core.utils.client_utils import validate_or_set_default_client
from unitycatalog.ai.core.utils.function_processing_utils import (
    generate_function_input_params_schema,
    get_tool_name,
    process_function_names,
)


class BedrockActionGroupFunction(BaseModel):
    """
    Model representing a Bedrock Action Group function.
    """

    name: str = Field(
        description="The name of the function.",
    )
    description: str = Field(
        description="A brief description of the function's purpose.",
    )
    parameters: Dict[str, Any] = Field(
        description="The parameters schema required by the function."
    )
    requireConfirmation: str = Field(
        default="ENABLED",
        description="Whether confirmation is required before executing the function."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the BedrockActionGroupFunction instance into a dictionary for the Bedrock API.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requireConfirmation": self.requireConfirmation
        }


class UCFunctionToolkit(BaseModel):
    """
    A toolkit for managing Unity Catalog functions and converting them into Bedrock Action Group functions.
    """

    function_names: List[str] = Field(
        default_factory=list,
        description="List of function names in 'catalog.schema.function' format.",
    )
    tools_dict: Dict[str, BedrockActionGroupFunction] = Field(
        default_factory=dict,
        description="Dictionary mapping function names to their corresponding Bedrock Action Group functions.",
    )
    client: Optional[BaseFunctionClient] = Field(
        default=None, description="The client for managing functions."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_toolkit(self) -> "UCFunctionToolkit":
        """
        Validates the toolkit, ensuring the client is properly set and function names are processed.
        """
        self.client = validate_or_set_default_client(self.client)

        self.tools_dict = process_function_names(
            function_names=self.function_names,
            tools_dict=self.tools_dict,
            client=self.client,
            uc_function_to_tool_func=self.uc_function_to_bedrock_action,
        )
        return self

    @staticmethod
    def uc_function_to_bedrock_action(
        *,
        client: Optional[BaseFunctionClient] = None,
        function_name: Optional[str] = None,
        function_info: Optional[Any] = None,
    ) -> BedrockActionGroupFunction:
        """
        Converts a Unity Catalog function to a Bedrock Action Group function.

        Args:
            client (Optional[BaseFunctionClient]): The client for managing functions.
            function_name (Optional[str]): The full name of the function in 'catalog.schema.function' format.
            function_info (Optional[Any]): The function info object returned by the client.

        Returns:
            BedrockActionGroupFunction: The corresponding Bedrock Action Group function.
        """
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

        return BedrockActionGroupFunction(
            name=get_tool_name(function_name),
            description=function_info.comment or "",
            parameters=parameters,
        )

    @property
    def action_group_functions(self) -> List[BedrockActionGroupFunction]:
        """
        Retrieves the list of Bedrock Action Group functions managed by the toolkit.
        """
        return list(self.tools_dict.values())
