
from unitycatalog.ai.core.client import UnitycatalogFunctionClient
from unitycatalog.client import ApiClient, Configuration
from unitycatalog.ai.bedrock.toolkit import UCFunctionToolkit
import boto3
import uuid
import os
from pprint import pprint


def setup_uc_client():
    """Set up Unity Catalog client"""
    config = Configuration()
    config.host = "http://0.0.0.0:8080/api/2.1/unity-catalog"
    api_client = ApiClient(configuration=config)
    return UnitycatalogFunctionClient(api_client=api_client)


def test_weather_function():
    """Test the weather function with Bedrock integration"""
    try:
        # Initialize Unity Catalog client
        client = setup_uc_client()

        # Define the weather function
        def location_weather_in_c(location_id: str, fetch_date: str) -> str:
            """Test function for AWS Bedrock integration.

            Args:
                location_id (str): The location ID for weather lookup
                fetch_date (str): The date to fetch weather for

            Returns:
                str: Weather result in Celsius.
            """
            return "23"

        # Create catalog and schema
        CATALOG = "AICatalog"
        SCHEMA = "AISchema"

        print("Creating catalog...")
        try:
            client.uc.create_catalog(name=CATALOG,
                                     comment="Catalog for AI functions")
        except Exception as e:
            if "already exists" not in str(e):
                raise e

        print("Creating schema...")
        try:
            client.uc.create_schema(catalog_name=CATALOG,
                                    name=SCHEMA,
                                    comment="Schema for AI functions")
        except Exception as e:
            if "already exists" not in str(e):
                raise e

        print("Creating function in Unity Catalog...")
        client.create_python_function(func=location_weather_in_c,
                                      catalog=CATALOG,
                                      schema=SCHEMA,
                                      replace=True)

        # Create toolkit with weather function
        function_name = f"{CATALOG}.{SCHEMA}.location_weather_in_c"
        toolkit = UCFunctionToolkit(function_names=[function_name],
                                    client=client)

        # Bedrock agent configuration
        agent_id = "AP5RQUVNTU"  # Replace with your agent ID
        agent_alias_id = "O6EXN8DJVZ"  # Replace with your alias ID

        # Create a session with Bedrock agent
        session = toolkit.create_session(agent_id=agent_id,
                                         agent_alias_id=agent_alias_id)
        session_id = str(uuid.uuid1())

        # Test streaming response
        print("\nTesting streaming response")
        response = session.invoke_agent(
            input_text="What is the weather for location 1234 and date of 2024-11-19",
            enable_trace=True,
            session_id=session_id,
            uc_client=client,
            stream=True)

        print("Streaming chunks:")
        for chunk in response.get_stream():
            print(chunk, end='', flush=True)

        if response.requires_tool_execution:
            print("\n\nTool calls required:")
            pprint(response.tool_calls)

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    test_weather_function()
