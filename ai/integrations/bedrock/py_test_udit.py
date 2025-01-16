
from unitycatalog.ai.core.client import UnitycatalogFunctionClient
from unitycatalog.client import ApiClient, Configuration
from unitycatalog.ai.bedrock.toolkit import UCFunctionToolkit
import boto3
import uuid
import os

def setup_uc_client():
    """Set up Unity Catalog client"""
    config = Configuration()
    config.host = "http://0.0.0.0:8080/api/2.1/unity-catalog"
    config.access_token = os.getenv("UC_TOKEN", "dummy-token")
    api_client = ApiClient(configuration=config)
    return UnitycatalogFunctionClient(api_client=api_client)

def test_weather_function():
    """Test the weather function with Bedrock integration"""
    try:
        # Initialize Unity Catalog client
        uc_client = setup_uc_client()

        # Create toolkit with weather function
        function_name = "AICatalog.AISchema.location_weather_in_c"
        toolkit = UCFunctionToolkit(function_names=[function_name],
                                  client=uc_client)

        # Bedrock agent configuration
        agent_id = os.getenv("BEDROCK_AGENT_ID", "your_agent_id")
        agent_alias_id = os.getenv("BEDROCK_ALIAS_ID", "your_alias_id")

        # Create a session with Bedrock agent
        session = toolkit.create_session(agent_id=agent_id,
                                      agent_alias_id=agent_alias_id)

        # Generate unique session ID
        session_id = str(uuid.uuid1())

        # Test cases with different cities
        test_cities = ["London", "New York", "Tokyo", "Paris"]
        
        for city in test_cities:
            print(f"\nTesting weather query for {city}")
            
            # Invoke agent with a weather question
            response = session.invoke_agent(
                input_text=f"What's the weather in {city}?",
                enable_trace=True,
                session_id=session_id)

            # Process the agent's response
            for event in response.get('completion', []):
                if 'returnControl' in event:
                    # Handle function invocation
                    function_input = event["returnControl"]["invocationInputs"][0][
                        "functionInvocationInput"]
                    print(f"Function to call: {function_input['function']}")
                    print(f"Parameters: {function_input['parameters']}")

                    # Simulate weather result (replace with actual API call)
                    weather_result = "23"  # Example temperature

                    # Send result back to agent
                    final_response = session.invoke_agent(
                        input_text="",
                        session_id=session_id,
                        enable_trace=True,
                        session_state={
                            'invocationId':
                            event["returnControl"]["invocationId"],
                            'returnControlInvocationResults': [{
                                'functionResult': {
                                    'actionGroup': function_input["actionGroup"],
                                    'function': function_input["function"],
                                    'confirmationState': 'CONFIRM',
                                    'responseBody': {
                                        "TEXT": {
                                            'body':
                                            f"weather_in_centigrade: {weather_result}"
                                        }
                                    }
                                }
                            }]
                        })

                    # Print final response
                    print("Agent Response:")
                    for final_event in final_response.get('completion', []):
                        print(f"  {final_event}")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    test_weather_function()
