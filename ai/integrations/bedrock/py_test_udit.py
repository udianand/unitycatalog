from unitycatalog.ai.core.client import UnitycatalogFunctionClient
from unitycatalog.client import ApiClient, Configuration
from unitycatalog.ai.bedrock.toolkit import UCFunctionToolkit
import boto3
import uuid


def setup_uc_client():
    """Set up Unity Catalog client"""
    config = Configuration()
    config.host = "http://0.0.0.0:8080/api/2.1/unity-catalog"
    config.access_token = "dummy-token"  # Replace with your token if needed
    api_client = ApiClient(configuration=config)
    return UnitycatalogFunctionClient(api_client=api_client)


def setup_bedrock_session():
    """Set up AWS Bedrock session"""
    return boto3.Session()


def main():
    # Initialize Unity Catalog client
    uc_client = setup_uc_client()

    # Create toolkit with weather function
    function_name = "AICatalog.AISchema.location_weather_in_c"
    toolkit = UCFunctionToolkit(function_names=[function_name],
                                client=uc_client)

    # Bedrock agent configuration
    agent_id = "your_agent_id"  # Replace with your Bedrock agent ID
    agent_alias_id = "your_alias_id"  # Replace with your Bedrock agent alias ID

    # Create a session with Bedrock agent
    session = toolkit.create_session(agent_id=agent_id,
                                     agent_alias_id=agent_alias_id)

    # Generate unique session ID
    session_id = str(uuid.uuid1())

    try:
        # Invoke agent with a question
        response = session.invoke_agent(
            input_text="What's the weather in London?",
            enable_trace=True,
            session_id=session_id)

        # Process the agent's response
        for event in response['completion']:
            if 'returnControl' in event:
                # Handle function invocation
                function_input = event["returnControl"]["invocationInputs"][0][
                    "functionInvocationInput"]
                print(f"Function to call: {function_input['function']}")
                print(f"Parameters: {function_input['parameters']}")

                # Example weather result
                weather_result = "23"  # This would come from your actual function

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
                for final_event in final_response['completion']:
                    print(f"Agent response: {final_event}")

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
