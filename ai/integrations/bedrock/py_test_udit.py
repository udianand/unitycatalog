from unitycatalog.ai.core.client import UnitycatalogFunctionClient
from unitycatalog.client import ApiClient, Configuration
from unitycatalog.ai.bedrock.toolkit import UCFunctionToolkit
import boto3

# Set up UC client with proper host and auth
config = Configuration()
config.host = "http://0.0.0.0:8080/api/2.1/unity-catalog"  # Update host if needed
config.access_token = "dummy-token"  # Add proper auth token if needed
api_client = ApiClient(configuration=config)
client = UnitycatalogFunctionClient(api_client=api_client)

# Create toolkit with your UC function
function_name = "AICatalog.AISchema.location_weather_in_c"  # Replace with your function name
toolkit = UCFunctionToolkit(function_names=[function_name], client=client)

# Create a Bedrock session with your agent IDs
session = toolkit.create_session(
    agent_id="your_bedrock_agent_id",  # Replace with your Bedrock agent ID
    agent_alias_id="your_bedrock_alias_id"  # Replace with your Bedrock agent alias ID
)

# Invoke the agent with your prompt
response = session.invoke_agent("What's the weather in London?")
print(response)