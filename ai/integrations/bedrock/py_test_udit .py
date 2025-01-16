from unitycatalog.ai.core.client import UnitycatalogFunctionClient
from unitycatalog.client import ApiClient, Configuration
from unitycatalog.ai.bedrock.toolkit import UCFunctionToolkit


# Set up UC client
config = Configuration()
config.host = "http://localhost:8080/api/2.1/unity-catalog"
api_client = ApiClient(configuration=config)
client = UnitycatalogFunctionClient(api_client=api_client)


# Sample weather function 

from datetime import datetime

def location_weather_in_c(location_id, fetch_date):
    try:
        # Fetch from Databricks SQL Warehouse based UC function execution 
        return "23"
    except Exception as e:
        raise Exception(f"Error occurred: {e}")


# Create toolkit with the weather function
function_name = f"AICatalog.AISchema.location_weather_in_c"
toolkit = UCFunctionToolkit(function_names=[function_name], client=client)