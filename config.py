import os
import sys
from dotenv import load_dotenv

def load_env():
    env_path = os.path.abspath(".env")

    if not os.path.exists(env_path):
        raise FileNotFoundError("Could not find .env file")

    load_dotenv(env_path)

    # Check for required field
    required_fields = ["VISION_API_KEY"]
    missing_fields = [field for field in required_fields if field not in os.environ]

    if missing_fields:    
        for missing_field in missing_fields:
                if missing_field == "VISION_API_KEY":
                    missing_vision_api_message = f"""Missing required field {missing_field} in .env file. 
                Please set this field to your Google Vision API key and try again.
                You can find your API key in the Google Cloud Console under 'APIs & Services' > 'Credentials'.
                    """
                    raise ValueError(missing_vision_api_message)
        missing_field_message = f"Missing required fields {missing_fields} in .env file."
        raise ValueError(missing_field_message)         

    # Set config to a dictionary of the environment variables
    config = {key: os.environ[key] for key in os.environ if key in required_fields}
    return config
def modules():
    sys.path.insert(0, './speech-bubble-detector')
   # sys.path.insert(1, './speech-bubble-detector/utils')
modules()
