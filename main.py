from flask import Flask, request, jsonify, make_response, Response
import requests
import time
import uuid
import warnings
from waitress import serve
import json
import tiktoken
import socket
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from pymemcache.client.base import Client
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import logging
from io import BytesIO
import coloredlogs
import printedcolors
import base64
import tempfile
import re
import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor
import docx2txt
from docx import Document as DocxDocument

# Suppress warnings from flask_limiter
warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter.extension")

# Set default port and debug mode
PORT = os.environ.get('PORT', 5001)
DEBUG_MODE = os.environ.get('DEBUG', 'False').lower() == 'true'

# Create a logger object
logger = logging.getLogger("1min-relay")

# Install coloredlogs with desired log level
coloredlogs.install(level='DEBUG', logger=logger)

warnings.filterwarnings("ignore", category=FutureWarning, module="mistral_common.tokens.tokenizers.mistral")

def check_memcached_connection(host='memcached', port=11211):
    try:
        client = Client((host, port))
        client.set('test_key', 'test_value')
        if client.get('test_key') == b'test_value':
            client.delete('test_key')  # Clean up
            return True
        else:
            return False
    except:
        return False
        
logger.info('''
    _ __  __ _      ___     _           
 / |  \/  (_)_ _ | _ \___| |__ _ _  _ 
 | | |\/| | | ' \|   / -_) / _` | || |
 |_|_|  |_|_|_||_|_|_\___|_\__,_|\_, |
                                 |__/ ''')


def calculate_token(sentence, model="DEFAULT"):
    """Calculate the number of tokens in a sentence based on the specified model."""
    
    if model.startswith("mistral"):
        # Initialize the Mistral tokenizer
        tokenizer = MistralTokenizer.v3(is_tekken=True)
        model_name = "open-mistral-nemo" # Default to Mistral Nemo
        tokenizer = MistralTokenizer.from_model(model_name)
        tokenized = tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                messages=[
                    UserMessage(content=sentence),
                ],
                model=model_name,
            )
        )
        tokens = tokenized.tokens
        return len(tokens)

    elif model in ["gpt-3.5-turbo", "gpt-4"]:
        # Use OpenAI's tiktoken for GPT models
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(sentence)
        return len(tokens)

    else:
        # Default to openai
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(sentence)
        return len(tokens)

# Initialize Flask app
app = Flask(__name__)
if check_memcached_connection():
    limiter = Limiter(
        get_remote_address,
        app=app,
        storage_uri="memcached://memcached:11211",  # Connect to Memcached created with docker
    )
else:
    # Used for ratelimiting without memcached
    limiter = Limiter(
        get_remote_address,
        app=app,
    )
    logger.warning("Memcached is not available. Using in-memory storage for rate limiting. Not-Recommended")


# 1minAI API endpoints
ONE_MIN_API_URL = "https://api.1min.ai/api/features"
ONE_MIN_CONVERSATION_API_URL = "https://api.1min.ai/api/conversations"
ONE_MIN_CONVERSATION_API_STREAMING_URL = "https://api.1min.ai/api/features?isStreaming=true"
ONE_MIN_ASSET_URL = "https://api.1min.ai/api/assets"
ONE_MIN_TOOLS_URL = "https://api.1min.ai/api/tools"
ONE_MIN_TEXT_TO_SPEECH_URL = "https://api.1min.ai/api/text-to-speech"
ONE_MIN_SPEECH_TO_TEXT_URL = "https://api.1min.ai/api/speech-to-text"
ONE_MIN_SEARCH_URL = "https://api.1min.ai/api/websearch"
ONE_MIN_ANALYTICS_URL = "https://api.1min.ai/api/analytics"
ONE_MIN_EMBEDDINGS_API_URL = "https://api.1min.ai/api/embeddings"
ONE_MIN_IMAGE_API_URL = "https://api.1min.ai/api/images"
ONE_MIN_MODERATION_API_URL = "https://api.1min.ai/api/moderations"
ONE_MIN_ASSISTANTS_API_URL = "https://api.1min.ai/api/assistants"
ONE_MIN_FILES_API_URL = "https://api.1min.ai/api/files"

# Define the models that are available for use
ALL_ONE_MIN_AVAILABLE_MODELS = [
    "deepseek-chat",
    "o1-preview",
    "o1-mini",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "claude-instant-1.2",
    "claude-2.1",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "gemini-1.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "mistral-large-latest",
    "mistral-small-latest",
    "mistral-nemo",
    "open-mistral-7b",

   # Replicate
   "meta/llama-2-70b-chat", 
   "meta/meta-llama-3-70b-instruct", 
   "meta/meta-llama-3.1-405b-instruct", 
   "command"
]

# Define the models that support vision inputs
vision_supported_models = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "gemini-1.5-pro",
    "gemini-1.5-flash"
]

# Define models that support tool use (function calling)
tools_supported_models = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]

# Define models that support text-to-speech
# tts_supported_models = [
#     "nova", 
#     "alloy", 
#     "echo", 
#     "fable", 
#     "onyx", 
#     "shimmer"
# ]

# Define models that support web search
web_search_supported_models = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "mistral-large-latest",
    "mistral-small-latest",
    "mistral-nemo",
    "deepseek-chat"
]

# Default values
SUBSET_OF_ONE_MIN_PERMITTED_MODELS = ["mistral-nemo", "gpt-4o", "deepseek-chat"]
PERMIT_MODELS_FROM_SUBSET_ONLY = False

# Read environment variables
one_min_models_env = os.getenv("SUBSET_OF_ONE_MIN_PERMITTED_MODELS")  # e.g. "mistral-nemo,gpt-4o,deepseek-chat"
permit_not_in_available_env = os.getenv("PERMIT_MODELS_FROM_SUBSET_ONLY")  # e.g. "True" or "False"

# Parse or fall back to defaults
if one_min_models_env:
    SUBSET_OF_ONE_MIN_PERMITTED_MODELS = one_min_models_env.split(",")

if permit_not_in_available_env and permit_not_in_available_env.lower() == "true":
    PERMIT_MODELS_FROM_SUBSET_ONLY = True

# Combine into a single list
AVAILABLE_MODELS = []
AVAILABLE_MODELS.extend(SUBSET_OF_ONE_MIN_PERMITTED_MODELS)

# Default model to use
DEFAULT_MODEL = "mistral-nemo"

def map_model_to_openai(model):
    """Map 1minAI model name to OpenAI compatible model name"""
    if model == "mistral-nemo":
        return "mistral-7b-text-chat"
    elif model.startswith("gpt-"):
        return model  # Already in OpenAI format
    elif model.startswith("claude-"):
        return model  # Return as is
    else:
        # For other models, return as is but prefixed with 1min-
        return f"1min-{model}"

def ERROR_HANDLER(code, model=None, key=None, detail=None):
    # Handle errors in OpenAI-Structured Error
    error_codes = { # Internal Error Codes
        1002: {"message": f"The model {model} does not exist.", "type": "invalid_request_error", "param": None, "code": "model_not_found", "http_code": 400},
        1020: {"message": f"Incorrect API key provided: {key}. You can find your API key at https://app.1min.ai/api.", "type": "authentication_error", "param": None, "code": "invalid_api_key", "http_code": 401},
        1021: {"message": "Invalid Authentication", "type": "invalid_request_error", "param": None, "code": None, "http_code": 401},
        1212: {"message": f"Incorrect Endpoint. Please use the /v1/chat/completions endpoint.", "type": "invalid_request_error", "param": None, "code": "model_not_supported", "http_code": 400},
        1044: {"message": f"This model does not support image inputs.", "type": "invalid_request_error", "param": None, "code": "model_not_supported", "http_code": 400},
        1045: {"message": f"This model does not support tool use.", "type": "invalid_request_error", "param": None, "code": "model_not_supported", "http_code": 400},
        1046: {"message": f"This model does not support text-to-speech.", "type": "invalid_request_error", "param": None, "code": "model_not_supported", "http_code": 400},
        1412: {"message": f"No message provided.", "type": "invalid_request_error", "param": "messages", "code": "invalid_request_error", "http_code": 400},
        1423: {"message": f"No content in last message.", "type": "invalid_request_error", "param": "messages", "code": "invalid_request_error", "http_code": 400},
        1500: {"message": f"1minAI API error: {detail}", "type": "api_error", "param": None, "code": "api_error", "http_code": 500},
        1600: {"message": f"Unsupported feature: {detail}", "type": "invalid_request_error", "param": None, "code": "unsupported_feature", "http_code": 400},
        1700: {"message": f"Invalid file format: {detail}", "type": "invalid_request_error", "param": None, "code": "invalid_file_format", "http_code": 400},
    }
    error_data = {k: v for k, v in error_codes.get(code, {"message": f"Unknown error: {detail}" if detail else "Unknown error", "type": "unknown_error", "param": None, "code": None}).items() if k != "http_code"} # Remove http_code from the error data
    logger.error(f"An error has occurred while processing the user's request. Error code: {code}")
    return jsonify({"error": error_data}), error_codes.get(code, {}).get("http_code", 400) # Return the error data without http_code inside the payload and get the http_code to return.

def handle_options_request():
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response, 204

def extract_api_key():
    """Extract API key from Authorization header"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    return auth_header.split(" ")[1]

def set_response_headers(response):
    """Set common response headers"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return ERROR_HANDLER(1212)
    if request.method == 'GET':
        internal_ip = socket.gethostbyname(socket.gethostname())
        return "Congratulations! Your API is working! You can now make requests to the API.\n\nEndpoint: " + internal_ip + ':5001/v1'

@app.route('/v1/models')
@limiter.limit("500 per minute")
def models():
    # Dynamically create the list of models with additional fields
    models_data = []
    if not PERMIT_MODELS_FROM_SUBSET_ONLY:
        one_min_models_data = [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "1minai",
                "created": 1727389042
            }
            for model_name in ALL_ONE_MIN_AVAILABLE_MODELS
        ]
    else:
        one_min_models_data = [
            {"id": model_name, "object": "model", "owned_by": "1minai", "created": 1727389042}
            for model_name in SUBSET_OF_ONE_MIN_PERMITTED_MODELS
        ]
    models_data.extend(one_min_models_data)
    
    # Add TTS models
    # tts_models_data = [
    #     {"id": model_name, "object": "model", "owned_by": "1minai", "created": 1727389042}
    #     for model_name in tts_supported_models
    # ]
    # models_data.extend(tts_models_data)
    
    return jsonify({"data": models_data, "object": "list"})

def format_conversation_history(messages, new_input, system_prompt=None):
    """
    Formats the conversation history into a structured string.
    
    Args:
        messages (list): List of message dictionaries from the request
        new_input (str): The new user input message
        system_prompt (str): Optional system prompt to prepend
    
    Returns:
        str: Formatted conversation history
    """
    formatted_history = []
    
    # Add system prompt if provided
    if system_prompt:
        formatted_history.append(f"System: {system_prompt}\n")
    
    formatted_history.append("Conversation History:\n")
    
    for message in messages:
        role = message.get('role', '').capitalize()
        content = message.get('content', '')
        
        # Handle potential list content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if 'text' in item:
                    text_parts.append(item['text'])
                elif 'type' in item and item['type'] == 'text':
                    text_parts.append(item.get('text', ''))
            content = '\n'.join(text_parts)
        
        formatted_history.append(f"{role}: {content}")
    
    # Append additional messages only if there are existing messages
    if messages: # Save credits if it is the first message.
        formatted_history.append("Respond like normal. The conversation history will be automatically updated on the next MESSAGE. DO NOT ADD User: or Assistant: to your output. Just respond like normal.")
        formatted_history.append("User Message:\n")
    formatted_history.append(new_input) 
    
    return '\n'.join(formatted_history)

def extract_images_from_message(messages, api_key, model):
    """
    Extracts images from message content and uploads them to 1minAI
    
    Returns:
        tuple: (user_input as text, list of image paths, image flag)
    """
    image = False
    image_paths = []
    user_input = ""
    has_ignored_images = False
    
    if not messages:
        return user_input, image_paths, image
    
    last_message = messages[-1]
    content = last_message.get('content', '')
    
    # If content is not a list, return as is
    if not isinstance(content, list):
        return content, image_paths, image
    
    # Process multi-modal content (text + images)
    text_parts = []
    
    for item in content:
        # Extract text
        if 'text' in item:
            text_parts.append(item['text'])
        elif 'type' in item and item['type'] == 'text':
            text_parts.append(item.get('text', ''))
            
        # Extract and process images
        try:
            if 'image_url' in item:
                if model not in vision_supported_models:
                    # If model doesn't support images, ignore them and log warning
                    has_ignored_images = True
                    logger.warning(f"Model {model} does not support images in 1minAI API. Images will be ignored.")
                    continue
                
                # Process base64 images
                if isinstance(item['image_url'], dict) and 'url' in item['image_url']:
                    image_url = item['image_url']['url']
                    if image_url.startswith("data:image/"):
                        # Handle base64 encoded image
                        mime_type = re.search(r'data:(image/[^;]+);base64,', image_url)
                        mime_type = mime_type.group(1) if mime_type else 'image/png'
                        base64_image = image_url.split(",")[1]
                        binary_data = base64.b64decode(base64_image)
                        
                        # Create a BytesIO object
                        image_data = BytesIO(binary_data)
                    else:
                        # Handle URL images
                        response = requests.get(image_url)
                        response.raise_for_status()
                        image_data = BytesIO(response.content)
                        mime_type = response.headers.get('content-type', 'image/png')
                    
                    # Upload to 1minAI
                    headers = {"API-KEY": api_key}
                    
                    # Add a detail parameter for better image analysis of people
                    if 'detail' in item and item['detail'] == 'high':
                        # High detail for photos with people or complex scenes
                        detail_level = "high"
                    else:
                        detail_level = "auto"
                    
                    # Generate a unique filename
                    image_filename = f"relay_image_{uuid.uuid4()}.{mime_type.split('/')[-1]}"
                    
                    files = {
                        'asset': (image_filename, image_data, mime_type)
                    }
                    
                    asset_response = requests.post(ONE_MIN_ASSET_URL, files=files, headers=headers)
                    asset_response.raise_for_status()
                    
                    # Get image path and add to list
                    image_path = asset_response.json()['fileContent']['path']
                    image_paths.append(image_path)
                    image = True
                    
                    # For models that need specific guidance on image analysis
                    if 'claude' in model and not any(text in item.get('text', '') for text in ["describe", "what do you see", "analyze", "explain"]):
                        # Add image analysis instructions for Claude models to improve recognition of people
                        analysis_prompts = [
                            "Describe this image in detail. If there are people in the image, describe them. If there is text in the image, read it.",
                            "What do you see in this image? Please describe all elements, including any people, objects, text, and scenery."
                        ]
                        
                        # Add the prompt to the text part only if no other text was provided
                        if not text_parts:
                            text_parts.append(analysis_prompts[0])
        except Exception as e:
            logger.error(f"Error processing image: {str(e)[:100]}")
            # Continue to process other content even if one image fails
    
    # Combine all text parts
    user_input = '\n'.join(text_parts)
    
    # If images were ignored, add a warning to the beginning of the message
    if has_ignored_images:
        user_input = f"Note: This model cannot process images. Please use one of these models instead: {', '.join(vision_supported_models[:3])} and others.\n\n{user_input}"
    
    return user_input, image_paths, image

def process_tools(tools, tool_choice, model):
    """
    Process tools (function calling) for compatible models
    
    Args:
        tools (list): List of tool definitions
        tool_choice (str/dict): Tool choice configuration
        model (str): Model name
    
    Returns:
        dict: 1minAI compatible tools configuration
    """
    if not tools:
        # Add default tools for supported models
        if model in tools_supported_models:
            # Default tools for all supported models
            default_tools = [
                {
                    "name": "get_current_datetime",
                    "description": "Get the current date and time in various formats",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "format": {
                                "type": "string",
                                "description": "Date format (default: ISO format). Options: iso, rfc, human, unix"
                            },
                            "timezone": {
                                "type": "string",
                                "description": "Timezone (default: UTC). Example: Europe/London, America/New_York"
                            }
                        },
                        "required": []
                    }
                },
                {
                    "name": "execute_python",
                    "description": "Execute Python code and return the result",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Maximum execution time in seconds (default: 10)"
                            }
                        },
                        "required": ["code"]
                    }
                },
                {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            ]
            
            # Combine with any user-provided tools
            tools = tools + default_tools if tools else default_tools
        else:
            # If the model does not support tools, just return None
            logger.warning(f"Model {model} does not support tool use, ignoring tools parameter")
            return None
    elif model not in tools_supported_models:
        # If the model does not support tools, just return None
        logger.warning(f"Model {model} does not support tool use, ignoring tools parameter")
        return None
    
    # Convert OpenAI tools format to 1minAI format
    one_min_tools = []
    
    for tool in tools:
        # Currently only 'function' type is supported
        if tool.get('type', 'function') == 'function':
            function_def = tool.get('function', tool)  # Handle both formats
            one_min_tool = {
                "name": function_def.get('name', ''),
                "description": function_def.get('description', ''),
                "parameters": function_def.get('parameters', {})
            }
            one_min_tools.append(one_min_tool)
    
    # Process tool_choice
    auto_invoke = True  # Default
    if tool_choice == "none":
        auto_invoke = False
    elif isinstance(tool_choice, dict) and tool_choice.get('type') == 'function':
        # Specific function is requested
        # 1minAI doesn't directly support this, but we can add this to the prompt
        pass
    
    return {
        "tools": one_min_tools,
        "autoInvoke": auto_invoke,
        "handlers": {
            "get_current_datetime": handle_get_datetime,
            "execute_python": handle_execute_python,
            "web_search": handle_web_search
        }
    }

def handle_get_datetime(params):
    """
    Handle the get_current_datetime tool call
    
    Args:
        params (dict): Tool parameters
    
    Returns:
        dict: Tool response
    """
    try:
        # Get parameters with defaults
        format_type = params.get('format', 'iso').lower()
        timezone_str = params.get('timezone', 'UTC')
        
        # Import tz handling if timezone is specified
        if timezone_str != 'UTC':
            try:
                from dateutil import tz
                timezone = tz.gettz(timezone_str)
                if not timezone:
                    timezone = tz.UTC
            except ImportError:
                timezone = None
                timezone_str = 'UTC'
        else:
            timezone = None
        
        # Get current datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        
        # Apply timezone if specified
        if timezone:
            now = now.astimezone(timezone)
        
        # Format according to requested format
        if format_type == 'iso':
            formatted_date = now.isoformat()
        elif format_type == 'rfc':
            formatted_date = now.strftime('%a, %d %b %Y %H:%M:%S %z')
        elif format_type == 'human':
            formatted_date = now.strftime('%A, %B %d, %Y %I:%M:%S %p %Z')
        elif format_type == 'unix':
            formatted_date = str(int(now.timestamp()))
        else:
            # Default to ISO
            formatted_date = now.isoformat()
        
        return {
            "datetime": formatted_date,
            "timezone": timezone_str,
            "format": format_type,
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
            "weekday": now.strftime('%A')
        }
    except Exception as e:
        logger.error(f"Error handling datetime tool: {str(e)}")
        return {"error": str(e)}

def handle_execute_python(params):
    """
    Handle the execute_python tool call
    
    Args:
        params (dict): Tool parameters
    
    Returns:
        dict: Tool response
    """
    try:
        code = params.get('code', '')
        timeout = int(params.get('timeout', 10))
        
        # Limit timeout to reasonable values
        if timeout < 1:
            timeout = 1
        elif timeout > 30:
            timeout = 30
        
        if not code:
            return {"error": "No code provided"}
        
        # Execute the code
        result = execute_python_code(code, timeout)
        return result
    except Exception as e:
        logger.error(f"Error handling execute_python tool: {str(e)}")
        return {"error": str(e)}

def handle_web_search(params):
    """
    Handle the web_search tool call
    
    Args:
        params (dict): Tool parameters
    
    Returns:
        dict: Tool response
    """
    try:
        query = params.get('query', '')
        if not query:
            return {"error": "No search query provided"}
        
        # Extract API key from request
        api_key = extract_api_key()
        if not api_key:
            return {"error": "API key not found"}
        
        # Perform web search
        search_results = web_search(query, api_key)
        
        # Format results
        formatted_results = []
        for result in search_results.get('results', []):
            formatted_results.append({
                "title": result.get('title', ''),
                "url": result.get('url', ''),
                "snippet": result.get('snippet', ''),
                "date": result.get('date', '')
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }
    except Exception as e:
        logger.error(f"Error handling web_search tool: {str(e)}")
        return {"error": str(e)}

def process_tts_request(request_data):
    """
    Process text-to-speech request
    
    Args:
        request_data (dict): Request data
    
    Returns:
        Response: Flask response with audio data or error
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return ERROR_HANDLER(1021)
    
    api_key = auth_header.split(" ")[1]
    
    # Extract parameters
    input_text = request_data.get('input', '')
    model = request_data.get('model', 'nova')
    voice = request_data.get('voice', 'alloy')
    response_format = request_data.get('response_format', 'mp3')
    speed = request_data.get('speed', 1.0)
    
    if not input_text:
        return ERROR_HANDLER(1412, detail="No input text provided for TTS")
    
    # if model not in tts_supported_models:
    #     return ERROR_HANDLER(1046, model=model)
    
    # Prepare request to 1minAI
    headers = {"API-KEY": api_key}
    payload = {
        "text": input_text,
        "voice": voice,
        "model": model,
        "speed": speed,
        "format": response_format
    }
    
    try:
        response = requests.post(ONE_MIN_TEXT_TO_SPEECH_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        # Get the audio data
        tts_response = response.json()
        audio_url = tts_response.get('audioUrl')
        
        if not audio_url:
            return ERROR_HANDLER(1500, detail="No audio URL returned from 1minAI TTS API")
        
        # Download the audio file
        audio_response = requests.get(audio_url)
        audio_response.raise_for_status()
        
        # Create response
        flask_response = make_response(audio_response.content)
        flask_response.headers['Content-Type'] = f'audio/{response_format}'
        
        return flask_response
    except requests.exceptions.RequestException as e:
        return ERROR_HANDLER(1500, detail=str(e))

def process_stt_request():
    """
    Process speech-to-text request
    
    Returns:
        Response: Flask response with transcription or error
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return ERROR_HANDLER(1021)
    
    api_key = auth_header.split(" ")[1]
    
    # Check if file is uploaded
    if 'file' not in request.files:
        return ERROR_HANDLER(1700, detail="No audio file provided")
    
    audio_file = request.files['file']
    model = request.form.get('model', 'whisper-1')
    
    # Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        audio_file.save(temp_file.name)
        temp_file_path = temp_file.name
    
    try:
        # Upload to 1minAI
        headers = {"API-KEY": api_key}
        files = {
            'asset': (audio_file.filename, open(temp_file_path, 'rb'), audio_file.content_type)
        }
        
        asset_response = requests.post(ONE_MIN_ASSET_URL, files=files, headers=headers)
        asset_response.raise_for_status()
        
        # Get audio path
        audio_path = asset_response.json()['fileContent']['path']
        
        # Transcribe audio
        payload = {
            "audioPath": audio_path,
            "model": model
        }
        
        response = requests.post(ONE_MIN_SPEECH_TO_TEXT_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        # Format response
        stt_response = response.json()
        transcription = stt_response.get('text', '')
        
        return jsonify({
            "text": transcription
        })
    except requests.exceptions.RequestException as e:
        return ERROR_HANDLER(1500, detail=str(e))
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass

def web_search(query, api_key, num_results=5):
    """
    Perform a web search using 1minAI API
    
    Args:
        query (str): Search query
        api_key (str): API key
        num_results (int): Number of results to return (default: 5)
    
    Returns:
        dict: Search results
    """
    headers = {"API-KEY": api_key}
    payload = {
        "query": query,
        "numResults": min(max(num_results, 1), 10)  # Limit between 1-10
    }
    
    try:
        logger.info(f"Performing web search for: {query}")
        response = requests.post(ONE_MIN_SEARCH_URL, json=payload, headers=headers)
        response.raise_for_status()
        results = response.json()
        
        # Process and format the results
        processed_results = {
            "results": [],
            "query": query
        }
        
        for result in results.get('results', []):
            # Extract relevant information and format consistently
            processed_results['results'].append({
                "title": result.get('title', ''),
                "url": result.get('url', ''),
                "snippet": result.get('snippet', ''),
                "date": result.get('date', '')
            })
            
        logger.info(f"Web search returned {len(processed_results['results'])} results")
        return processed_results
    except requests.exceptions.RequestException as e:
        logger.error(f"Web search error: {str(e)}")
        return {"results": [], "query": query, "error": str(e)}

@app.route('/v1/audio/transcriptions', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def audio_transcriptions():
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    return process_stt_request()

@app.route('/v1/audio/speech', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def audio_speech():
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    # Temporarily return an error that the function is disabled
    return ERROR_HANDLER(1600, detail="TTS functionality is temporarily disabled")
    
    # request_data = request.json
    # return process_tts_request(request_data)

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def conversation():
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error("Invalid Authentication")
        return ERROR_HANDLER(1021)
    
    api_key = auth_header.split(" ")[1]
    request_data = request.json
    
    headers = {
        "API-KEY": api_key, 
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if request_data.get('stream', False) else "application/json"
    }
    
    # Get model
    model = request_data.get('model', 'mistral-nemo')
    
    # Check to see if the TTS model is a model
    # if model in tts_supported_models:
    #     return ERROR_HANDLER(1600, detail="TTS models can only be used with the /v1/audio/speech endpoint")
    
    if PERMIT_MODELS_FROM_SUBSET_ONLY and model not in AVAILABLE_MODELS:
        return ERROR_HANDLER(1002, model)
    
    # Get messages
    messages = request_data.get('messages', [])
    if not messages:
        return ERROR_HANDLER(1412)
    
    # Extract system message if present
    system_prompt = None
    for msg in messages:
        if msg.get('role') == 'system':
            system_prompt = msg.get('content', '')
            break
    
    # Extract user input from the last message
    try:
        user_input, image_paths, has_image = extract_images_from_message(messages, api_key, model)
    except Exception as e:
        logger.error(f"Error extracting images: {str(e)}")
        # If an error occurs during image processing, continue without images
        user_input = messages[-1].get('content', '')
        if isinstance(user_input, list):
            # Merge the text parts of the message
            text_parts = []
            for item in user_input:
                if 'text' in item:
                    text_parts.append(item['text'])
                elif 'type' in item and item['type'] == 'text':
                    text_parts.append(item.get('text', ''))
            user_input = '\n'.join(text_parts)
        
        image_paths = []
        has_image = False
        
        if not user_input:
            return ERROR_HANDLER(1423)
    
    # Format conversation history
    all_messages = format_conversation_history(messages, user_input, system_prompt)
    prompt_token = calculate_token(str(all_messages))
    
    # Process tools (function calling)
    tools_config = None
    try:
        if 'tools' in request_data:
            tools_config = process_tools(
                request_data.get('tools', []),
                request_data.get('tool_choice', 'auto'),
                model
            )
    except Exception as e:
        # Any errors other than tool incompatibilities are treated as internal errors
        return ERROR_HANDLER(1500, detail=str(e))
    
    # Check for web search
    use_web_search = request_data.get('web_search', False)
    if use_web_search and model not in web_search_supported_models:
        logger.warning(f"Model {model} does not support web search, ignoring web search parameter")
        use_web_search = False
    
    # Prepare base payload
    payload = {
        "model": model,
        "promptObject": {
            "prompt": all_messages,
            "isMixed": False,
            "webSearch": use_web_search
        }
    }
    
    # Set request type based on content
    if has_image:
        payload["type"] = "CHAT_WITH_IMAGE"
        payload["promptObject"]["imageList"] = image_paths
    else:
        payload["type"] = "CHAT_WITH_AI"
    
    # Add tools if configured
    if tools_config:
        payload["toolsConfig"] = tools_config
    
    # Add additional parameters
    if 'temperature' in request_data:
        payload["temperature"] = request_data['temperature']
    
    if 'top_p' in request_data:
        payload["topP"] = request_data['top_p']
    
    if 'max_tokens' in request_data:
        payload["maxTokens"] = request_data['max_tokens']
    
    # Handle response format
    response_format = request_data.get('response_format', {})
    if response_format.get('type') == 'json_object':
        payload["responseFormat"] = "json"
    
    logger.debug(f"Processing {prompt_token} prompt tokens with model {model}")
    
    # Check for cases where we need to disable streaming
    stream_enabled = request_data.get('stream', False)
    
    # Disable streaming for claude-instant-1.2 only, since it doesn't work with it for sure
    if model == "claude-instant-1.2" and stream_enabled:
        logger.warning(f"Model {model} might have issues with streaming, falling back to non-streaming mode")
        stream_enabled = False
    
    # For all other models we will try streaming if requested
    
    if not stream_enabled:
        # Non-Streaming Response
        logger.debug("Non-Streaming AI Response")
        try:
            response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            one_min_response = response.json()
            
            transformed_response = transform_response(one_min_response, request_data, prompt_token)
            flask_response = make_response(jsonify(transformed_response))
            set_response_headers(flask_response)
            
            return flask_response, 200
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return ERROR_HANDLER(1020, key="[REDACTED]")
            return ERROR_HANDLER(1500, detail=str(e))
    
    else:
        # Streaming Response
        logger.debug("Streaming AI Response")
        try:
            response_stream = requests.post(ONE_MIN_CONVERSATION_API_STREAMING_URL, 
                                           data=json.dumps(payload), 
                                           headers=headers, 
                                           stream=True)
            response_stream.raise_for_status()
            
            return Response(stream_response(response_stream, request_data, request_data.get('model', DEFAULT_MODEL), prompt_token), 
                            mimetype='text/event-stream')
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return ERROR_HANDLER(1020, key="[REDACTED]")
            
            # If you get an error while streaming, try without streaming
            logger.warning(f"Streaming request failed for model {model}, trying non-streaming mode")
            try:
                response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
                response.raise_for_status()
                one_min_response = response.json()
                
                transformed_response = transform_response(one_min_response, request_data, prompt_token)
                flask_response = make_response(jsonify(transformed_response))
                set_response_headers(flask_response)
                
                return flask_response, 200
            except requests.exceptions.HTTPError as retry_e:
                if retry_e.response.status_code == 401:
                    return ERROR_HANDLER(1020, key="[REDACTED]")
                return ERROR_HANDLER(1500, detail=str(retry_e))
            
        except Exception as e:
            return ERROR_HANDLER(1500, detail=str(e))

def transform_streaming_response(data, request_data, last_output, prompt_tokens):
    """Transform 1minAI streaming response format to OpenAI streaming format"""
    current_time = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    model = map_model_to_openai(request_data.get('model', DEFAULT_MODEL))
    
    # Initialize the response structure
    transformed_response = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": current_time,
        "model": model,
        "choices": []
    }
    
    # Handle different response formats
    choices = []
    
    # Log the data structure for debugging
    logger.debug(f"Transform streaming response data keys: {list(data.keys())}")
    
    # Check for content field
    if 'content' in data:
        content = data.get('content', '')
        if content:  # Only add if content is not empty
            choice = {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None
            }
            choices.append(choice)
    
    # Check for function_call field
    elif 'function_call' in data:
        # Handle function call streaming
        func_call = data['function_call']
        if isinstance(func_call, dict):
            choice = {
                "index": 0,
                "delta": {
                    "function_call": {
                        "name": func_call.get('name', ''),
                        "arguments": func_call.get('arguments', '')
                    }
                },
                "finish_reason": None
            }
            choices.append(choice)
    
    # Check for stop signal
    elif 'stop' in data and data['stop']:
        # Handle the stop signal
        choice = {
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }
        choices.append(choice)
    
    # Handle other formats that may have a text/message field
    elif 'text' in data:
        choice = {
            "index": 0,
            "delta": {"content": data.get('text', '')},
            "finish_reason": None
        }
        choices.append(choice)
    
    elif 'message' in data:
        choice = {
            "index": 0,
            "delta": {"content": data.get('message', '')},
            "finish_reason": None
        }
        choices.append(choice)
    
    # Fallback case - try to find any text-like field in the data
    else:
        for key, value in data.items():
            if isinstance(value, str) and value:
                logger.debug(f"Using fallback field {key} for content")
                choice = {
                    "index": 0,
                    "delta": {"content": value},
                    "finish_reason": None
                }
                choices.append(choice)
                break
    
    # If we still have no choices, create an empty delta to keep the stream alive
    if not choices:
        choice = {
            "index": 0,
            "delta": {},
            "finish_reason": None
        }
        choices.append(choice)
    
    transformed_response["choices"] = choices
    return transformed_response

# Add a function to transform the response from 1minAI API into OpenAI API format
def transform_response(one_min_response, request_data, prompt_tokens):
    """Transform 1minAI response format to OpenAI format"""
    current_time = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    model = map_model_to_openai(request_data.get('model', DEFAULT_MODEL))
    
    # Initialize transformed response
    transformed_response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": current_time,
        "model": model,
        "choices": [],
        "usage": {}
    }
    
    # Extract content from 1minAI response
    content = one_min_response.get('content', '')
    
    # Handle function calling if present
    if 'function_call' in one_min_response:
        function_call = one_min_response['function_call']
        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": function_call.get('name', ''),
                    "arguments": function_call.get('arguments', '{}')
                }
            },
            "finish_reason": "function_call"
        }
    else:
        # Regular text response
        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }
    
    transformed_response["choices"].append(choice)
    
    # Calculate completion tokens
    completion_tokens = calculate_token(content, request_data.get('model', 'DEFAULT'))
    
    # Add usage information
    transformed_response["usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }
    
    return transformed_response

@app.route('/v1/embeddings', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def embeddings():
    """Handle embeddings requests"""
    if request.method == 'OPTIONS':
        return handle_options_request()

    try:
        # Get the request data
        request_data = request.get_json()
        
        # Extract API key from Authorization header
        api_key = extract_api_key()
        if not api_key:
            return ERROR_HANDLER(1001)
        
        # Validate the input data
        if not request_data.get('input'):
            return ERROR_HANDLER(1002, detail="Input is required")
        
        model = request_data.get('model', 'text-embedding-ada-002')
        input_text = request_data.get('input', '')
        
        # Prepare the payload for 1minAI API
        payload = {
            "text": input_text if isinstance(input_text, str) else input_text,
            "api_key": api_key
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Make the request to 1minAI embeddings API
        response = requests.post(ONE_MIN_EMBEDDINGS_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        one_min_response = response.json()
        
        # Transform the response to OpenAI format
        transformed_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": one_min_response.get('embedding', []),
                    "index": 0
                }
            ],
            "model": map_model_to_openai(model),
            "usage": {
                "prompt_tokens": calculate_token(input_text if isinstance(input_text, str) else json.dumps(input_text)),
                "total_tokens": calculate_token(input_text if isinstance(input_text, str) else json.dumps(input_text))
            }
        }
        
        # Return the response
        flask_response = make_response(jsonify(transformed_response))
        set_response_headers(flask_response)
        
        return flask_response, 200
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return ERROR_HANDLER(1020, key="[REDACTED]")
        return ERROR_HANDLER(1500, detail=str(e))
    except Exception as e:
        return ERROR_HANDLER(1500, detail=str(e))

@app.route('/v1/images/generations', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def images_generations():
    """Handle image generation requests"""
    if request.method == 'OPTIONS':
        return handle_options_request()

    try:
        # Get the request data
        request_data = request.get_json()
        
        # Extract API key from Authorization header
        api_key = extract_api_key()
        if not api_key:
            return ERROR_HANDLER(1001)
        
        # Validate the input data
        if not request_data.get('prompt'):
            return ERROR_HANDLER(1002, detail="Prompt is required")
        
        # Prepare the payload for 1minAI API
        payload = {
            "prompt": request_data.get('prompt'),
            "n": request_data.get('n', 1),
            "size": request_data.get('size', '1024x1024'),
            "api_key": api_key,
            "model": request_data.get('model', 'dall-e-3')
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Make the request to 1minAI image generation API
        response = requests.post(ONE_MIN_IMAGE_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        one_min_response = response.json()
        
        # Transform the response to OpenAI format
        transformed_response = {
            "created": int(time.time()),
            "data": []
        }
        
        # Process the images from the response
        if 'images' in one_min_response:
            for i, img_url in enumerate(one_min_response['images']):
                transformed_response['data'].append({
                    "url": img_url,
                    "revised_prompt": one_min_response.get('revised_prompt', request_data.get('prompt'))
                })
        
        # Return the response
        flask_response = make_response(jsonify(transformed_response))
        set_response_headers(flask_response)
        
        return flask_response, 200
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return ERROR_HANDLER(1020, key="[REDACTED]")
        return ERROR_HANDLER(1500, detail=str(e))
    except Exception as e:
        return ERROR_HANDLER(1500, detail=str(e))

@app.route('/v1/moderations', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def moderations():
    """Handle content moderation requests"""
    if request.method == 'OPTIONS':
        return handle_options_request()

    try:
        # Get the request data
        request_data = request.get_json()
        
        # Extract API key from Authorization header
        api_key = extract_api_key()
        if not api_key:
            return ERROR_HANDLER(1001)
        
        # Validate the input data
        if not request_data.get('input'):
            return ERROR_HANDLER(1002, detail="Input is required")
        
        # Prepare the payload for 1minAI API
        payload = {
            "text": request_data.get('input', ''),
            "api_key": api_key
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Make the request to 1minAI moderation API
        response = requests.post(ONE_MIN_MODERATION_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        one_min_response = response.json()
        
        # Transform the response to OpenAI format
        transformed_response = {
            "id": f"modr-{uuid.uuid4()}",
            "model": "text-moderation-latest",
            "results": []
        }
        
        # Process the moderation results
        if 'results' in one_min_response:
            for result in one_min_response['results']:
                transformed_response['results'].append({
                    "flagged": result.get('flagged', False),
                    "categories": result.get('categories', {}),
                    "category_scores": result.get('category_scores', {})
                })
        
        # Return the response
        flask_response = make_response(jsonify(transformed_response))
        set_response_headers(flask_response)
        
        return flask_response, 200
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return ERROR_HANDLER(1020, key="[REDACTED]")
        return ERROR_HANDLER(1500, detail=str(e))
    except Exception as e:
        return ERROR_HANDLER(1500, detail=str(e))

@app.route('/v1/assistants', methods=['GET', 'POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def assistants():
    """Handle assistants API requests"""
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    # Extract API key from Authorization header
    api_key = extract_api_key()
    if not api_key:
        return ERROR_HANDLER(1001)
    
    if request.method == 'GET':
        # List assistants
        return ERROR_HANDLER(1500, detail="Assistants API listing is not implemented in this version")
    
    elif request.method == 'POST':
        try:
            # Create a new assistant
            request_data = request.get_json()
            
            # Forward the request to 1minAI assistants API
            payload = {
                "api_key": api_key,
                "name": request_data.get('name'),
                "description": request_data.get('description'),
                "instructions": request_data.get('instructions'),
                "model": request_data.get('model', DEFAULT_MODEL),
                "tools": request_data.get('tools', [])
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(ONE_MIN_ASSISTANTS_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            
            one_min_response = response.json()
            
            # Transform to OpenAI format
            transformed_response = {
                "id": one_min_response.get('id', f"asst_{uuid.uuid4()}"),
                "object": "assistant",
                "created_at": int(time.time()),
                "name": one_min_response.get('name'),
                "description": one_min_response.get('description'),
                "instructions": one_min_response.get('instructions'),
                "model": map_model_to_openai(one_min_response.get('model')),
                "tools": one_min_response.get('tools', [])
            }
            
            flask_response = make_response(jsonify(transformed_response))
            set_response_headers(flask_response)
            
            return flask_response, 200
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return ERROR_HANDLER(1020, key="[REDACTED]")
            return ERROR_HANDLER(1500, detail=str(e))
        except Exception as e:
            return ERROR_HANDLER(1500, detail=str(e))

@app.route('/v1/files', methods=['GET', 'POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def files():
    """Handle file upload and retrieval"""
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    # Extract API key from Authorization header
    api_key = extract_api_key()
    if not api_key:
        return ERROR_HANDLER(1001)
    
    if request.method == 'GET':
        # List files
        try:
            params = {
                "api_key": api_key
            }
            
            response = requests.get(ONE_MIN_FILES_API_URL, params=params)
            response.raise_for_status()
            
            one_min_response = response.json()
            
            # Transform to OpenAI format
            transformed_response = {
                "object": "list",
                "data": []
            }
            
            if 'files' in one_min_response:
                for file in one_min_response['files']:
                    transformed_response['data'].append({
                        "id": file.get('id'),
                        "object": "file",
                        "bytes": file.get('size', 0),
                        "created_at": file.get('created_at', int(time.time())),
                        "filename": file.get('filename'),
                        "purpose": file.get('purpose', 'assistants')
                    })
            
            flask_response = make_response(jsonify(transformed_response))
            set_response_headers(flask_response)
            
            return flask_response, 200
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return ERROR_HANDLER(1020, key="[REDACTED]")
            return ERROR_HANDLER(1500, detail=str(e))
        except Exception as e:
            return ERROR_HANDLER(1500, detail=str(e))
    
    elif request.method == 'POST':
        # Upload a file
        try:
            # Check if file is present in request
            if 'file' not in request.files:
                return ERROR_HANDLER(1002, detail="File is required")
            
            file = request.files['file']
            purpose = request.form.get('purpose', 'assistants')
            
            # Check if filename is empty
            if file.filename == '':
                return ERROR_HANDLER(1700, detail="Empty filename")

            try:
                # Check file type - support PDF, TXT, MD, DOCX
                allowed_mime_types = ['application/pdf', 'text/plain', 'text/markdown', 
                                     'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                                     'application/msword']
                
                # Read file content
                file_content = file.read()
                file.seek(0)  # Reset file pointer
                
                # Determine MIME type
                mime_type = file.content_type
                file_ext = os.path.splitext(file.filename)[1].lower()
                
                # Additional checks for text files
                if not mime_type or mime_type == 'application/octet-stream':
                    if file_ext in ['.txt', '.md', '.markdown']:
                        mime_type = 'text/plain'
                    elif file_ext == '.pdf':
                        mime_type = 'application/pdf'
                    elif file_ext == '.docx':
                        mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    elif file_ext == '.doc':
                        mime_type = 'application/msword'
                
                if mime_type not in allowed_mime_types:
                    return ERROR_HANDLER(1700, detail=f"Unsupported file type: {mime_type}")
                
                # Process DOC/DOCX files
                if mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or mime_type == 'application/msword':
                    logger.debug(f"Processing DOCX/DOC file: {file.filename}")
                    
                    # Save temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                        temp_file_path = temp_file.name
                        temp_file.write(file_content)
                    
                    try:
                        # Extract text
                        if file_ext == '.docx':
                            # Use docx2txt for DOCX
                            text_content = docx2txt.process(temp_file_path)
                        else:
                            # Use python-docx for DOC (with conversion limitations)
                            try:
                                doc = DocxDocument(temp_file_path)
                                text_content = "\n".join([para.text for para in doc.paragraphs])
                            except Exception as doc_err:
                                logger.error(f"Error processing DOC file with python-docx: {str(doc_err)}")
                                text_content = "Error extracting text from DOC file. Please convert to DOCX or PDF format."
                        
                        # Create a text file with the extracted content
                        text_filename = os.path.splitext(file.filename)[0] + ".txt"
                        text_file_io = BytesIO(text_content.encode('utf-8'))
                        
                        # Upload the text file instead
                        logger.debug(f"Uploading converted text file: {text_filename}")
                        upload_result = upload_file_to_1min(text_file_io, text_filename, 'text/plain', api_key)
                    finally:
                        # Clean up
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                else:
                    # Process other file types directly
                    file_io = BytesIO(file_content)
                    upload_result = upload_file_to_1min(file_io, file.filename, mime_type, api_key)
                
                # Format response in OpenAI format
                transformed_response = {
                    "id": upload_result['id'],
                    "object": "file",
                    "bytes": len(file_content),
                    "created_at": int(time.time()),
                    "filename": upload_result['name'],
                    "purpose": purpose,
                    "status": "processed"
                }
                
                flask_response = make_response(jsonify(transformed_response))
                set_response_headers(flask_response)
                
                return flask_response, 200
            
            except Exception as file_error:
                logger.error(f"Error processing file: {str(file_error)}")
                return ERROR_HANDLER(1700, detail=f"Error processing file: {str(file_error)}")
            
            # Legacy code path (should not be reached)
            # Prepare multipart form data
            files = {
                'file': (file.filename, file.stream, file.content_type)
            }
            
            data = {
                'api_key': api_key,
                'purpose': purpose
            }
            
            # Make request to 1minAI file upload API
            response = requests.post(ONE_MIN_FILES_API_URL, files=files, data=data)
            response.raise_for_status()
            
            one_min_response = response.json()
            
            # Transform to OpenAI format
            transformed_response = {
                "id": one_min_response.get('id', f"file-{uuid.uuid4()}"),
                "object": "file",
                "bytes": one_min_response.get('size', 0),
                "created_at": one_min_response.get('created_at', int(time.time())),
                "filename": one_min_response.get('filename', file.filename),
                "purpose": purpose
            }
            
            flask_response = make_response(jsonify(transformed_response))
            set_response_headers(flask_response)
            
            return flask_response, 200
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return ERROR_HANDLER(1020, key="[REDACTED]")
            return ERROR_HANDLER(1500, detail=str(e))
        except Exception as e:
            return ERROR_HANDLER(1500, detail=str(e))

def stream_response(response, request_data, model, prompt_tokens):
    """
    Process streaming response from 1minAI API and convert it to OpenAI format
    """
    all_chunks = ""
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            chunk_text = chunk.decode('utf-8')
            all_chunks += chunk_text
            
            return_chunk = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": map_model_to_openai(request_data.get('model', DEFAULT_MODEL)),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk_text
                        },
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(return_chunk)}\n\n"
    
    # Calculate tokens from all chunks
    completion_tokens = calculate_token(all_chunks, request_data.get('model', 'DEFAULT'))
    logger.debug(f"Finished processing streaming response. Completion tokens: {str(completion_tokens)}")
    logger.debug(f"Total tokens: {str(completion_tokens + prompt_tokens)}")
    
    # Final chunk when iteration stops
    final_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": map_model_to_openai(request_data.get('model', DEFAULT_MODEL)),
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

def upload_file_to_1min(file_data, file_name, mime_type, api_key):
    """
    Upload a file to 1minAI via Asset API
    
    Args:
        file_data (BytesIO): File data
        file_name (str): File name
        mime_type (str): File MIME type
        api_key (str): API key
    
    Returns:
        dict: Upload result with file ID and path
    """
    try:
        headers = {"API-KEY": api_key}
        files = {
            'asset': (file_name, file_data, mime_type)
        }
        
        logger.debug(f"Uploading file {file_name} to 1minAI")
        response = requests.post(ONE_MIN_ASSET_URL, files=files, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        logger.debug(f"File successfully uploaded, ID: {result['fileContent']['uuid']}")
        
        return {
            "id": result['fileContent']['uuid'],
            "path": result['fileContent']['path'],
            "type": result['fileContent']['type'],
            "name": result['fileContent']['name']
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise

def execute_python_code(code, timeout=10):
    """
    Execute Python code in a safe environment with timeout
    
    Args:
        code (str): Python code to execute
        timeout (int): Maximum execution time in seconds
    
    Returns:
        dict: Execution result with stdout, stderr and execution status
    """
    try:
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(code)
        
        # Execute the code in a separate process with timeout
        logger.debug(f"Executing Python code with {timeout}s timeout")
        process = subprocess.Popen(
            ['python', temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return {
                "status": "success" if process.returncode == 0 else "error",
                "stdout": stdout,
                "stderr": stderr,
                "return_code": process.returncode
            }
        except subprocess.TimeoutExpired:
            process.kill()
            return {
                "status": "timeout",
                "stdout": "",
                "stderr": f"Execution timed out after {timeout} seconds",
                "return_code": -1
            }
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        logger.error(f"Error executing Python code: {str(e)}")
        return {
            "status": "error",
            "stdout": "",
            "stderr": str(e),
            "return_code": -1
        }

def create_conversation_with_files(api_key, title, model, file_ids):
    """
    Create a conversation with files via Conversation API
    
    Args:
        api_key (str): API key
        title (str): Conversation title
        model (str): Model name
        file_ids (list): List of file IDs
        
    Returns:
        str: Created conversation ID
    """
    try:
        headers = {
            "API-KEY": api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "title": title[:90],  # Limit to 90 characters
            "type": "CHAT_WITH_PDF",
            "model": model,
            "fileList": file_ids
        }
        
        logger.debug(f"Creating conversation with files: {file_ids}")
        response = requests.post(ONE_MIN_CONVERSATION_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        conversation_id = response.json()['conversation']['uuid']
        logger.debug(f"Conversation successfully created, ID: {conversation_id}")
        
        return conversation_id
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise

# The main function to start the server
if __name__ == "__main__":
    # Set the logging level
    if os.environ.get('DEBUG') == 'True':
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Start the server
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
