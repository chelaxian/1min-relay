import base64
import datetime
import hashlib
import io
import json
import logging
import os
import re
import socket
import sys
import tempfile
import time
import traceback
import uuid
import warnings
from io import BytesIO

import coloredlogs
import requests
import tiktoken
from flask import Flask, request, jsonify, make_response, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from pymemcache.client.base import Client

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

model = 'gpt-4o-mini'


# noinspection PyBroadException
def check_memcached_connection(host='memcached', port=11211):
    # noinspection PyBroadException
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


# noinspection PyShadowingNames
def calculate_token(sentence, model="DEFAULT"):
    """Calculate the number of tokens in a sentence based on the specified model."""

    if model.startswith("mistral"):
        # Initialize the Mistral tokenizer
        MistralTokenizer.v3(is_tekken=True)
        model_name = "open-mistral-nemo"  # Default to Mistral Nemo
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
    # STT
    # "whisper-1",
    # TTS
    # "alloy",
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
    "gpt-4-turbo"
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
tts_supported_models = [
    "alloy"
]

stt_supported_models = [
    "whisper-1"
]

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
DEFAULT_MODEL = "gpt-4o-mini"


# noinspection PyShadowingNames
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


# noinspection PyShadowingNames
def error_handler(code, model=None, key=None, detail=None):
    # Handle errors in OpenAI-Structured Error
    error_codes = {  # Internal Error Codes
        1002: {"message": f"The model {model} does not exist.", "type": "invalid_request_error", "param": None,
               "code": "model_not_found", "http_code": 400},
        1020: {"message": f"Incorrect API key provided: {key}. You can find your API key at https://app.1min.ai/api.",
               "type": "authentication_error", "param": None, "code": "invalid_api_key", "http_code": 401},
        1021: {"message": "Invalid Authentication", "type": "invalid_request_error", "param": None, "code": None,
               "http_code": 401},
        1212: {"message": f"Incorrect Endpoint. Please use the /v1/chat/completions endpoint.",
               "type": "invalid_request_error", "param": None, "code": "model_not_supported", "http_code": 400},
        1044: {"message": f"This model does not support image inputs.", "type": "invalid_request_error", "param": None,
               "code": "model_not_supported", "http_code": 400},
        1045: {"message": f"This model does not support tool use.", "type": "invalid_request_error", "param": None,
               "code": "model_not_supported", "http_code": 400},
        1046: {"message": f"This model does not support text-to-speech.", "type": "invalid_request_error",
               "param": None, "code": "model_not_supported", "http_code": 400},
        1412: {"message": f"No message provided.", "type": "invalid_request_error", "param": "messages",
               "code": "invalid_request_error", "http_code": 400},
        1423: {"message": f"No content in last message.", "type": "invalid_request_error", "param": "messages",
               "code": "invalid_request_error", "http_code": 400},
        1500: {"message": f"1minAI API error: {detail}", "type": "api_error", "param": None, "code": "api_error",
               "http_code": 500},
        1600: {"message": f"Unsupported feature: {detail}", "type": "invalid_request_error", "param": None,
               "code": "unsupported_feature", "http_code": 400},
        1700: {"message": f"Invalid file format: {detail}", "type": "invalid_request_error", "param": None,
               "code": "invalid_file_format", "http_code": 400},
    }
    error_data = {k: v for k, v in error_codes.get(code, {
        "message": f"Unknown error: {detail}" if detail else "Unknown error", "type": "unknown_error", "param": None,
        "code": None}).items() if k != "http_code"}  # Remove http_code from the error data
    logger.error(f"An error has occurred while processing the user's request. Error code: {code}")
    return jsonify({"error": error_data}), error_codes.get(code, {}).get("http_code",
                                                                         400)  # Return the error data without
    # http_code inside the payload and get the http_code to return.


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
        return error_handler(1212)
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

    # Add STT models
    # stt_models_data = [
    #     {"id": model_name, "object": "model", "owned_by": "1minai", "created": 1727389042}
    #     for model_name in stt_supported_models
    # ]
    # models_data.extend(stt_models_data)

    return jsonify({"data": models_data, "object": "list"})


# noinspection DuplicatedCode
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
    if messages:  # Save credits if it is the first message.
        formatted_history.append(
            "Respond like normal. The conversation history will be automatically updated on the next MESSAGE. DO NOT "
            "ADD User: or Assistant: to your output. Just respond like normal.")
        formatted_history.append("User Message:\n")
    formatted_history.append(new_input)

    return '\n'.join(formatted_history)


# noinspection PyShadowingNames
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
                        "high"
                    else:
                        "auto"

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
                    if 'claude' in model and not any(text in item.get('text', '') for text in
                                                     ["describe", "what do you see", "analyze", "explain"]):
                        # Add image analysis instructions for Claude models to improve recognition of people
                        analysis_prompts = [
                            "Describe this image in detail. If there are people in the image, describe them. If there "
                            "is text in the image, read it.",
                            "What do you see in this image? Please describe all elements, including any people, "
                            "objects, text, and scenery."
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


# noinspection PyShadowingNames
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

    # ВАЖНО: Не включаем ссылки на функции в возвращаемый объект
    return {
        "tools": one_min_tools,
        "autoInvoke": auto_invoke
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


# noinspection PyShadowingNames
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
        return error_handler(1021)

    api_key = auth_header.split(" ")[1]

    # Extract parameters
    input_text = request_data.get('input', '')
    model = request_data.get('model', 'nova')
    voice = request_data.get('voice', 'alloy')
    response_format = request_data.get('response_format', 'mp3')
    speed = request_data.get('speed', 1.0)

    if not input_text:
        return error_handler(1412, detail="No input text provided for TTS")

    if model not in tts_supported_models:
        return error_handler(1046, model=model)

    # Prepare request to 1minAI API using unified endpoint
    headers = {
        "API-KEY": api_key,
        "Content-Type": "application/json"
    }

    # Использование единого эндпоинта с типом TEXT_TO_SPEECH
    payload = {
        "type": "TEXT_TO_SPEECH",
        "model": model,
        "promptObject": {
            "text": input_text,
            "voice": voice,
            "speed": speed,
            "response_format": response_format
        }
    }

    try:
        logger.info(f"Sending TTS request to: {ONE_MIN_API_URL}")
        logger.debug(f"TTS payload: {json.dumps(payload)}")

        response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
        response.raise_for_status()

        # Get the audio data
        tts_response = response.json()
        audio_url = tts_response.get('audioUrl')

        if not audio_url:
            return error_handler(1500, detail="No audio URL returned from 1minAI TTS API")

        # Download the audio file
        audio_response = requests.get(audio_url)
        audio_response.raise_for_status()

        # Create response
        flask_response = make_response(audio_response.content)
        flask_response.headers['Content-Type'] = f'audio/{response_format}'

        return flask_response
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in TTS processing: {str(e)}")
        return error_handler(1500, detail=str(e))


def process_stt_request():
    """
    Process speech-to-text request

    Returns:
        Response: Flask response with transcription or error
    """
    global model
    logger.info("Processing speech-to-text request")

    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error("Missing or invalid authorization header")
        return error_handler(1021)

    api_key = auth_header.split(" ")[1]

    # Check if file is uploaded
    if 'file' not in request.files:
        logger.error("No audio file in request")
        return error_handler(1700, detail="No audio file provided")

    if model not in stt_supported_models:
        return error_handler(1046, model=model)

    audio_file = request.files['file']

    # Get the originally requested model (which will process the transcribed text)
    original_model = request.form.get('model', 'gpt-4o-mini')
    logger.info(f"Original model requested: {original_model}")

    # Для STT всегда используем whisper-1
    model = "whisper-1"
    logger.info(f"Processing audio file: {audio_file.filename} with {model} for transcription")

    # Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        audio_file.save(temp_file.name)
        temp_file_path = temp_file.name
        logger.debug(f"Saved audio file temporarily at: {temp_file_path}")

    try:
        # Upload to 1minAI
        headers = {
            "API-KEY": api_key,
        }
        files = {
            'asset': (audio_file.filename, open(temp_file_path, 'rb'), audio_file.content_type)
        }

        logger.info(f"Uploading audio file to 1minAI asset URL: {ONE_MIN_ASSET_URL}")
        asset_response = requests.post(ONE_MIN_ASSET_URL, files=files, headers=headers)
        asset_response.raise_for_status()

        # Get audio path
        asset_data = asset_response.json()
        logger.debug(f"Asset response: {json.dumps(asset_data)[:200]}...")
        audio_path = asset_data['fileContent']['path']
        logger.info(f"Audio file uploaded successfully. Path: {audio_path}")

        # Отправка запроса на транскрипцию в 1minAI
        features_url = "https://api.1min.ai/api/features"
        payload = {
            "type": "SPEECH_TO_TEXT",
            "model": model,
            "promptObject": {
                "audioUrl": audio_path,
                "response_format": "text"
            }
        }

        logger.info(f"Sending transcription request to: {features_url}")
        logger.debug(f"Transcription payload: {payload}")

        # Используем тот же формат заголовка API-KEY
        headers = {'API-KEY': api_key}
        transcription_response = requests.post(
            features_url,
            json=payload,
            headers=headers
        )

        # Проверка статуса ответа
        logger.debug(f"Speech-to-text response status: {transcription_response.status_code}")
        logger.debug(f"Speech-to-text response headers: {transcription_response.headers}")
        logger.debug(f"Speech-to-text response body: {transcription_response.text}")

        if transcription_response.status_code != 200:
            logger.error(f"Error from 1minAI API: {transcription_response.text}")
            return jsonify({"error": f"Error from 1minAI API: {transcription_response.text}"}), 500

        # Обработка ответа от 1minAI
        ""
        response_data = transcription_response.json()

        # Извлечение текста из ответа от 1minAI (формат может различаться)
        if 'aiRecord' in response_data and 'aiRecordDetail' in response_data['aiRecord']:
            details = response_data['aiRecord']['aiRecordDetail']
            if 'resultObject' in details and isinstance(details['resultObject'], list) and details['resultObject']:
                transcript = "".join(details['resultObject'])
                logger.info(f"Transcription successful: {transcript}")
            else:
                logger.error("Invalid response format from 1minAI API")
                return jsonify({"error": "Invalid response format from 1minAI API"}), 500
        else:
            logger.error("Invalid response format from 1minAI API")
            return jsonify({"error": "Invalid response format from 1minAI API"}), 500

        # Если требуется также получить ответ на транскрибированный текст
        response_model = request.form.get('response_model', None)
        if response_model:
            logger.info(f"Forwarding transcribed text to original model: {response_model}")

            # Отправляем запрос к модели для генерации ответа
            try:
                # Устанавливаем API URL для запроса
                api_url = "https://api.1min.ai/api/features"

                # Формируем данные запроса
                payload = {
                    "type": "CHAT_WITH_AI",
                    "model": response_model,
                    "promptObject": {
                        "prompt": transcript,
                        "streaming": False
                    }
                }

                logger.debug(f"Sending request to {api_url} with payload: {payload}")

                # Отправляем запрос с правильным заголовком API-KEY
                headers = {'API-KEY': api_key}
                response = requests.post(
                    api_url,
                    json=payload,
                    headers=headers
                )

                # Обрабатываем ответ
                if response.status_code == 200:
                    response_data = response.json()
                    logger.debug(f"Received response: {response_data}")

                    # Извлекаем содержимое из ответа
                    content = ""
                    if 'aiRecord' in response_data and 'aiRecordDetail' in response_data['aiRecord']:
                        details = response_data['aiRecord']['aiRecordDetail']
                        if 'resultObject' in details:
                            result_obj = details['resultObject']
                            if isinstance(result_obj, dict) and 'content' in result_obj:
                                content = result_obj['content']
                            elif isinstance(result_obj, str):
                                content = result_obj
                            elif isinstance(result_obj, list) and len(result_obj) > 0:
                                content = "".join([str(item) for item in result_obj])

                    logger.info(f"Extracted content from response: {content[:100]}...")

                    # Подготавливаем ответ в формате OpenAI API
                    openai_format_response = {
                        "id": f"chatcmpl-{str(uuid.uuid4())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": response_model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": content
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(transcript.split()),
                            "completion_tokens": len(content.split()),
                            "total_tokens": len(transcript.split()) + len(content.split())
                        }
                    }

                    logger.debug(f"Successfully processed audio response with {len(content.split())} tokens")
                    logger.debug(f"Returning OpenAI format response: {openai_format_response}")

                    # Здесь сохраняем полный ответ для отладки
                    with open('/tmp/last_voice_response.json', 'w', encoding='utf-8') as f:
                        json.dump(openai_format_response, f, ensure_ascii=False, indent=2)

                    # Удаление временного файла
                    try:
                        os.remove(temp_file_path)
                        logger.debug(f"Removed temporary file: {temp_file_path}")
                    except Exception as e:
                        logger.error(f"Error removing temporary file: {str(e)}")

                    return jsonify(openai_format_response)
                else:
                    logger.error(f"Error response from AI API: {response.status_code} - {response.text}")
                    return jsonify({
                        "error": f"Error response from AI API: {response.status_code}"
                    }), 500

            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return jsonify({
                    "error": f"Error generating response: {str(e)}"
                }), 500

        # Формирование ответа в формате OpenAI API
        openai_format_response = {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": transcript
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(transcript.split()),
                "total_tokens": len(transcript.split())
            }
        }

        # Удаление временного файла
        try:
            os.remove(temp_file_path)
            logger.debug(f"Removed temporary file: {temp_file_path}")
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")

        return jsonify(openai_format_response)

    except Exception as e:
        logger.error(f"Error processing audio transcription: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Error processing audio transcription: {str(e)}"}), 500


@app.route('/v1/audio/transcriptions', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def audio_transcriptions():
    """Endpoint для транскрипции аудио в текст с использованием модели Whisper-1.
    Также поддерживает перенаправление текста на выбранную модель для генерации ответа.
    """
    if request.method == 'OPTIONS':
        return handle_options_request()

    return process_stt_request()


@app.route('/v1/audio/speech', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def audio_speech():
    if request.method == 'OPTIONS':
        return handle_options_request()
    request_data = request.json
    return process_tts_request(request_data)
    # Temporarily return an error that the function is disabled
    # return error_handler(1600, detail="TTS functionality is temporarily disabled")


# noinspection PyShadowingNames,DuplicatedCode
@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def conversation():
    if request.method == 'OPTIONS':
        return handle_options_request()

    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error("Invalid Authentication")
        return error_handler(1021)

    api_key = auth_header.split(" ")[1]
    request_data = request.json

    headers = {
        "API-KEY": api_key,
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if request_data.get('stream', False) else "application/json"
    }

    # Get model
    model = request_data.get('model', 'gpt-4o-mini')

    # Check to see if the TTS model is a model
    if model in tts_supported_models:
        return error_handler(1600, detail="TTS models can only be used with the /v1/audio/speech endpoint")

    # Check to see if the STT model is a model
    if model in stt_supported_models:
        return error_handler(1600, detail="STT models can only be used with the /v1/audio/transcriptions endpoint")

    if PERMIT_MODELS_FROM_SUBSET_ONLY and model not in AVAILABLE_MODELS:
        return error_handler(1002, model)

    # Get messages
    messages = request_data.get('messages', [])
    if not messages:
        return error_handler(1412)

    # Add system info to last user message if it was a converted DOC/DOCX file
    last_message = messages[-1]
    if last_message.get('role') == 'user' and 'doc_file_conversion' in request.headers:
        original_filename = request.headers.get('doc_file_conversion')
        # Only modify content if it's a string
        if isinstance(last_message.get('content'), str):
            logger.debug(f"Adding note about DOC/DOCX conversion for file: {original_filename}")
            last_message[
                'content'] += f"\n\n(Примечание: Этот текст был извлечен из файла {original_filename}. Некоторое " \
                              f"форматирование могло быть потеряно при конвертации.)"

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
            return error_handler(1423)

    # Format conversation history
    all_messages = format_conversation_history(messages, user_input, system_prompt)
    prompt_token = calculate_token(str(all_messages))

    # Process tools (function calling)
    tools_config = None
    try:
        # Определяем базовый набор инструментов
        base_tools = request_data.get('tools', [])

        # Для поддерживаемых моделей автоматически добавляем стандартные инструменты
        if model in tools_supported_models:
            # Добавляем стандартные инструменты, если они не указаны явно
            standard_tools = [
                {"type": "function", "function": {"name": "get_datetime", "description": "Get current date and time"}},
                {"type": "function", "function": {"name": "execute_python", "description": "Execute Python code"}},
                {"type": "function", "function": {"name": "web_search", "description": "Search the web"}}
            ]

            # Если инструменты не заданы, используем стандартные
            if not base_tools:
                base_tools = standard_tools
            # Иначе добавляем стандартные, если их еще нет
            else:
                existing_tool_names = [t.get('function', {}).get('name', '') for t in base_tools if
                                       t.get('type') == 'function']
                for std_tool in standard_tools:
                    if std_tool['function']['name'] not in existing_tool_names:
                        base_tools.append(std_tool)

            tools_config = process_tools(
                base_tools,
                request_data.get('tool_choice', 'auto'),
                model
            )
            logger.info(
                f"Added tools configuration for supported model {model}: {[t.get('function', {}).get('name', '') for t in base_tools if t.get('type') == 'function']}")
        else:
            # Для неподдерживаемых моделей пробуем обработать только явно указанные инструменты
            if base_tools:
                try:
                    tools_config = process_tools(
                        base_tools,
                        request_data.get('tool_choice', 'auto'),
                        model
                    )
                    logger.warning(f"Processing tools for potentially unsupported model {model}")
                except Exception as tool_e:
                    logger.warning(f"Failed to process tools for model {model}: {str(tool_e)}")
                    # Продолжаем без инструментов
                    pass
    except Exception as e:
        logger.error(f"Error processing tools: {str(e)}")
        # Продолжаем без инструментов при ошибке
        pass

    # Check for web search
    use_web_search = request_data.get('web_search', False)
    # Автоматически включаем веб-поиск для поддерживаемых моделей, если в запросе есть ключевые слова
    if model in web_search_supported_models and not use_web_search:
        # Проверяем наличие ключевых слов для поиска в интернете
        search_keywords = ['найди', 'поищи', 'search', 'найти', 'поиск', 'погугли', 'загугли', 'ищи', 'интернете',
                           'internet', 'интернета', 'искать', 'интернет', 'интернетом', 'google', 'browse', 'find',
                           'узнай', 'почитай', 'прочитай', 'уточни', 'проверь', 'check', 'онлайн', 'online', 'confirm']
        content = messages[-1].get('content', '')

        # Проверка типа контента и обработка соответствующим образом
        if isinstance(content, list):
            # Для мультимодального контента извлекаем текст
            text_parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
            user_message = ' '.join(text_parts).lower()
        else:
            # Для обычного текстового контента
            user_message = content.lower()

        if any(keyword in user_message for keyword in search_keywords):
            use_web_search = True
            logger.info(f"Automatically enabling web search for model {model} based on user message")

    if use_web_search and model not in web_search_supported_models:
        logger.warning(f"Model {model} does not support web search, ignoring web search parameter")
        use_web_search = False

    # Prepare base payload
    payload = {
        "model": model,
        "promptObject": {
            "prompt": all_messages,
            "isMixed": False,
            "webSearch": use_web_search  # ,
            # "webSearch": true,
            # "numOfSite": 1,
            # "maxWord": 500
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
                return error_handler(1020, key="[REDACTED]")
            return error_handler(1500, detail=str(e))

    else:
        # Streaming Response
        logger.debug("Streaming AI Response")
        try:
            response_stream = requests.post(ONE_MIN_CONVERSATION_API_STREAMING_URL,
                                            data=json.dumps(payload),
                                            headers=headers,
                                            stream=True)
            response_stream.raise_for_status()

            return Response(
                stream_response(response_stream, request_data, prompt_token),
                mimetype='text/event-stream')
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return error_handler(1020, key="[REDACTED]")

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
                    return error_handler(1020, key="[REDACTED]")
                return error_handler(1500, detail=str(retry_e))

        except Exception as e:
            return error_handler(1500, detail=str(e))


# noinspection PyShadowingNames,PyTypedDict
def transform_streaming_chunk(data, model):
    """Transform a streaming chunk from 1minAI to OpenAI format"""
    # Generate a consistent ID for the completion
    completion_id = f"chatcmpl-{uuid.uuid4()}"

    # Create the initial response structure
    response = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": map_model_to_openai(model),
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant"
                } if not data else {"content": data.get('content', '')} if 'content' in data else {},
                "finish_reason": None
            }
        ]
    }

    # If this is an end of stream marker
    if data and data.get('stop', False):
        response["choices"][0]["finish_reason"] = stop

    return response


# noinspection PyShadowingNames
def transform_streaming_response(data, request_data):
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
    elif 'function_call' in data or 'tool_calls' in data:
        # Handle function call streaming
        if 'function_call' in data:
            func_call = data['function_call']
            # Generate a stable ID for the function call
            call_id = f"call_{hashlib.md5(json.dumps(func_call).encode()).hexdigest()}"
            choice = {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": func_call.get('name', ''),
                                "arguments": func_call.get('arguments', '')
                            }
                        }
                    ]
                },
                "finish_reason": None
            }
            choices.append(choice)

            # If we have a result, add it as a separate choice
            if 'result' in data:
                result_choice = {
                    "index": 1,
                    "delta": {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_call.get('name', ''),
                        "content": json.dumps(data['result'])
                    },
                    "finish_reason": "tool_result"
                }
                choices.append(result_choice)
        else:
            # Handle tool_calls format
            tool_calls = data.get('tool_calls', [])
            if tool_calls:
                for i, tool_call in enumerate(tool_calls):
                    # Generate a stable ID for each tool call
                    call_id = f"call_{hashlib.md5(json.dumps(tool_call).encode()).hexdigest()}"
                    choice = {
                        "index": i,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": i,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.get('function', {}).get('name', ''),
                                        "arguments": tool_call.get('function', {}).get('arguments', '')
                                    }
                                }
                            ]
                        },
                        "finish_reason": None
                    }
                    choices.append(choice)

                    # If we have a result for this tool call, add it
                    if 'result' in tool_call:
                        result_choice = {
                            "index": len(choices),
                            "delta": {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": tool_call.get('function', {}).get('name', ''),
                                "content": json.dumps(tool_call['result'])
                            },
                            "finish_reason": "tool_result"
                        }
                        choices.append(result_choice)

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
# noinspection PyShadowingNames
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

    # Handle tool calls if present
    if 'function_call' in one_min_response or 'tool_calls' in one_min_response:

        # Convert old function_call format to new tool_calls
        if 'function_call' in one_min_response:
            tool_calls = [{
                'name': one_min_response['function_call'].get('name', ''),
                'arguments': one_min_response['function_call'].get('arguments', '{}'),
                'id': f"call_{uuid.uuid4()}"
            }]
        else:
            tool_calls = one_min_response.get('tool_calls', [])
            for call in tool_calls:
                if 'id' not in call:
                    call['id'] = f"call_{uuid.uuid4()}"

        # Process each tool call
        processed_calls = []
        for tool_call in tool_calls:
            function_name = tool_call.get('name', '')
            function_args = tool_call.get('arguments', '{}')
            call_id = tool_call.get('id')

            # Prepare arguments
            try:
                args = json.loads(function_args) if isinstance(function_args, str) else function_args
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in arguments for {function_name}: {function_args}")
                args = {}

            # Process with appropriate handler

            try:
                if function_name == 'get_datetime':
                    result = handle_get_datetime(args)
                elif function_name == 'execute_python':
                    result = handle_execute_python(args)
                elif function_name == 'web_search':
                    result = handle_web_search(args)
                else:
                    logger.warning(f"Unknown function {function_name}")
                    result = {"error": f"Unknown function {function_name}"}
            except Exception as e:
                logger.error(f"Error handling function {function_name}: {str(e)}")
                result = {"error": str(e)}

            processed_calls.append({
                "call": {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(args)
                    }
                },
                "result": result
            })

        # Create response with tool calls
        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [pc["call"] for pc in processed_calls]
            },
            "finish_reason": "tool_calls"
        }
        transformed_response["choices"].append(choice)

        # Add tool call results
        for i, pc in enumerate(processed_calls):
            if pc["result"]:
                result_choice = {
                    "index": i + 1,
                    "message": {
                        "role": "tool",
                        "tool_call_id": pc["call"]["id"],
                        "name": pc["call"]["function"]["name"],
                        "content": json.dumps(pc["result"])
                    },
                    "finish_reason": "tool_result"
                }
                transformed_response["choices"].append(result_choice)
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
            return error_handler(1001)

        # Validate the input data
        if not request_data.get('prompt'):
            return error_handler(1002, detail="Prompt is required")

        # Prepare the payload for 1minAI API
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": request_data.get('model', 'dall-e-3'),
            "promptObject": {
                "prompt": request_data.get('prompt'),
                "n": request_data.get('n', 1),
                "size": request_data.get('size', '1024x1024')
            }
        }

        headers = {
            "API-KEY": api_key,
            "Content-Type": "application/json"
        }

        logger.info(f"Sending image generation request to: {ONE_MIN_API_URL}")
        logger.debug(f"Image generation payload: {json.dumps(payload)[:200]}...")

        # Make the request to 1minAI unified API endpoint
        response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
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
            return error_handler(1020, key="[REDACTED]")
        return error_handler(1500, detail=str(e))
    except Exception as e:
        return error_handler(1500, detail=str(e))


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

        logger.debug(f"Uploading file {file_name} to 1minAI Asset API: {ONE_MIN_ASSET_URL}")
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
        logger.info(f"Executing Python code: {code[:100]}{'...' if len(code) > 100 else ''}")

        # Create a string buffer to capture output
        output = io.StringIO()
        error = io.StringIO()

        # Redirect stdout and stderr
        sys.stdout = output
        sys.stderr = error

        # Create safe globals
        safe_globals = {
            '__builtins__': {
                name: getattr(__builtins__, name)
                for name in ['abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
                             'chr', 'complex', 'dict', 'dir', 'divmod', 'enumerate', 'filter',
                             'float', 'format', 'frozenset', 'hash', 'hex', 'int', 'isinstance',
                             'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min', 'next',
                             'object', 'oct', 'ord', 'pow', 'print', 'range', 'repr', 'reversed',
                             'round', 'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'type',
                             'zip']
            }
        }

        # Add some safe modules
        for module_name in ['math', 'random', 'datetime', 're', 'json', 'collections', 'mpmath']:
            try:
                safe_globals[module_name] = __import__(module_name)
            except ImportError:
                pass

        # Execute the code
        try:
            # Create a thread to execute the code with timeout
            def exec_code():
                exec(code, safe_globals, {})

            import threading
            thread = threading.Thread(target=exec_code)
            thread.daemon = True

            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                # Timeout occurred
                logger.warning(f"Python code execution timed out after {timeout} seconds")
                return {
                    "status": "timeout",
                    "stdout": output.getvalue(),
                    "stderr": f"Execution timed out after {timeout} seconds",
                    "return_code": -1
                }

            result = {
                "status": "success",
                "stdout": output.getvalue(),
                "stderr": error.getvalue(),
                "return_code": 0
            }
            logger.info(
                f"Python code executed successfully: {result['stdout'][:100]}{'...' if len(result['stdout']) > 100 else ''}")
        except Exception as e:
            result = {
                "status": "error",
                "stdout": output.getvalue(),
                "stderr": str(e),
                "return_code": -1
            }

        # Restore stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        return result

    except Exception as e:
        logger.error(f"Error executing Python code: {str(e)}")
        return {
            "status": "error",
            "stdout": "",
            "stderr": str(e),
            "return_code": -1
        }


# noinspection PyShadowingNames
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


def stream_response(response, request_data, prompt_tokens):
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


def save_audio_file(file):
    """
    Сохраняет аудиофайл во временной директории.

    Args:
        file: Объект файла из request.files

    Returns:
        str: Путь к сохраненному файлу
    """
    import uuid
    temp_dir = tempfile.gettempdir()
    # Создаем уникальное имя файла без вложенных директорий
    filename = f"audio_{uuid.uuid4().hex}.mp3"
    temp_file_path = os.path.join(temp_dir, filename)
    file.save(temp_file_path)
    logger.debug(f"Saved audio file temporarily at: {temp_file_path}")
    return temp_file_path


# The main function to start the server
if __name__ == "__main__":
    # Set the logging level
    if os.environ.get('DEBUG') == 'True':
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Start the server
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
