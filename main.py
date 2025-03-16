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
import io
import sys
from itertools import islice
import hashlib
from flask_cors import CORS
import ast
import traceback
import multiprocessing
import signal
from contextlib import contextmanager
from functools import wraps
import copy

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

# Включаем CORS для всех доменов
CORS(app)

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
        1400: {"message": f"Invalid request format: {detail}", "type": "invalid_request_error", "param": None, "code": "invalid_request_format", "http_code": 400},
    }
    error_data = {k: v for k, v in error_codes.get(code, {"message": f"Unknown error: {detail}" if detail else "Unknown error", "type": "unknown_error", "param": None, "code": None}).items() if k != "http_code"} # Remove http_code from the error data
    logger.error(f"An error has occurred while processing the user's request. Error code: {code}")
    return jsonify({"error": error_data}), error_codes.get(code, {}).get("http_code", 400) # Return the error data without http_code inside the payload and get the http_code to return.

def set_response_headers(response):
    """Set CORS and other headers for the response"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

def handle_options_request():
    """Handle OPTIONS request for CORS preflight"""
    response = make_response()
    set_response_headers(response)
    return response, 204

def extract_api_key():
    """Extract API key from Authorization header"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    return auth_header.split(" ")[1]

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

def process_tools(request_data):
    """Process tools configuration for the request"""
    model = request_data.get('model')
    logger.debug(f"Processing tools for model: {model}")
    
    # Извлечем последнее сообщение пользователя для анализа
    user_message = ""
    messages = request_data.get('messages', [])
    if messages and messages[-1].get('role') == 'user':
        user_content = messages[-1].get('content', '')
        if isinstance(user_content, list):
            # Для мультимодальных запросов извлекаем текст
            user_message = ' '.join([item.get('text', '') for item in user_content if item.get('type') == 'text'])
        else:
            user_message = user_content
    
    # Проверяем явные запросы на инструменты в сообщении
    python_keywords = ['python', 'код', 'code', 'выполни', 'запусти', 'run', 'execute']
    search_keywords = ['поиск', 'найди', 'search', 'web', 'интернет', 'погода', 'weather']
    
    contains_python_request = any(kw.lower() in user_message.lower() for kw in python_keywords)
    contains_search_request = any(kw.lower() in user_message.lower() for kw in search_keywords)
    
    # Логирование запросов на инструменты
    if contains_python_request:
        logger.info(f"Detected Python code execution request in user message")
    if contains_search_request:
        logger.info(f"Detected web search request in user message")
    
    # Списки моделей, которые поддерживают вызов инструментов
    tool_supported_models = ["gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4-turbo-preview", 
                          "gpt-4o", "gpt-4o-mini", "claude-3-opus-20240229", "claude-3-sonnet-20240229", 
                          "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"]
    
    # Проверяем, есть ли явные настройки инструментов в запросе
    tools = request_data.get('tools')
    tool_choice = request_data.get('tool_choice')
    
    # Добавление инструментов на основе ключевых слов в сообщении
    if not tools and (contains_python_request or contains_search_request):
        tools = []
        
        if contains_python_request:
            logger.info("Automatically enabling Python code execution based on user message")
            tools.append({
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Executes Python code and returns the output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to execute"
                            }
                        },
                        "required": ["code"]
                    }
                }
            })
        
        if contains_search_request:
            logger.info("Automatically enabling web search based on user message")
            tools.append({
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for real-time information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_term": {
                                "type": "string",
                                "description": "The search term to look up on the web"
                            }
                        },
                        "required": ["search_term"]
                    }
                }
            })
    
    # Если инструменты не указаны, но модель поддерживает их, добавим стандартные инструменты
    if not tools and model in tool_supported_models:
        logger.debug(f"Adding default tools for supported model {model}")
        tools = [
            {
                "type": "function", 
                "function": {
                    "name": "get_current_datetime",
                    "description": "Get the current date and time",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Executes Python code and returns the output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to execute"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for real-time information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_term": {
                                "type": "string",
                                "description": "The search term to look up on the web"
                            }
                        },
                        "required": ["search_term"]
                    }
                }
            }
        ]
    
    # Если инструменты не указаны и модель не поддерживает их, вернем None
    if not tools:
        logger.debug(f"No tools configured for model {model}")
        return None, None, False
    
    # Проверка и преобразование инструментов
    tools_config = []
    for tool in tools:
        # Если инструмент в формате OpenAI, преобразуем его в формат 1minAI
        if 'type' in tool and tool['type'] == 'function' and 'function' in tool:
            function_info = tool['function']
            function_name = function_info.get('name')
            
            if function_name:
                logger.debug(f"Processing tool: {function_name}")
                tools_config.append({
                    'name': function_name,
                    'description': function_info.get('description', ''),
                    'parameters': function_info.get('parameters', {})
                })
    
    # Обработка tool_choice
    has_forced_tool = False
    tool_type_mentioned = None
    
    # Если есть явный выбор инструмента
    if tool_choice:
        if tool_choice == "auto":
            logger.debug("Using automatic tool choice")
        elif tool_choice == "none":
            logger.debug("Tool choice set to none, disabling tools")
            return None, None, False
        elif isinstance(tool_choice, dict) and 'type' in tool_choice and tool_choice['type'] == 'function':
            # Форсируем использование конкретного инструмента
            function_name = tool_choice.get('function', {}).get('name')
            if function_name:
                logger.info(f"Forcing tool choice: {function_name}")
                has_forced_tool = True
                tool_type_mentioned = function_name
    
    # Если нет явного выбора, но есть ключевые слова, попробуем определить инструмент
    if not has_forced_tool:
        if contains_python_request:
            tool_type_mentioned = "execute_python"
            has_forced_tool = True
        elif contains_search_request:
            tool_type_mentioned = "web_search"
            has_forced_tool = True
    
    logger.info(f"Configured {len(tools_config)} tools for model {model}")
    if has_forced_tool:
        logger.info(f"Forcing tool choice: {tool_type_mentioned}")
    
    return tools_config, tool_type_mentioned, has_forced_tool

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
        
        # Расширенное логирование
        logger.info(f"Executing Python code with timeout {timeout}s: {code[:100]}...")
        
        # Execute the code
        result = execute_python_code(code, timeout)
        
        # Детальное логирование результата
        log_prefix = "Success" if result["status"] == "success" else "Failed"
        logger.info(f"{log_prefix} Python execution. Return code: {result['return_code']}")
        
        if result["stderr"]:
            logger.warning(f"Python execution stderr: {result['stderr'][:200]}...")
        
        # Добавим больше информации в ответ
        response = {
            "status": result["status"],
            "output": result["stdout"],
            "error": result["stderr"] if result["stderr"] else None,
            "return_code": result["return_code"],
            "execution_time": f"{timeout}s (timeout)" if result["status"] == "timeout" else None
        }
        
        return response
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

def web_search(query, api_key=None, num_results=5):
    """
    Perform a web search using Google Search API with fallback to DuckDuckGo.
    Enhanced to provide better results for weather and time-sensitive queries.
    
    Args:
        query (str): Search query
        api_key (str): API key for Google Search
        num_results (int): Number of results to return (default: 5)
    
    Returns:
        dict: Search results
    """
    try:
        # Enhance query for better results
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        current_hour = datetime.datetime.now().strftime("%H:%M")
        
        # Проверяем, является ли запрос о погоде
        weather_keywords = ['погода', 'weather', 'temperature', 'forecast', 'температура', 'прогноз']
        is_weather_query = any(keyword in query.lower() for keyword in weather_keywords)
        
        # Улучшаем запрос в зависимости от типа
        if is_weather_query:
            # Для запросов о погоде добавляем текущую дату и время
            enhanced_query = f"{query} {current_date} {current_hour}"
            logger.info(f"Enhanced weather query with date and time: {enhanced_query}")
        else:
            # Для обычных запросов добавляем только дату
            enhanced_query = f"{query} {current_date}"
            logger.info(f"Enhanced search query with date: {enhanced_query}")
        
        # Try Google Search first if keys are available
        google_api_key = os.environ.get('GOOGLE_API_KEY')
        google_cse_id = os.environ.get('GOOGLE_CSE_ID')
        
        if google_api_key and google_cse_id:
            logger.info(f"Performing Google search for: {enhanced_query}")
            
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'q': enhanced_query,
                'key': google_api_key,
                'cx': google_cse_id,
                'num': min(num_results, 10),  # Google limits to 10 results
                'sort': 'date'  # Сортировка по дате для получения свежих результатов
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json()
            
            formatted_results = []
            for item in results.get('items', []):
                # Извлекаем дату публикации, если доступна
                pub_date = ''
                if 'pagemap' in item and 'metatags' in item['pagemap'] and len(item['pagemap']['metatags']) > 0:
                    pub_date = item['pagemap']['metatags'][0].get('article:published_time', '')
                
                # Добавляем результат
                formatted_results.append({
                    "title": item.get('title', ''),
                    "url": item.get('link', ''),
                    "snippet": item.get('snippet', ''),
                    "date": pub_date
                })
            
            # Для запросов о погоде пытаемся получить актуальные данные
            if is_weather_query and formatted_results:
                for result in formatted_results:
                    # Проверяем, содержит ли сниппет числа температуры
                    if re.search(r'[-+]?\d+°[CF]', result['snippet']) or re.search(r'[-+]?\d+\s*градус', result['snippet'].lower()):
                        # Помечаем как высокоприоритетный результат погоды
                        result['is_weather_data'] = True
                        # Перемещаем в начало списка
                        formatted_results.remove(result)
                        formatted_results.insert(0, result)
            
            logger.info(f"Google search returned {len(formatted_results)} results")
            return {
                "results": formatted_results,
                "query": query,
                "enhanced_query": enhanced_query
            }
        
        # Fallback to DuckDuckGo if Google Search is not configured
        logger.info("Google Search API keys not found, falling back to DuckDuckGo")
        from duckduckgo_search import DDGS
        
        results = []
        try:
            with DDGS() as ddgs:
                # Определяем временной диапазон в зависимости от запроса
                time_range = 'd' if is_weather_query else 'w'  # d = day, w = week
                
                # Используем timerange для получения свежих результатов
                ddgs_results = list(ddgs.text(enhanced_query, max_results=num_results, timerange=time_range))
                for r in ddgs_results:
                    results.append({
                        "title": r.get('title', ''),
                        "url": r.get('href', ''),
                        "snippet": r.get('body', ''),
                        "date": r.get('published', '')
                    })
                
                # Если запрос о погоде и результатов мало, пытаемся использовать более специфичный запрос
                if is_weather_query and len(results) < 2:
                    specific_weather_query = query + " site:weather.com OR site:accuweather.com OR site:weatherunderground.com"
                    logger.info(f"Using specific weather query: {specific_weather_query}")
                    
                    weather_results = list(ddgs.text(specific_weather_query, max_results=3, timerange='d'))
                    for r in weather_results:
                        results.append({
                            "title": r.get('title', ''),
                            "url": r.get('href', ''),
                            "snippet": r.get('body', ''),
                            "date": r.get('published', ''),
                            "is_weather_data": True
                        })
                
                # Если недостаточно результатов, добавляем обычный поиск
                if len(results) < num_results:
                    logger.info(f"Not enough recent results, adding general results for: {enhanced_query}")
                    general_results = list(ddgs.text(enhanced_query, max_results=num_results - len(results)))
                    for r in general_results:
                        results.append({
                            "title": r.get('title', ''),
                            "url": r.get('href', ''),
                            "snippet": r.get('body', ''),
                            "date": r.get('published', '')
                        })
        except Exception as ddg_err:
            logger.error(f"DuckDuckGo search error: {str(ddg_err)}")
            # Альтернативный вариант - использовать регулярный DuckDuckGo поиск
            try:
                logger.info(f"Trying alternative DuckDuckGo search for: {enhanced_query}")
                with DDGS() as ddgs:
                    ddgs_results = list(ddgs.text(enhanced_query, max_results=num_results))
                    for r in ddgs_results:
                        results.append({
                            "title": r.get('title', ''),
                            "url": r.get('href', ''),
                            "snippet": r.get('body', ''),
                            "date": r.get('published', '')
                        })
            except Exception as alt_err:
                logger.error(f"Alternative DuckDuckGo search also failed: {str(alt_err)}")
        
        logger.info(f"DuckDuckGo search returned {len(results)} results")
        return {
            "results": results,
            "query": query,
            "enhanced_query": enhanced_query
        }
        
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        # If Google Search fails, try DuckDuckGo as fallback
        if google_api_key and google_cse_id:
            logger.info("Google Search failed, trying DuckDuckGo as fallback")
            try:
                from duckduckgo_search import DDGS
                
                results = []
                with DDGS() as ddgs:
                    # Используем timerange='d' для свежих результатов
                    time_range = 'd' if is_weather_query else 'w'
                    ddgs_results = list(ddgs.text(enhanced_query, max_results=num_results, timerange=time_range))
                    for r in ddgs_results:
                        results.append({
                            "title": r.get('title', ''),
                            "url": r.get('href', ''),
                            "snippet": r.get('body', ''),
                            "date": r.get('published', '')
                        })
                
                logger.info(f"DuckDuckGo fallback search returned {len(results)} results")
                return {
                    "results": results,
                    "query": query,
                    "enhanced_query": enhanced_query
                }
            except Exception as ddg_e:
                logger.error(f"DuckDuckGo fallback search error: {str(ddg_e)}")
                
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

@limiter.limit("500 per minute")
def process_user_input(messages):
    """
    Process the user input from messages, extracting text and images.
    
    Args:
        messages (list): List of message objects
        
    Returns:
        tuple: (user_input, image_paths, has_image)
    """
    user_input = ""
    image_paths = []
    has_image = False
    
    # Check for empty messages
    if not messages:
        logger.warning("No messages provided")
        return "", [], False
    
    # Get the last user message
    last_user_message = None
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            last_user_message = msg
            break
    
    if not last_user_message:
        logger.warning("No user message found")
        return "", [], False
    
    # Extract content from the last user message
    content = last_user_message.get('content', '')
    
    # Handle multimodal content
    if isinstance(content, list):
        # Content is a list of objects
        text_parts = []
        
        for item in content:
            item_type = item.get('type')
            
            if item_type == 'text':
                text_parts.append(item.get('text', ''))
            elif item_type == 'image_url':
                # Process image URL
                image_url = item.get('image_url', {}).get('url', '')
                if image_url.startswith('data:'):
                    # Base64 encoded image
                    try:
                        # Extract content type and base64 data
                        parts = image_url.split(',', 1)
                        if len(parts) == 2:
                            content_type = parts[0].split(';')[0].split(':')[1]
                            base64_data = parts[1]
                            
                            # Determine file extension
                            ext = 'jpg'  # Default
                            if 'png' in content_type:
                                ext = 'png'
                            elif 'webp' in content_type:
                                ext = 'webp'
                            elif 'gif' in content_type:
                                ext = 'gif'
                            
                            # Save image to temp file
                            import base64
                            import tempfile
                            import os
                            
                            temp_dir = tempfile.gettempdir()
                            temp_file = os.path.join(temp_dir, f"image_{uuid.uuid4()}.{ext}")
                            
                            with open(temp_file, 'wb') as f:
                                f.write(base64.b64decode(base64_data))
                            
                            image_paths.append(temp_file)
                            has_image = True
                            logger.debug(f"Saved image to {temp_file}")
                        else:
                            logger.error("Invalid data URL format")
                    except Exception as e:
                        logger.error(f"Error processing base64 image: {str(e)}")
                else:
                    # URL to image - not supported in this implementation
                    logger.warning(f"External image URLs not supported: {image_url[:30]}...")
        
        user_input = "\n".join(text_parts)
    else:
        # Content is a string
        user_input = content
    
    return user_input, image_paths, has_image

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def conversation():
    """
    Main endpoint for the OpenAI-compatible API conversation
    """
    # Parse request data and validate
    if not request.is_json:
        return ERROR_HANDLER(1400, detail="Request must be JSON")
    
    # Извлекаем API-ключ
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error("Invalid Authentication")
        return ERROR_HANDLER(1021)
    
    api_key = auth_header.split(" ")[1]
    
    # Определяем заголовки для запросов к API 1minAI
    headers = {
        "API-KEY": api_key, 
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    request_data = request.get_json()
    if 'messages' not in request_data or not isinstance(request_data['messages'], list):
        return ERROR_HANDLER(1400, detail="Missing messages field or not a list")
    
    # Get model
    model = request_data.get('model', DEFAULT_MODEL)
    logger.debug(f"Received conversation request for model {model}")
    
    # Clone the request data to avoid modifying the original
    request_data_clone = copy.deepcopy(request_data)
    
    # Extract system message if present
    system_message = None
    messages = request_data_clone.get('messages', [])
    
    if messages and messages[0].get('role') == 'system':
        system_message = messages[0].get('content', '')
        logger.debug(f"System message detected, length: {len(system_message)}")
    
    # Process tools configuration
    tools_config, tool_type_mentioned, has_tools_request = process_tools(request_data_clone)
    if tools_config:
        logger.info(f"Adding {len(tools_config)} tools to request for model {model}")
        if tool_type_mentioned:
            logger.info(f"Tool type mentioned: {tool_type_mentioned}")
    
    # Process user input from the last user message
    user_input, image_paths, has_image = process_user_input(messages)
    
    # Process multimodal content
    if has_image:
        logger.debug("Request contains images, processing as multimodal content")
    
    # Count input tokens
    prompt_token = calculate_token(' '.join([msg.get('content', '') for msg in messages if isinstance(msg.get('content'), str)]), model)
    
    # Prepare final messages for 1minAI API
    all_messages = []
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content')
        
        # Skip system message as it's handled separately
        if role == 'system':
            continue
        
        if role == 'user' or role == 'assistant':
            # For multimodal content
            if isinstance(content, list):
                # Content is a list of objects, handle text and image separately
                msg_content = ""
                for item in content:
                    if item.get('type') == 'text':
                        msg_content += item.get('text', '')
                
                all_messages.append({"role": role, "content": msg_content})
            else:
                all_messages.append({"role": role, "content": content})
        
        # Tool responses
        elif role == 'tool':
            tool_call_id = msg.get('tool_call_id')
            name = msg.get('name')
            content = msg.get('content')
            
            # Valid tool response
            if tool_call_id and name and content:
                all_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "content": content
                })
    
    # Prepare 1minAI API request
    # Convert all_messages to the right format for 1minAI API
    processed_messages = []
    for msg in all_messages:
        if msg["role"] == "system":
            # Skip system messages or handle them differently if needed
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle multi-part content (text + images)
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content = " ".join(text_parts)
        processed_messages.append({
            "role": msg["role"],
            "content": content
        })
    
    # Convert to JSON string for 1minAI API
    # Только последнее сообщение пользователя будет отправлено в API
    user_messages = [msg for msg in processed_messages if msg["role"] == "user"]
    if user_messages:
        last_user_message = user_messages[-1]["content"]
    else:
        last_user_message = "Привет"  # Дефолтное сообщение, если не найдено сообщение пользователя
    
    payload = {
        "type": "CHAT_WITH_AI",
        "model": model,
        "promptObject": {
            "prompt": last_user_message,
            "isMixed": False,
            "webSearch": False
        }
    }
    
    # Добавляем webSearch: true если используется инструмент web_search
    if any(tool.get("name") == "get_search_results" for tool in tools_config):
        payload["promptObject"]["webSearch"] = True
        payload["promptObject"]["numOfSite"] = 3
        payload["promptObject"]["maxWord"] = 1000
        logger.debug("Enabling web search in the request")
    
    # Set request type based on content
    if has_image:
        payload["type"] = "CHAT_WITH_IMAGE"
        payload["promptObject"]["imageList"] = image_paths
    
    # Add tools if configured
    if tools_config:
        payload["toolsConfig"] = {
            "tools": tools_config
        }
        
        # Add tool_choice if specified
        if tool_type_mentioned:
            payload["toolsConfig"]["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_type_mentioned}
            }
        
        logger.debug(f"Adding tools configuration to payload: {json.dumps(payload['toolsConfig'])}")
    
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
            # Логирование полного запроса для отладки
            if has_tools_request:
                tools_part = payload.get("toolsConfig", {})
                logger.info(f"Sending request with tools for {tool_type_mentioned}: {json.dumps(tools_part)}")
            
            response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            one_min_response = response.json()
            
            # Логирование ответа API и проверка наличия вызовов инструментов
            if 'function_call' in one_min_response or 'tool_calls' in one_min_response:
                logger.info("Response contains function/tool calls!")
                if 'function_call' in one_min_response:
                    logger.info(f"Function call: {json.dumps(one_min_response['function_call'])}")
                if 'tool_calls' in one_min_response:
                    logger.info(f"Tool calls: {json.dumps(one_min_response['tool_calls'])}")
            elif has_tools_request:
                logger.warning(f"Request mentioned {tool_type_mentioned} but response doesn't contain any tool calls")
            
            transformed_response = transform_response(one_min_response, request_data, prompt_token)
            flask_response = make_response(jsonify(transformed_response))
            return set_response_headers(flask_response)
            
        except requests.RequestException as e:
            logger.error(f"Error with 1minAI API: {str(e)}")
            return ERROR_HANDLER(1500, detail=str(e))
    else:
        # Streaming Response
        logger.debug("Streaming AI Response")
        try:
            # Логирование полного запроса для отладки
            if has_tools_request:
                tools_part = payload.get("toolsConfig", {})
                logger.info(f"Sending streaming request with tools for {tool_type_mentioned}: {json.dumps(tools_part)}")
                
            def generate():
                response = requests.post(ONE_MIN_CONVERSATION_API_STREAMING_URL, json=payload, headers=headers, stream=True)
                
                if response.status_code != 200:
                    logger.warning(f"Streaming request failed for model {model}, trying non-streaming mode")
                    non_streaming_response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
                    non_streaming_response.raise_for_status()
                    one_min_response = non_streaming_response.json()
                    transformed_response = transform_response(one_min_response, request_data, prompt_token)
                    
                    # Формируем ответ как событие SSE
                    yield f"data: {json.dumps(transformed_response)}\n\n"
                    yield f"data: [DONE]\n\n"
                    return
                
                # Обработка потокового ответа
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]  # Удаляем префикс "data: "
                            if data == "[DONE]":
                                yield f"data: [DONE]\n\n"
                                break
                            
                            try:
                                chunk = json.loads(data)
                                # Проверяем наличие вызовов инструментов в потоковом ответе
                                if 'function_call' in chunk or 'tool_calls' in chunk:
                                    logger.info("Streaming response contains function/tool calls!")
                                    if 'function_call' in chunk:
                                        logger.info(f"Function call in stream: {json.dumps(chunk['function_call'])}")
                                    if 'tool_calls' in chunk:
                                        logger.info(f"Tool calls in stream: {json.dumps(chunk['tool_calls'])}")
                                
                                transformed_chunk = transform_streaming_response(chunk, request_data, model)
                                yield f"data: {json.dumps(transformed_chunk)}\n\n"
                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing streaming response: {str(e)}")
                                logger.error(f"Raw data: {data}")
                                
                # Проверка отсутствия вызовов инструментов в конце потока
                if has_tools_request:
                    logger.warning(f"Streaming request mentioned {tool_type_mentioned} but no tool calls were received in the stream")
            
            return Response(generate(), mimetype='text/event-stream')
            
        except requests.RequestException as e:
            logger.error(f"Error with 1minAI API streaming: {str(e)}")
            return ERROR_HANDLER(1500, detail=str(e))
        except Exception as e:
            logger.error(f"Error with 1minAI API streaming: {str(e)}")
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

def transform_response(response_data, request_data, prompt_token=0):
    """
    Transform 1minAI response to OpenAI format
    """
    transformed_response = {
        "id": "chatcmpl-" + str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_data.get('model', DEFAULT_MODEL),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_token,
            "completion_tokens": 0,
            "total_tokens": prompt_token
        }
    }
    
    # Extract content from 1minAI response
    if "choices" in response_data and len(response_data["choices"]) > 0:
        if "message" in response_data["choices"][0]:
            content = response_data["choices"][0]["message"].get("content")
            transformed_response["choices"][0]["message"]["content"] = content
    elif "content" in response_data:
        content = response_data.get("content")
        transformed_response["choices"][0]["message"]["content"] = content
    
    logger.debug(f"Response received, preparing transformation to OpenAI format")
    
    # Handle function/tool calls if present
    tool_calls = []
    
    # Check for old function call format
    if "function_call" in response_data:
        logger.info(f"Old function call format detected: {json.dumps(response_data['function_call'])}")
        # Convert old function call to new tool call
        function_call = response_data["function_call"]
        if "name" in function_call and "arguments" in function_call:
            try:
                # Try to parse the arguments as JSON
                arguments = json.loads(function_call["arguments"])
                logger.info(f"Function {function_call['name']} arguments parsed successfully")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing function arguments: {str(e)}")
                # If parsing fails, use the raw string
                arguments = function_call["arguments"]
            
            tool_call = {
                "id": "call_" + str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": function_call["name"],
                    "arguments": function_call["arguments"]
                }
            }
            tool_calls.append(tool_call)
            
            logger.info(f"Converted function call to tool call: {json.dumps(tool_call)}")
    
    # Check for new tool calls format
    if "tool_calls" in response_data:
        logger.info(f"New tool calls format detected: {json.dumps(response_data['tool_calls'])}")
        for tool_call in response_data["tool_calls"]:
            if "function" in tool_call:
                tool_calls.append(tool_call)
    
    # Process tool calls if present
    if tool_calls:
        processed_tool_calls = []
        
        for tool_call in tool_calls:
            if "function" in tool_call:
                function_name = tool_call["function"]["name"]
                arguments_str = tool_call["function"]["arguments"]
                
                logger.info(f"Processing tool call for function: {function_name}")
                
                try:
                    # Try to parse the arguments as JSON
                    arguments = json.loads(arguments_str)
                    logger.debug(f"Arguments parsed successfully: {json.dumps(arguments)}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing arguments: {str(e)}")
                    # If parsing fails, use the raw string
                    arguments = arguments_str
                
                # Execute the function call based on the function name
                if function_name == "execute_python":
                    logger.info(f"Executing Python code, length: {len(arguments.get('code', ''))}")
                    start_time = time.time()
                    try:
                        result = execute_python_code(arguments.get("code", ""))
                        execution_time = time.time() - start_time
                        logger.info(f"Python execution completed in {execution_time:.2f}s, output length: {len(result)}")
                    except Exception as e:
                        logger.error(f"Error executing Python code: {str(e)}")
                        result = f"Error: {str(e)}"
                    
                    tool_call["function"]["output"] = result
                    processed_tool_calls.append(tool_call)
                
                elif function_name == "web_search":
                    logger.info(f"Performing web search for query: {arguments.get('search_term', '')}")
                    
                    try:
                        # Извлечение API ключа DuckDuckGo из переменных окружения
                        ddg_api_key = os.environ.get("DUCKDUCKGO_API_KEY")
                        if not ddg_api_key:
                            logger.warning("DuckDuckGo API key not found in environment variables")
                        
                        start_time = time.time()
                        search_results = web_search(arguments.get("search_term", ""))
                        execution_time = time.time() - start_time
                        
                        logger.info(f"Web search completed in {execution_time:.2f}s, found {len(search_results)} results")
                        
                        # Преобразуем результаты в строку JSON для вывода
                        result = json.dumps(search_results, ensure_ascii=False)
                        
                        tool_call["function"]["output"] = result
                        processed_tool_calls.append(tool_call)
                    except Exception as e:
                        logger.error(f"Error performing web search: {str(e)}")
                        tool_call["function"]["output"] = f"Error: {str(e)}"
                        processed_tool_calls.append(tool_call)
                
                else:
                    logger.warning(f"Unknown function: {function_name}")
                    tool_call["function"]["output"] = f"Error: Unknown function {function_name}"
                    processed_tool_calls.append(tool_call)
        
        # Add the processed tool calls to the response
        if processed_tool_calls:
            logger.info(f"Adding {len(processed_tool_calls)} processed tool calls to response")
            transformed_response["choices"][0]["message"]["tool_calls"] = processed_tool_calls
            transformed_response["choices"][0]["finish_reason"] = "tool_calls"
    
    # Handle usage information if available
    if "usage" in response_data:
        usage = response_data["usage"]
        if "completion_tokens" in usage:
            transformed_response["usage"]["completion_tokens"] = usage["completion_tokens"]
            transformed_response["usage"]["total_tokens"] = prompt_token + usage["completion_tokens"]
    
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
                        logger.debug(f"Extracting text from file with extension {file_ext}")
                        if file_ext == '.docx':
                            # Use docx2txt for DOCX
                            logger.debug("Using docx2txt to process DOCX file")
                            try:
                                text_content = docx2txt.process(temp_file_path)
                                logger.debug(f"Successfully extracted text from DOCX file, length: {len(text_content)} characters")
                            except Exception as docx_err:
                                logger.error(f"Error processing DOCX file with docx2txt: {str(docx_err)}")
                                text_content = "Error extracting text from DOCX file."
                        else:
                            # Use python-docx for DOC (with conversion limitations)
                            logger.debug("Using python-docx to process DOC file")
                            try:
                                doc = DocxDocument(temp_file_path)
                                text_content = "\n".join([para.text for para in doc.paragraphs])
                                logger.debug(f"Successfully extracted text from DOC file, length: {len(text_content)} characters")
                            except Exception as doc_err:
                                logger.error(f"Error processing DOC file with python-docx: {str(doc_err)}")
                                text_content = "Error extracting text from DOC file. Please convert to DOCX or PDF format."
                        
                        # Create a text file with the extracted content
                        text_filename = os.path.splitext(file.filename)[0] + ".txt"
                        text_file_io = BytesIO(text_content.encode('utf-8'))
                        
                        # Upload the text file instead
                        logger.debug(f"Uploading converted text file: {text_filename}")
                        upload_result = upload_file_to_1min(text_file_io, text_filename, 'text/plain', api_key)
                        
                        # Add a note about the conversion for the original request
                        logger.debug(f"Adding note about file conversion from {file.filename} to {text_filename}")
                        original_filename = file.filename
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

class CodeSecurityChecker(ast.NodeVisitor):
    """
    Проверяет Python-код на наличие потенциально опасных операций.
    """
    
    def __init__(self):
        self.has_dangerous_operations = False
        self.dangerous_operations = []
    
    def visit_Import(self, node):
        dangerous_modules = [
            'os', 'subprocess', 'sys', 'shutil', 
            'pathlib', 'pickle', 'marshal', 'socket',
            'multiprocessing', 're', 'tempfile',
            'glob', 'ftplib', 'smtplib', 'winreg'
        ]
        
        for name in node.names:
            if name.name in dangerous_modules:
                self.has_dangerous_operations = True
                self.dangerous_operations.append(f"Importing dangerous module: {name.name}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        dangerous_modules = [
            'os', 'subprocess', 'sys', 'shutil', 
            'pathlib', 'pickle', 'marshal', 'socket',
            'multiprocessing', 're', 'tempfile',
            'glob', 'ftplib', 'smtplib', 'winreg'
        ]
        
        if node.module in dangerous_modules:
            self.has_dangerous_operations = True
            self.dangerous_operations.append(f"Importing from dangerous module: {node.module}")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        dangerous_functions = [
            'exec', 'eval', 'compile', 'open', 'input', 
            '__import__', 'getattr', 'setattr', 'delattr'
        ]
        
        # Проверяем вызов функции
        if isinstance(node.func, ast.Name) and node.func.id in dangerous_functions:
            self.has_dangerous_operations = True
            self.dangerous_operations.append(f"Calling dangerous function: {node.func.id}")
        
        # Проверяем атрибуты объектов
        elif isinstance(node.func, ast.Attribute):
            dangerous_methods = [
                'system', 'popen', 'spawn', 'call', 'check_output', 
                'check_call', 'open', 'read', 'write', 'remove', 
                'rmdir', 'mkdir', 'chdir', 'delete', 'unlink', 
                'makedirs', 'chmod', 'chown', 'rename', 'loadtxt', 
                'savetxt', 'load', 'dump', 'loads', 'dumps'
            ]
            
            if node.func.attr in dangerous_methods:
                self.has_dangerous_operations = True
                self.dangerous_operations.append(f"Calling dangerous method: {node.func.attr}")
        
        self.generic_visit(node)

@contextmanager
def time_limit(seconds):
    """
    Контекстный менеджер для ограничения времени выполнения кода.
    
    Args:
        seconds (int): максимальное время выполнения в секундах
    """
    def signal_handler(signum, frame):
        raise TimeoutError("Code execution timed out")
    
    previous_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

def execute_python_code(code):
    """
    Безопасно выполняет Python-код с ограничением времени и проверками безопасности.
    
    Args:
        code (str): Python-код для выполнения
        
    Returns:
        dict: результат выполнения кода (результат, статус, ошибка если есть)
    """
    # Проверка безопасности кода
    try:
        parsed_ast = ast.parse(code)
        security_checker = CodeSecurityChecker()
        security_checker.visit(parsed_ast)
        
        if security_checker.has_dangerous_operations:
            logger.warning(f"Потенциально опасный код обнаружен: {security_checker.dangerous_operations}")
            return {
                "result": None,
                "status": "error",
                "error": f"Код содержит потенциально опасные операции: {', '.join(security_checker.dangerous_operations)}"
            }
    except SyntaxError as e:
        logger.error(f"Синтаксическая ошибка в коде: {str(e)}")
        return {
            "result": None,
            "status": "error",
            "error": f"Синтаксическая ошибка: {str(e)}"
        }
    
    # Подготовка для перехвата stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Создаем временное окружение для выполнения
    execution_environment = {'__builtins__': __builtins__}
    
    # Функция выполнения в отдельном процессе
    def execute_in_subprocess(code, result_queue):
        try:
            # Перенаправляем вывод
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Выполняем код
            exec(code, execution_environment)
            
            # Получаем результаты
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            # Помещаем результаты в очередь
            result_queue.put({
                "result": stdout_output,
                "error": stderr_output if stderr_output else None,
                "status": "success" if not stderr_output else "warning"
            })
        except Exception as e:
            error_message = f"{type(e).__name__}: {str(e)}"
            traceback_str = traceback.format_exc()
            result_queue.put({
                "result": stdout_capture.getvalue(),
                "error": error_message,
                "traceback": traceback_str,
                "status": "error"
            })
        finally:
            # Восстанавливаем стандартные выводы
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
    
    # Запускаем выполнение с ограничением времени
    try:
        # Создаем очередь для получения результатов
        result_queue = multiprocessing.Queue()
        
        # Создаем и запускаем процесс
        process = multiprocessing.Process(
            target=execute_in_subprocess,
            args=(code, result_queue)
        )
        
        # Запускаем процесс с ограничением времени
        process.start()
        
        # Ждем выполнения в течение 10 секунд
        process.join(10)
        
        # Проверяем, завершился ли процесс
        if process.is_alive():
            # Если процесс все еще работает, завершаем его
            process.terminate()
            process.join()
            
            logger.warning("Код выполнялся слишком долго и был прерван")
            return {
                "result": stdout_capture.getvalue(),
                "status": "error",
                "error": "Превышено время выполнения (10 секунд)"
            }
        
        # Получаем результат выполнения
        if not result_queue.empty():
            result = result_queue.get()
            return result
        else:
            return {
                "result": None,
                "status": "error",
                "error": "Не удалось получить результат выполнения"
            }
            
    except Exception as e:
        logger.error(f"Ошибка при выполнении кода: {str(e)}")
        return {
            "result": None,
            "status": "error",
            "error": f"Ошибка выполнения: {str(e)}"
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

@app.route('/', methods=['POST'])
def relay():
    """
    Main handler for API requests
    """
    start_time = datetime.datetime.now()
    response_data = {
        "status": "error",
        "message": "Unknown error occurred",
        "data": None,
        "processing_time": None
    }
    
    try:
        data = request.get_json()
        if not data:
            response_data["message"] = "No data provided"
            return jsonify(response_data)
        
        query = data.get("query", "")
        api_key = data.get("api_key", "")
        model = data.get("model", DEFAULT_MODEL)
        language = data.get("language", "en")
        tool = data.get("tool", "")
        
        if not query and not tool:
            response_data["message"] = "No query or tool specified"
            return jsonify(response_data)
        
        # Определяем, какой инструмент использовать
        if tool == "web_search":
            result = web_search(query, api_key)
            response_data["status"] = "success"
            response_data["data"] = result
            response_data["message"] = "Web search completed successfully"
        elif tool == "execute_python":
            code = data.get("code", "")
            if not code:
                response_data["message"] = "No code provided for execution"
                return jsonify(response_data)
            
            result = execute_python_code(code)
            response_data["status"] = "success"
            response_data["data"] = result
            response_data["message"] = "Python code executed successfully"
        else:
            response_data["message"] = f"Unknown tool: {tool}"
            
    except Exception as e:
        logger.error(f"Error in relay: {str(e)}")
        response_data["message"] = f"Error: {str(e)}"
        
    finally:
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        response_data["processing_time"] = processing_time
        logger.info(f"Request processed in {processing_time:.2f} seconds")
        
        response = jsonify(response_data)
        return set_response_headers(response)

@app.route('/api/tools', methods=['POST', 'OPTIONS'])
def tools_endpoint():
    """
    Эндпоинт для прямого выполнения инструментов через HTTP API
    """
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    start_time = datetime.datetime.now()
    response_data = {
        "status": "error",
        "message": "Unknown error occurred",
        "data": None,
        "processing_time": None
    }
    
    try:
        data = request.get_json()
        if not data:
            response_data["message"] = "No data provided"
            return jsonify(response_data)
        
        tool = data.get("tool", "")
        if not tool:
            response_data["message"] = "No tool specified"
            return jsonify(response_data)
        
        query = data.get("query", "")
        code = data.get("code", "")
        api_key = data.get("api_key", "")
        
        # Определяем, какой инструмент использовать
        if tool == "web_search":
            if not query:
                response_data["message"] = "No query provided for web search"
                return jsonify(response_data)
            
            logger.info(f"Processing web_search request for query: {query}")
            result = web_search(query, api_key)
            response_data["status"] = "success"
            response_data["data"] = result
            response_data["message"] = "Web search completed successfully"
            
        elif tool == "execute_python":
            if not code:
                response_data["message"] = "No code provided for execution"
                return jsonify(response_data)
            
            logger.info(f"Processing execute_python request: {code[:50]}...")
            result = execute_python_code(code)
            response_data["status"] = "success"
            response_data["data"] = result
            response_data["message"] = "Python code executed successfully"
            
        else:
            response_data["message"] = f"Unknown tool: {tool}"
            
    except Exception as e:
        logger.error(f"Error in tools_endpoint: {str(e)}")
        response_data["message"] = f"Error: {str(e)}"
        
    finally:
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        response_data["processing_time"] = processing_time
        logger.info(f"Tools request processed in {processing_time:.2f} seconds")
        
        response = jsonify(response_data)
        return set_response_headers(response)

def main():
    # Start the server
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)

# Запуск приложения
if __name__ == "__main__":
    main()

