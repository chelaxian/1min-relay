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
import pdfplumber
from docx import Document as DocxDocument
import yaml
from PIL import Image
import io

# Suppress warnings from flask_limiter
warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter.extension")

# Create a logger object
logger = logging.getLogger("1min-relay")
coloredlogs.install(level='DEBUG', logger=logger)

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
 / |  \/  (_)_ _ | _ \___| |    
 | | |\/| | | ' \|   / -_) / _` | || |
 |_|_|  |_|_|_||_|_|_\___|_\__,_|\_, |
        |__/ ''')

def calculate_token(sentence, model="DEFAULT"):
    """Calculate the number of tokens in a sentence based on the specified model."""
    if model.startswith("mistral"):
        tokenizer = MistralTokenizer.v3(is_tekken=True)
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
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(sentence)
        return len(tokens)
    else:
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(sentence)
        return len(tokens)

app = Flask(__name__)

if check_memcached_connection():
    limiter = Limiter(
        get_remote_address,
        app=app,
        storage_uri="memcached://memcached:11211",  # Connect to Memcached created with docker
    )
else:
    limiter = Limiter(
        get_remote_address,
        app=app,
    )
    logger.warning("Memcached is not available. Using in-memory storage for rate limiting. Not-Recommended")

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
    models_data = []
    if not PERMIT_MODELS_FROM_SUBSET_ONLY:
        one_min_models_data = [
            {"id": model_name, "object": "model", "owned_by": "1minai", "created": 1727389042}
            for model_name in ALL_ONE_MIN_AVAILABLE_MODELS
        ]
    else:
        one_min_models_data = [
            {"id": model_name, "object": "model", "owned_by": "1minai", "created": 1727389042}
            for model_name in SUBSET_OF_ONE_MIN_PERMITTED_MODELS
        ]
    models_data.extend(one_min_models_data)
    return jsonify({"data": models_data, "object": "list"})

def ERROR_HANDLER(code, model=None, key=None):
    error_codes = {
        1002: {"message": f"The model {model} does not exist.", "type": "invalid_request_error", "param": None, "code": "model_not_found", "http_code": 400},
        1020: {"message": f"Incorrect API key provided: {key}. You can find your API key at https://app.1min.ai/api.", "type": "authentication_error", "param": None, "code": "invalid_api_key", "http_code": 401},
        1021: {"message": "Invalid Authentication", "type": "invalid_request_error", "param": None, "code": None, "http_code": 401},
        1212: {"message": f"Incorrect Endpoint. Please use the /v1/chat/completions endpoint.", "type": "invalid_request_error", "param": None, "code": "model_not_supported", "http_code": 400},
        1044: {"message": f"This model does not support image inputs.", "type": "invalid_request_error", "param": None, "code": "model_not_supported", "http_code": 400},
        1412: {"message": f"No message provided.", "type": "invalid_request_error", "param": "messages", "code": "invalid_request_error", "http_code": 400},
        1423: {"message": f"No content in last message.", "type": "invalid_request_error", "param": "messages", "code": "invalid_request_error", "http_code": 400},
    }
    error_data = {k: v for k, v in error_codes.get(code, {"message": "Unknown error", "type": "unknown_error", "param": None, "code": None}).items() if k != "http_code"}
    logger.error(f"An error has occurred while processing the user's request. Error code: {code}")
    return jsonify({"error": error_data}), error_codes.get(code, {}).get("http_code", 400)

def format_conversation_history(messages, new_input):
    formatted_history = ["Conversation History:\n"]
    for message in messages:
        role = message.get('role', '').capitalize()
        content = message.get('content', '')
        if isinstance(content, list):
            content = '\n'.join(item['text'] for item in content if 'text' in item)
        formatted_history.append(f"{role}: {content}")
    if messages:
        formatted_history.append("Respond like normal. The conversation history will be automatically updated on the next MESSAGE. DO NOT ADD User: or Assistant: to your output. Just respond like normal.")
        formatted_history.append("User Message:\n")
    formatted_history.append(new_input) 
    return '\n'.join(formatted_history)

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
    headers = {'API-KEY': api_key}

    request_data = request.json
    messages = request_data.get('messages', [])

    if not messages:
        return ERROR_HANDLER(1412)

    user_input = messages[-1].get('content')

    if not user_input:
        return ERROR_HANDLER(1423)

    # Проверка на наличие изображений
    image = False
    image_path = None
    combined_text = ""

    if isinstance(user_input, list):
        for item in user_input:
            if 'text' in item:
                combined_text += item['text'] + "\n"
            if 'image_url' in item:
                if request_data.get('model', 'mistral-nemo') not in vision_supported_models:
                    return ERROR_HANDLER(1044, request_data.get('model', 'mistral-nemo'))

                try:
                    image_url = item['image_url']['url']
                    logger.debug(f"Processing image URL: {image_url}")
                    
                    binary_data = handle_image_upload(image_url)
                    logger.debug("Image successfully processed and converted to binary data.")

                    files = {'asset': ("relay" + str(uuid.uuid4()), binary_data, 'image/png')}
                    logger.debug("Uploading image to 1minAI asset endpoint...")
                    
                    asset_response = requests.post(ONE_MIN_ASSET_URL, files=files, headers=headers)
                    asset_response.raise_for_status()
                    
                    logger.debug(f"Asset upload response: {asset_response.json()}")
                    image_path = asset_response.json()['fileContent']['path']
                    logger.debug(f"Image uploaded successfully. Image path: {image_path}")
                    
                    image = True
                except Exception as e:
                    logger.error(f"An error occurred while processing the image: {str(e)}")
                    return ERROR_HANDLER(1044, request_data.get('model', 'mistral-nemo'))

        user_input = combined_text.strip()

    # Формирование истории диалога
    all_messages = format_conversation_history(messages, user_input)
    prompt_token = calculate_token(str(all_messages))

    if PERMIT_MODELS_FROM_SUBSET_ONLY and request_data.get('model', 'mistral-nemo') not in AVAILABLE_MODELS:
        return ERROR_HANDLER(1002, request_data.get('model', 'mistral-nemo'))

    logger.debug(f"Processing {prompt_token} prompt tokens with model {request_data.get('model', 'mistral-nemo')}")

    if not image:
        payload = {
            "type": "CHAT_WITH_AI",
            "model": request_data.get('model', 'mistral-nemo'),
            "promptObject": {
                "prompt": all_messages,
                "isMixed": False,
                "webSearch": False
            }
        }
    else:
        payload = {
            "type": "CHAT_WITH_IMAGE",
            "model": request_data.get('model', 'mistral-nemo'),
            "promptObject": {
                "prompt": all_messages,
                "isMixed": False,
                "imageList": [image_path]
            }
        }

    headers = {"API-KEY": api_key, 'Content-Type': 'application/json'}

    if not request_data.get('stream', False):
        response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        one_min_response = response.json()
        transformed_response = transform_response(one_min_response, request_data, prompt_token)
        response = make_response(jsonify(transformed_response))
        set_response_headers(response)
        return response, 200
    else:
        response_stream = requests.post(ONE_MIN_CONVERSATION_API_STREAMING_URL, data=json.dumps(payload), headers=headers, stream=True)
        if response_stream.status_code != 200:
            if response_stream.status_code == 401:
                return ERROR_HANDLER(1020)
            logger.error(f"An unknown error occurred while processing the user's request. Error code: {response_stream.status_code}")
            return ERROR_HANDLER(response_stream.status_code)
        return Response(stream_response(response_stream, request_data, request_data.get('model', 'mistral-nemo'), int(prompt_token)), content_type='text/event-stream')

        user_input = combined_text.strip()

    # Формирование истории диалога
    all_messages = format_conversation_history(messages, user_input)
    prompt_token = calculate_token(str(all_messages))

    if PERMIT_MODELS_FROM_SUBSET_ONLY and request_data.get('model', 'mistral-nemo') not in AVAILABLE_MODELS:
        return ERROR_HANDLER(1002, request_data.get('model', 'mistral-nemo'))

    logger.debug(f"Processing {prompt_token} prompt tokens with model {request_data.get('model', 'mistral-nemo')}")

    if not image:
        payload = {
            "type": "CHAT_WITH_AI",
            "model": request_data.get('model', 'mistral-nemo'),
            "promptObject": {
                "prompt": all_messages,
                "isMixed": False,
                "webSearch": False
            }
        }
    else:
        payload = {
            "type": "CHAT_WITH_IMAGE",
            "model": request_data.get('model', 'mistral-nemo'),
            "promptObject": {
                "prompt": all_messages,
                "isMixed": False,
                "imageList": [image_path]
            }
        }

    headers = {"API-KEY": api_key, 'Content-Type': 'application/json'}

    if not request_data.get('stream', False):
        response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        one_min_response = response.json()
        transformed_response = transform_response(one_min_response, request_data, prompt_token)
        response = make_response(jsonify(transformed_response))
        set_response_headers(response)
        return response, 200
    else:
        response_stream = requests.post(ONE_MIN_CONVERSATION_API_STREAMING_URL, data=json.dumps(payload), headers=headers, stream=True)
        if response_stream.status_code != 200:
            if response_stream.status_code == 401:
                return ERROR_HANDLER(1020)
            logger.error(f"An unknown error occurred while processing the user's request. Error code: {response_stream.status_code}")
            return ERROR_HANDLER(response_stream.status_code)
        return Response(stream_response(response_stream, request_data, request_data.get('model', 'mistral-nemo'), int(prompt_token)), content_type='text/event-stream')

def handle_file_upload(file):
    if file.filename.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages)
        return text
    elif file.filename.endswith('.docx'):
        doc = DocxDocument(file)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        return text
    elif file.filename.endswith('.yaml') or file.filename.endswith('.yml'):
        yaml_content = yaml.safe_load(file)
        return str(yaml_content)
    elif file.filename.endswith('.txt'):
        return file.read().decode('utf-8')
    else:
        return None

def handle_image_upload(image_url):
    if image_url.startswith("data:image"):
        # Base64-encoded image
        logger.debug("Detected Base64-encoded image.")
        base64_image = image_url.split(",")[1]
        binary_data = base64.b64decode(base64_image)
        logger.debug("Base64 image successfully decoded.")
    else:
        # URL of the image
        logger.debug(f"Downloading image from URL: {image_url}")
        response = requests.get(image_url)
        response.raise_for_status()
        binary_data = response.content
        logger.debug("Image successfully downloaded from URL.")
    return binary_data

def handle_options_request():
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response, 204

def transform_response(one_min_response, request_data, prompt_token):
    completion_token = calculate_token(one_min_response['aiRecord']["aiRecordDetail"]["resultObject"][0])
    logger.debug(f"Finished processing Non-Streaming response. Completion tokens: {str(completion_token)}")
    logger.debug(f"Total tokens: {str(completion_token + prompt_token)}")
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_data.get('model', 'mistral-nemo'),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": one_min_response['aiRecord']["aiRecordDetail"]["resultObject"][0],
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_token,
            "completion_tokens": completion_token,
            "total_tokens": prompt_token + completion_token
        }
    }

def set_response_headers(response):
    response.headers['Content-Type'] = 'application/json'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['X-Request-ID'] = str(uuid.uuid4())

def stream_response(response, request_data, model, prompt_tokens):
    all_chunks = ""
    for chunk in response.iter_content(chunk_size=1024):
        finish_reason = None
        return_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request_data.get('model', 'mistral-nemo'),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk.decode('utf-8')
                    },
                    "finish_reason": finish_reason
                }
            ]
        }
        all_chunks += chunk.decode('utf-8')
        yield f"data: {json.dumps(return_chunk)}\n\n"
    tokens = calculate_token(all_chunks)
    logger.debug(f"Finished processing streaming response. Completion tokens: {str(tokens)}")
    logger.debug(f"Total tokens: {str(tokens + prompt_tokens)}")
    final_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request_data.get('model', 'mistral-nemo'),
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": ""    
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens,
            "total_tokens": tokens + prompt_tokens
        }
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

if __name__ == '__main__':
    internal_ip = socket.gethostbyname(socket.gethostname())
    response = requests.get('https://api.ipify.org')
    public_ip = response.text
    logger.info(f"""{printedcolors.Color.fg.lightcyan}  
Server is ready to serve at:
Internal IP: {internal_ip}:5001
Public IP: {public_ip} (only if you've setup port forwarding on your router.)
Enter this url to OpenAI clients supporting custom endpoint:
{internal_ip}:5001/v1
If does not work, try:
{internal_ip}:5001/v1/chat/completions
{printedcolors.Color.reset}""")
    serve(app, host='0.0.0.0', port=5001, threads=6)
