# routes/functions.py
# Общие утилиты для маршрутов

# Реэкспортируем основные зависимости
from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import (
    ERROR_HANDLER, 
    handle_options_request, 
    set_response_headers, 
    create_session, 
    api_request, 
    safe_temp_file, 
    calculate_token
)
from utils.memcached import safe_memcached_operation

# Экспортируем общие функции
from .functions.shared_func import (
    validate_auth,
    handle_api_error,
    format_openai_response,
    format_image_response,
    stream_response,
    get_full_url,
    extract_data_from_api_response,
    extract_text_from_response,
    extract_image_urls,
    extract_audio_url
)

# Экспортируем функции для текстовых моделей
from .functions.txt_func import (
    format_conversation_history,
    get_model_capabilities,
    prepare_payload,
    transform_response,
    emulate_stream_response
)

# Экспортируем функции для изображений
from .functions.img_func import (
    build_generation_payload,
    parse_aspect_ratio,
    create_image_variations,
    process_image_tool_calls
)

# Экспортируем функции для аудио
from .functions.audio_func import (
    prepare_models_list,
    prepare_whisper_payload,
    prepare_tts_payload,
    try_models_in_sequence,
    upload_audio_file
)

# Экспортируем функции для файлов
from .functions.file_func import (
    get_user_files,
    save_user_files,
    create_temp_file,
    upload_asset,
    get_mime_type,
    retry_image_upload,
    format_file_response,
    create_api_response,
    find_conversation_id,
    find_file_by_id,
    create_conversation_with_files
)





