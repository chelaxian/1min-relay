# routes/__init__.py
# Инициализация пакета routes

# Импортируем необходимые модули
from utils.logger import logger
from utils.imports import *
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

# Делаем app и limiter доступными при импорте routes
import sys
mod = sys.modules[__name__]

# Импортируем app и limiter из корневого модуля
try:
    import app as root_app
    # Переносим объекты в текущий модуль
    mod.app = root_app.app
    mod.limiter = root_app.limiter
    mod.IMAGE_CACHE = root_app.IMAGE_CACHE
    mod.MEMORY_STORAGE = root_app.MEMORY_STORAGE
    mod.MAX_CACHE_SIZE = 100  # Максимальный размер кэша изображений
    logger.info("Глобальные объекты успешно переданы в модуль маршрутов")
    
    # Импортируем модуль функций вместо прямого импорта из shared_func
    from .functions import (
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
    
    # Импортируем модули маршрутов (blueprints регистрируются при импорте)
    from . import text, images, audio, files
    
    logger.info("Все модули маршрутов импортированы")
    
    # Обеспечиваем доступ к маршрутам из корневого модуля
    root_app.routes = mod
    logger.info("Модуль маршрутов добавлен в корневой модуль app")
    
except ImportError as e:
    logger.error(f"Не удалось импортировать app.py: {str(e)}. Маршруты могут работать некорректно.")

logger.info("Инициализация маршрутов завершена")
