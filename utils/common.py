# version 1.0.3 #increment every time you make changes
# utils/common.py
# Общие утилиты
from .imports import *
from .logger import logger
from .constants import *

def calculate_token(sentence, model="DEFAULT"):
    """
    Рассчитывает количество токенов в строке, используя соответствующую модели токенизацию.
    
    Args:
        sentence (str): Текст для подсчета токенов
        model (str): Модель, для которой необходимо посчитать токены
        
    Returns:
        int: Количество токенов в строке
    """
    if not sentence:
        return 0
        
    try:
        # Если это аудио модель (TTS или STT), используем специальную логику
        if model in TEXT_TO_SPEECH_MODELS:
            # Для TTS считаем количество символов и слов
            char_count = len(sentence)
            word_count = len(sentence.split())
            # Приблизительно 1 токен = 4 символа или 0.75 слова
            token_count = max(char_count // 4, word_count * 3 // 4)
            logger.debug(f"Подсчет токенов для TTS модели {model}: {token_count} токенов")
            return token_count
            
        if model in SPEECH_TO_TEXT_MODELS:
            # Whisper и другие STT модели работают с аудио, поэтому возвращаем 0
            # Для них токены считаются отдельно по длительности аудио
            return 0
        
        # Выбираем энкодер в зависимости от модели
        encoder_name = "gpt-4"  # Дефолтный энкодер
        
        # OpenAI модели
        if model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini", "o3-mini"]:
            encoder_name = "gpt-4"  # Для новых моделей OpenAI используем gpt-4 токенизатор
        
        # Модели Anthropic (Claude)
        elif model.startswith("claude"):
            encoder_name = "cl100k_base"  # Claude использует токенизатор, похожий на cl100k
        
        # Модели Mistral
        elif model.startswith("mistral") or "mixtral" in model:
            encoder_name = "gpt-4"  # Для Mistral используем OpenAI токенизатор
        
        # Модели LLaMA
        elif "llama" in model.lower():
            encoder_name = "gpt-4"  # LLaMA близок к токенизации GPT
        
        # Модели Google (Gemini)
        elif model.startswith("gemini"):
            encoder_name = "cl100k_base"  # Gemini ближе к cl100k
            
        # Другие модели, основанные на трансформерах
        elif any(m in model.lower() for m in ["command", "deepseek", "grok", "falcon", "mpt"]):
            encoder_name = "gpt-4"  # Большинство моделей близки к GPT токенизации
            
        # Получаем токенизатор и считаем токены
        try:
            # Сначала пробуем получить энкодер по имени модели
            encoding = tiktoken.encoding_for_model(encoder_name)
        except KeyError:
            # Если не удалось, используем базовый энкодер cl100k_base
            encoding = tiktoken.get_encoding("cl100k_base")
            
        tokens = encoding.encode(sentence)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Ошибка при подсчете токенов для модели {model}: {str(e)}. Используем приблизительную оценку.")
        # Приблизительно оцениваем количество токенов как 3/4 количества символов
        return len(sentence) * 3 // 4

def api_request(req_method, url, headers=None, requester_ip=None, data=None,
                files=None, stream=False, timeout=None, json=None, **kwargs):
    """
    Выполняет HTTP-запрос к API с нормализацией URL и обработкой ошибок.
    
    Args:
        req_method (str): Метод запроса (GET, POST, и т.д.)
        url (str): URL для запроса
        headers (dict, optional): Заголовки запроса
        requester_ip (str, optional): IP запрашивающего для логирования
        data (dict/str, optional): Данные для запроса
        files (dict, optional): Файлы для запроса
        stream (bool, optional): Флаг для потоковой передачи данных
        timeout (int, optional): Таймаут запроса в секундах
        json (dict, optional): JSON-данные для запроса
        **kwargs: Дополнительные параметры для requests
        
    Returns:
        Response: Объект ответа от API
    """
    req_url = url.strip()
    logger.debug(f"API request URL: {req_url}")

    # Формируем параметры запроса
    req_params = {k: v for k, v in {
        "headers": headers, 
        "data": data, 
        "files": files, 
        "stream": stream, 
        "json": json
    }.items() if v is not None}
    
    # Добавляем остальные параметры
    req_params.update(kwargs)

    # Определяем, является ли запрос операцией с изображениями
    is_image_operation = False
    if json and isinstance(json, dict):
        operation_type = json.get("type", "")
        if operation_type in [IMAGE_GENERATOR, IMAGE_VARIATOR]:
            is_image_operation = True
            logger.debug(f"Обнаружена операция с изображением: {operation_type}, используем расширенный таймаут")

    # Устанавливаем таймаут в зависимости от типа операции
    req_params["timeout"] = timeout or (MIDJOURNEY_TIMEOUT if is_image_operation else DEFAULT_TIMEOUT)

    # Выполняем запрос
    try:
        response = requests.request(req_method, req_url, **req_params)
        return response
    except Exception as e:
        logger.error(f"Ошибка API запроса: {str(e)}")
        raise

def set_response_headers(response):
    """
    Устанавливает стандартные заголовки для всех ответов API.
    
    Args:
        response: Объект ответа Flask
        
    Returns:
        Response: Модифицированный объект ответа с добавленными заголовками
    """
    response.headers.update({
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "X-Request-ID": str(uuid.uuid4()),
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Authorization, Content-Type, Accept"
    })
    return response

def create_session():
    """
    Создает новую сессию с оптимальными настройками для API запросов.
    
    Returns:
        Session: Настроенная сессия requests
    """
    session = requests.Session()

    # Настраиваем стратегию повторных попыток для всех запросов
    retry_strategy = requests.packages.urllib3.util.retry.Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Устанавливаем увеличенные таймауты по умолчанию для всей сессии
    # 30 секунд на подключение, 120 секунд на получение данных
    session.request = functools.partial(session.request, timeout=(60, 300))

    return session

def safe_temp_file(prefix, request_id=None):
    """
    Безопасно создает временный файл и гарантирует его удаление после использования.

    Args:
        prefix (str): Префикс для имени файла
        request_id (str, optional): ID запроса для логирования

    Returns:
        str: Путь к временному файлу
    """
    request_id = request_id or str(uuid.uuid4())[:8]
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")

    # Создаем временную директорию, если её нет
    os.makedirs(temp_dir, exist_ok=True)

    # Очищаем старые файлы (старше 1 часа)
    try:
        current_time = time.time()
        for old_file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, old_file)
            if os.path.isfile(file_path) and (current_time - os.path.getmtime(file_path) > 3600):
                try:
                    os.remove(file_path)
                    logger.debug(f"[{request_id}] Удален старый временный файл: {file_path}")
                except Exception as e:
                    logger.warning(f"[{request_id}] Не удалось удалить старый временный файл {file_path}: {str(e)}")
    except Exception as e:
        logger.warning(f"[{request_id}] Ошибка при очистке старых временных файлов: {str(e)}")

    # Создаем новый временный файл
    temp_file_path = os.path.join(temp_dir, f"{prefix}_{request_id}_{random_string}")
    return temp_file_path

def ERROR_HANDLER(code, model=None, key=None):
    """
    Обработчик ошибок в формате совместимом с OpenAI API.
    
    Args:
        code (int): Внутренний код ошибки
        model (str, optional): Имя модели (для ошибок, связанных с моделями)
        key (str, optional): API ключ (для ошибок аутентификации)
        
    Returns:
        tuple: (JSON с ошибкой, HTTP-код ответа)
    """
    # Словарь кодов ошибок
    error_codes = {
        1002: {
            "message": f"The model {model} does not exist.",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_found",
            "http_code": 400,
        },
        1020: {
            "message": f"Incorrect API key provided: {key}. You can find your API key at https://app.1min.ai/api.",
            "type": "authentication_error",
            "param": None,
            "code": "invalid_api_key",
            "http_code": 401,
        },
        1021: {
            "message": "Invalid Authentication",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
            "http_code": 401,
        },
        1212: {
            "message": f"Incorrect Endpoint. Please use the /v1/chat/completions endpoint.",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_supported",
            "http_code": 400,
        },
        1044: {
            "message": f"This model does not support image inputs.",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_supported",
            "http_code": 400,
        },
        1412: {
            "message": f"No message provided.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "invalid_request_error",
            "http_code": 400,
        },
        1423: {
            "message": f"No content in last message.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "invalid_request_error",
            "http_code": 400,
        },
    }
    
    # Получаем данные об ошибке или используем данные по умолчанию
    error_data = error_codes.get(code, {
        "message": f"Unknown error (code: {code})",
        "type": "unknown_error",
        "param": None,
        "code": None,
        "http_code": 400
    })
    
    # Удаляем http_code из данных ответа
    http_code = error_data.pop("http_code", 400)
    
    logger.error(f"Ошибка при обработке запроса пользователя. Код ошибки: {code}")
    return jsonify({"error": error_data}), http_code

def handle_options_request():
    """
    Обработчик OPTIONS запросов для CORS.
    
    Returns:
        tuple: (Объект ответа, HTTP-код ответа 204)
    """
    response = make_response()
    response.headers.update({
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,Authorization",
        "Access-Control-Allow-Methods": "POST, OPTIONS"
    })
    return response, 204

def split_text_for_streaming(text, chunk_size=6):
    """
    Разбивает текст на небольшие части для эмуляции потокового вывода.

    Args:
        text (str): Текст для разбивки
        chunk_size (int): Примерный размер частей в словах

    Returns:
        list: Список частей текста
    """
    if not text:
        return [""]

    # Разбиваем текст на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return [text]

    # Группируем предложения в чанки
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words_in_sentence = len(sentence.split())

        # Если текущий чанк пустой или добавление предложения не превысит лимит слов
        if not current_chunk or current_word_count + words_in_sentence <= chunk_size:
            current_chunk.append(sentence)
            current_word_count += words_in_sentence
        else:
            # Формируем чанк и начинаем новый
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = words_in_sentence

    # Добавляем последний чанк, если он не пустой
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks or [text]

def calculate_image_cost(model, mode=None, aspect_ratio=None):
    """
    Рассчитывает стоимость генерации изображения в условных единицах.
    
    Args:
        model (str): Название модели для генерации изображения
        mode (str, optional): Режим генерации (fast/relax, для Midjourney)
        aspect_ratio (str, optional): Соотношение сторон (для некоторых моделей)
        
    Returns:
        int: Стоимость в условных единицах
    """
    # Проверяем, является ли запрос генерацией изображения или вариацией
    if model in IMAGE_GENERATION_PRICES:
        price_data = IMAGE_GENERATION_PRICES.get(model)
        
        # Для моделей с разными режимами (Midjourney)
        if isinstance(price_data, dict) and mode:
            return price_data.get(mode.lower(), price_data.get("fast", 120000))
        
        # Для обычных моделей
        if isinstance(price_data, int):
            return price_data
            
        # Если не удалось определить цену
        logger.warning(f"Не удалось определить цену для генерации изображения с моделью {model}")
        return 120000  # Значение по умолчанию
    
    # Для вариаций изображений
    elif model in IMAGE_VARIATION_PRICES:
        price_data = IMAGE_VARIATION_PRICES.get(model)
        
        # Для моделей с разными режимами (Midjourney)
        if isinstance(price_data, dict) and mode:
            return price_data.get(mode.lower(), price_data.get("fast", 150000))
        
        # Для обычных моделей
        if isinstance(price_data, int):
            return price_data
            
        # Если не удалось определить цену
        logger.warning(f"Не удалось определить цену для вариации изображения с моделью {model}")
        return 150000  # Значение по умолчанию
    
    # Для неизвестной модели
    logger.warning(f"Неизвестная модель {model} для расчета стоимости изображения")
    return 120000  # Значение по умолчанию

def calculate_audio_cost(model, duration_seconds=None, text=None):
    """
    Рассчитывает стоимость аудио операций в условных единицах.
    
    Args:
        model (str): Название модели для аудио операции (TTS или STT)
        duration_seconds (float, optional): Длительность аудио в секундах (для STT)
        text (str, optional): Текст для озвучивания (для TTS)
        
    Returns:
        int: Стоимость в условных единицах
    """
    # Стоимость TTS моделей (преобразование текста в речь)
    if model in TEXT_TO_SPEECH_MODELS:
        if text is None:
            return 0
            
        # Получаем цену из констант
        base_price = TTS_PRICES.get(model, 15000)
        
        # Рассчитываем стоимость на основе длины текста (1000 символов = базовая стоимость)
        char_count = len(text)
        cost = max(1, char_count) * base_price // 1000
        
        logger.debug(f"Расчет стоимости TTS для модели {model}: текст {char_count} символов, стоимость {cost}")
        return cost
        
    # Стоимость STT моделей (распознавание речи)
    elif model in SPEECH_TO_TEXT_MODELS:
        if duration_seconds is None:
            return 0
            
        # Получаем цену из констант
        base_price = STT_PRICES.get(model, 6000)
        
        # Рассчитываем стоимость на основе длительности (60 секунд = базовая стоимость)
        cost = max(1, int(duration_seconds)) * base_price // 60
        
        logger.debug(f"Расчет стоимости STT для модели {model}: аудио {duration_seconds} секунд, стоимость {cost}")
        return cost
    
    # Для неизвестной модели
    logger.warning(f"Неизвестная аудио модель {model} для расчета стоимости")
    return 10000  # Значение по умолчанию
