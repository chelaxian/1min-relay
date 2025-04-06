# routes/audio.py

# Импортируем только необходимые модули
from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import ERROR_HANDLER, handle_options_request, set_response_headers, create_session, api_request, safe_temp_file, calculate_token
from . import app, limiter, MEMORY_STORAGE  # Импортируем app, limiter и MEMORY_STORAGE из модуля routes

# Маршруты для работы с аудио
@app.route("/v1/audio/transcriptions", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def audio_transcriptions():
    """
    Route for converting speech into text (analogue of Openai Whisper API)
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: /v1/audio/transcriptions")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    # Checking the availability of the Audio file
    if "file" not in request.files:
        logger.error(f"[{request_id}] No audio file provided")
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]
    model = request.form.get("model", "whisper-1")
    response_format = request.form.get("response_format", "text")
    language = request.form.get("language", None)
    temperature = request.form.get("temperature", 0)

    logger.info(f"[{request_id}] Processing audio transcription with model {model}")

    try:
        # We create a new session for loading audio
        session = create_session()
        headers = {"API-KEY": api_key}

        # Audio loading in 1min.ai
        files = {"asset": (audio_file.filename, audio_file, "audio/mpeg")}

        try:
            asset_response = session.post(
                ONE_MIN_ASSET_URL, files=files, headers=headers
            )
            logger.debug(
                f"[{request_id}] Audio upload response status code: {asset_response.status_code}"
            )

            if asset_response.status_code != 200:
                session.close()
                return (
                    jsonify(
                        {
                            "error": asset_response.json().get(
                                "error", "Failed to upload audio"
                            )
                        }
                    ),
                    asset_response.status_code,
                )

            audio_path = asset_response.json()["fileContent"]["path"]
            logger.debug(f"[{request_id}] Successfully uploaded audio: {audio_path}")
        finally:
            session.close()

        # Подготовка моделей для перебора
        models_to_try = []
        
        # Если запрошена конкретная модель, пробуем её первой
        if model in SPEECH_TO_TEXT_MODELS:
            models_to_try = [model]
            # Добавляем остальные модели из списка, кроме уже добавленной
            models_to_try.extend([m for m in SPEECH_TO_TEXT_MODELS if m != model])
        else:
            # Если запрошенная модель не в списке, используем все модели из списка
            models_to_try = SPEECH_TO_TEXT_MODELS
            
        logger.debug(f"[{request_id}] Will try these models in order: {models_to_try}")
        
        last_error = None
        
        # Перебираем модели, пока одна не сработает
        for current_model in models_to_try:
            try:
                # We form Payload for request Speech_to_text
                payload = {
                    "type": "SPEECH_TO_TEXT",
                    "model": current_model,
                    "promptObject": {
                        "audioUrl": audio_path,
                        "response_format": response_format,
                    },
                }

                # Add additional parameters if they are provided
                if language:
                    payload["promptObject"]["language"] = language

                if temperature is not None:
                    payload["promptObject"]["temperature"] = float(temperature)

                headers = {"API-KEY": api_key, "Content-Type": "application/json"}

                # We send a request
                logger.debug(
                    f"[{request_id}] Sending transcription request to {ONE_MIN_API_URL} with model {current_model}"
                )
                response = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
                logger.debug(
                    f"[{request_id}] Transcription response status code: {response.status_code} for model {current_model}"
                )

                # Если запрос успешный, обрабатываем ответ
                if response.status_code == 200:
                    # We convert the answer to the Openai API format
                    one_min_response = response.json()

                    # We extract the text from the answer
                    result_text = ""

                    if (
                            "aiRecord" in one_min_response
                            and "aiRecordDetail" in one_min_response["aiRecord"]
                    ):
                        result_text = one_min_response["aiRecord"]["aiRecordDetail"].get(
                            "resultObject", [""]
                        )[0]
                    elif "resultObject" in one_min_response:
                        result_text = (
                            one_min_response["resultObject"][0]
                            if isinstance(one_min_response["resultObject"], list)
                            else one_min_response["resultObject"]
                        )

                    # Check if the result_text json is
                    try:
                        # If result_text is a json line, we rush it
                        if result_text and result_text.strip().startswith("{"):
                            parsed_json = json.loads(result_text)
                            # If Parsed_json has a "Text" field, we use its value
                            if "text" in parsed_json:
                                result_text = parsed_json["text"]
                                logger.debug(f"[{request_id}] Extracted inner text from JSON: {result_text}")
                    except (json.JSONDecodeError, TypeError, ValueError):
                        # If it was not possible to steam like JSON, we use it as it is
                        logger.debug(f"[{request_id}] Using result_text as is: {result_text}")
                        pass

                    if not result_text:
                        logger.error(
                            f"[{request_id}] Could not extract transcription text from API response"
                        )
                        continue  # Пробуем следующую модель

                    # The most simple and reliable response format
                    logger.info(f"[{request_id}] Successfully processed audio transcription with model {current_model}: {result_text}")

                    # Create json strictly in Openai API format
                    response_data = {"text": result_text}

                    # Add Cors headlines
                    response = jsonify(response_data)
                    response.headers["Access-Control-Allow-Origin"] = "*"
                    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"

                    return response
                else:
                    # Сохраняем ошибку и пробуем следующую модель
                    last_error = response
                    logger.warning(
                        f"[{request_id}] Model {current_model} failed with status {response.status_code}. Trying next model if available."
                    )
                    # Не делаем continue здесь - падаем на следующую итерацию цикла автоматически
            
            except Exception as e:
                # Записываем ошибку в лог и пробуем следующую модель
                logger.error(
                    f"[{request_id}] Error with model {current_model}: {str(e)}"
                )
                last_error = e
                # Продолжаем цикл, пробуя следующую модель
        
        # Если мы дошли до сюда, значит ни одна модель не сработала
        logger.error(f"[{request_id}] All available models failed: {models_to_try}")
        
        # Если ни одна модель не сработала, возвращаем последнюю ошибку
        if isinstance(last_error, requests.Response):
            if last_error.status_code == 401:
                return ERROR_HANDLER(1020, key=api_key)
            logger.error(
                f"[{request_id}] All models failed. Last error: {last_error.text[:200]}"
            )
            error_text = "No available providers at the moment"
            try:
                error_json = last_error.json()
                if "error" in error_json:
                    error_text = error_json["error"]
            except:
                pass
            
            return (
                jsonify({"error": f"All available models failed. {error_text}"}),
                last_error.status_code,
            )
        else:
            logger.error(f"[{request_id}] All models failed. Last error: {str(last_error)}")
            return jsonify({"error": f"All available models failed. {str(last_error)}"}), 500

    except Exception as e:
        logger.error(f"[{request_id}] Exception during transcription request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/audio/translations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def audio_translations():
    """
    Route for translating audio to text (analogue Openai Whisper API)
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: /v1/audio/translations")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    # Checking the availability of the Audio file
    if "file" not in request.files:
        logger.error(f"[{request_id}] No audio file provided")
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]
    model = request.form.get("model", "whisper-1")
    response_format = request.form.get("response_format", "text")
    temperature = request.form.get("temperature", 0)

    logger.info(f"[{request_id}] Processing audio translation with model {model}")

    try:
        # We create a new session for loading audio
        session = create_session()
        headers = {"API-KEY": api_key}

        # Audio loading in 1min.ai
        files = {"asset": (audio_file.filename, audio_file, "audio/mpeg")}

        try:
            asset_response = session.post(
                ONE_MIN_ASSET_URL, files=files, headers=headers
            )
            logger.debug(
                f"[{request_id}] Audio upload response status code: {asset_response.status_code}"
            )

            if asset_response.status_code != 200:
                session.close()
                return (
                    jsonify(
                        {
                            "error": asset_response.json().get(
                                "error", "Failed to upload audio"
                            )
                        }
                    ),
                    asset_response.status_code,
                )

            audio_path = asset_response.json()["fileContent"]["path"]
            logger.debug(f"[{request_id}] Successfully uploaded audio: {audio_path}")
        finally:
            session.close()

        # Подготовка моделей для перебора
        models_to_try = []
        
        # Если запрошена конкретная модель, пробуем её первой
        if model in SPEECH_TO_TEXT_MODELS:
            models_to_try = [model]
            # Добавляем остальные модели из списка, кроме уже добавленной
            models_to_try.extend([m for m in SPEECH_TO_TEXT_MODELS if m != model])
        else:
            # Если запрошенная модель не в списке, используем все модели из списка
            models_to_try = SPEECH_TO_TEXT_MODELS
            
        logger.debug(f"[{request_id}] Will try these models: {models_to_try}")
        
        last_error = None
        
        # Перебираем модели, пока одна не сработает
        for current_model in models_to_try:
            try:
                # We form Payload for request Audio_Translator
                payload = {
                    "type": "AUDIO_TRANSLATOR",
                    "model": current_model,
                    "promptObject": {
                        "audioUrl": audio_path,
                        "response_format": response_format,
                        "temperature": float(temperature),
                    },
                }

                headers = {"API-KEY": api_key, "Content-Type": "application/json"}

                # We send a request
                logger.debug(f"[{request_id}] Sending translation request to {ONE_MIN_API_URL} with model {current_model}")
                response = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
                logger.debug(
                    f"[{request_id}] Translation response status code: {response.status_code} for model {current_model}"
                )

                # Если запрос успешный, обрабатываем ответ
                if response.status_code == 200:
                    # We convert the answer to the Openai API format
                    one_min_response = response.json()

                    # We extract the text from the answer
                    result_text = ""

                    if (
                            "aiRecord" in one_min_response
                            and "aiRecordDetail" in one_min_response["aiRecord"]
                    ):
                        result_text = one_min_response["aiRecord"]["aiRecordDetail"].get(
                            "resultObject", [""]
                        )[0]
                    elif "resultObject" in one_min_response:
                        result_text = (
                            one_min_response["resultObject"][0]
                            if isinstance(one_min_response["resultObject"], list)
                            else one_min_response["resultObject"]
                        )

                    if not result_text:
                        logger.error(
                            f"[{request_id}] Could not extract translation text from API response"
                        )
                        continue  # Пробуем следующую модель

                    # The most simple and reliable response format
                    logger.info(f"[{request_id}] Successfully processed audio translation with model {current_model}: {result_text}")

                    # Create json strictly in Openai API format
                    response_data = {"text": result_text}

                    # Add Cors headlines
                    response = jsonify(response_data)
                    response.headers["Access-Control-Allow-Origin"] = "*"
                    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"

                    return response
                else:
                    # Сохраняем ошибку и пробуем следующую модель
                    last_error = response
                    logger.warning(
                        f"[{request_id}] Model {current_model} failed with status {response.status_code}. Trying next model."
                    )
                    continue
            
            except Exception as e:
                # Записываем ошибку в лог и пробуем следующую модель
                logger.error(
                    f"[{request_id}] Error with model {current_model}: {str(e)}"
                )
                last_error = e
                continue
        
        # Если ни одна модель не сработала, возвращаем последнюю ошибку
        if isinstance(last_error, requests.Response):
            if last_error.status_code == 401:
                return ERROR_HANDLER(1020, key=api_key)
            logger.error(
                f"[{request_id}] All models failed. Last error: {last_error.text[:200]}"
            )
            return (
                jsonify({"error": last_error.json().get("error", "All available models failed")}),
                last_error.status_code,
            )
        else:
            logger.error(f"[{request_id}] All models failed. Last error: {str(last_error)}")
            return jsonify({"error": str(last_error)}), 500

    except Exception as e:
        logger.error(f"[{request_id}] Exception during translation request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/audio/speech", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def text_to_speech():
    """
    Route for converting text into speech (analogue Openai TTS API)
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = request.args.get('request_id', str(uuid.uuid4())[:8])
    logger.info(f"[{request_id}] Received request: /v1/audio/speech")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    # We get data data
    request_data = {}
    
    # We check the availability of data in Memcached if the request has been redirected
    if 'request_id' in request.args and 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
        tts_session_key = f"tts_request_{request.args.get('request_id')}"
        try:
            session_data = safe_memcached_operation('get', tts_session_key)
            if session_data:
                if isinstance(session_data, str):
                    request_data = json.loads(session_data)
                elif isinstance(session_data, bytes):
                    request_data = json.loads(session_data.decode('utf-8'))
                else:
                    request_data = session_data
                    
                # We delete data from the cache, they are no longer needed
                safe_memcached_operation('delete', tts_session_key)
                logger.debug(f"[{request_id}] Retrieved TTS request data from memcached")
        except Exception as e:
            logger.error(f"[{request_id}] Error retrieving TTS session data: {str(e)}")
    
    # If the data is not found in Memcache, we try to get them from the query body
    if not request_data and request.is_json:
        request_data = request.json
        
    model = request_data.get("model", "tts-1")
    input_text = request_data.get("input", "")
    voice = request_data.get("voice", "alloy")
    response_format = request_data.get("response_format", "mp3")
    speed = request_data.get("speed", 1.0)

    logger.info(f"[{request_id}] Processing TTS request with model {model}")
    logger.debug(f"[{request_id}] Text input: {input_text[:100]}..." if input_text and len(input_text) > 100 else f"[{request_id}] Text input: {input_text}")

    if not input_text:
        logger.error(f"[{request_id}] No input text provided")
        return jsonify({"error": "No input text provided"}), 400

    try:
        # We form Payload for request_to_Speech
        payload = {
            "type": "TEXT_TO_SPEECH",
            "model": model,
            "promptObject": {
                "text": input_text,
                "voice": voice,
                "response_format": response_format,
                "speed": speed
            }
        }

        headers = {"API-KEY": api_key, "Content-Type": "application/json"}

        # We send a request
        logger.debug(f"[{request_id}] Sending TTS request to {ONE_MIN_API_URL}")
        response = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
        logger.debug(f"[{request_id}] TTS response status code: {response.status_code}")

        if response.status_code != 200:
            if response.status_code == 401:
                return ERROR_HANDLER(1020, key=api_key)
            logger.error(f"[{request_id}] Error in TTS response: {response.text[:200]}")
            return (
                jsonify({"error": response.json().get("error", "Unknown error")}),
                response.status_code,
            )

        # We process the answer
        one_min_response = response.json()

        try:
            # We get a URL audio from the answer
            audio_url = ""

            if "aiRecord" in one_min_response and "aiRecordDetail" in one_min_response["aiRecord"]:
                result_object = one_min_response["aiRecord"]["aiRecordDetail"].get("resultObject", "")
                if isinstance(result_object, list) and result_object:
                    audio_url = result_object[0]
                else:
                    audio_url = result_object
            elif "resultObject" in one_min_response:
                result_object = one_min_response["resultObject"]
                if isinstance(result_object, list) and result_object:
                    audio_url = result_object[0]
                else:
                    audio_url = result_object

            if not audio_url:
                logger.error(f"[{request_id}] Could not extract audio URL from API response")
                return jsonify({"error": "Could not extract audio URL"}), 500

            # We get audio data by URL
            audio_response = api_request("GET", f"https://asset.1min.ai/{audio_url}")

            if audio_response.status_code != 200:
                logger.error(f"[{request_id}] Failed to download audio: {audio_response.status_code}")
                return jsonify({"error": "Failed to download audio"}), 500

            # We return the audio to the client
            logger.info(f"[{request_id}] Successfully generated speech audio")

            # We create an answer with audio and correct MIME-type
            content_type = "audio/mpeg" if response_format == "mp3" else f"audio/{response_format}"
            response = make_response(audio_response.content)
            response.headers["Content-Type"] = content_type
            set_response_headers(response)

            return response

        except Exception as e:
            logger.error(f"[{request_id}] Error processing TTS response: {str(e)}")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"[{request_id}] Exception during TTS request: {str(e)}")
        return jsonify({"error": str(e)}), 500
