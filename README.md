# 1min-relay

## Описание проекта
1min-relay - это сервер-посредник (прокси), реализующий API, совместимый с OpenAI API, для работы с различными AI-моделями через сервис 1min.ai. Он позволяет использовать клиентские приложения, поддерживающие OpenAI API, с моделями различных провайдеров через единый интерфейс.

## Особенности
- Полностью совместим с OpenAI API, включая chat/completions, images, audio и files
- Поддерживает большое количество моделей от различных провайдеров: OpenAI, Claude, Mistral, Google и других
- Работает с различными типами запросов: текстовыми, изображениями, аудио и файлами
- Реализует потоковую передачу данных (streaming)
- Имеет функцию ограничения запросов (rate limiting) с использованием Memcached
- Позволяет задать подмножество разрешенных моделей через переменные окружения

## Структура проекта
Проект имеет модульную структуру для облегчения разработки и поддержки:

```
1min-relay/
├── app.py                  # Основной файл приложения
├── utils/                  # Утилиты и вспомогательные функции
│   ├── __init__.py         # Экспорты функций и констант
│   ├── common.py           # Общие утилиты
│   ├── constants.py        # Константы проекта
│   └── memcached.py        # Функции для работы с Memcached
├── routes/                 # Маршруты API
│   ├── __init__.py         # Определение Blueprint и импорты
│   ├── text.py             # Маршруты для текстовых моделей
│   ├── images.py           # Маршруты для работы с изображениями
│   ├── audio.py            # Маршруты для работы с аудио
│   └── files.py            # Маршруты для работы с файлами
└── README.md               # Документация проекта
```

## Требования
- Python 3.7+
- Flask и связанные библиотеки
- Memcached (опционально для rate limiting)
- API ключ сервиса 1min.ai

## Установка и запуск

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Настройка переменных окружения
Создайте файл `.env` в корневой директории проекта:
```
PORT=5001
SUBSET_OF_ONE_MIN_PERMITTED_MODELS=gpt-4o-mini,mistral-nemo,claude-3-haiku-20240307,gemini-1.5-flash
PERMIT_MODELS_FROM_SUBSET_ONLY=false
```

### Запуск сервера
```bash
python app.py
```

После запуска сервер будет доступен по адресу `http://localhost:5001/`.

## Использование с клиентами OpenAI API
Большинство клиентов OpenAI API могут быть настроены для использования этого сервера путем указания базового URL:
```
http://localhost:5001/v1
```

При отправке запросов к API используйте свой API ключ 1min.ai в заголовке Authorization:
```
Authorization: Bearer your-1min-api-key
```

## Запуск с использованием Docker
Вы также можете запустить сервер в Docker-контейнере:

```bash
docker build -t 1min-relay .
docker run -d -p 5001:5001 --name 1min-relay-container -e PORT=5001 1min-relay
```

## Лицензия
[MIT License](LICENSE)
