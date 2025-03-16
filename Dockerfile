FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install flask requests waitress mistral_common flask-limiter limits[memcached] coloredlogs printedcolors pymemcache docx2txt python-docx python-dateutil

COPY . .

EXPOSE 5001

CMD ["python", "main.py"]
