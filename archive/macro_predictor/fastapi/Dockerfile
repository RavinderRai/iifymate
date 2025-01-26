FROM python:3.11

WORKDIR /app

RUN pip install nltk
RUN python -m nltk.downloader stopwords

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python3 -c "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data')"
RUN python3 -c "import nltk; nltk.download('wordnet', download_dir='/usr/local/nltk_data')"
RUN python3 -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data')"

COPY . .

EXPOSE 8080

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080"]  