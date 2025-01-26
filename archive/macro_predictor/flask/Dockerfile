FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python3 -c "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data')"
RUN python3 -c "import nltk; nltk.download('wordnet', download_dir='/usr/local/nltk_data')"
RUN python3 -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data')"

COPY . .

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]