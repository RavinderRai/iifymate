FROM python:3.11

WORKDIR /app

RUN pip install nltk
RUN python -m nltk.downloader stopwords

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader stopwords

COPY . .

EXPOSE 8080

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080"]  