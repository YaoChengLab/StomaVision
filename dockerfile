FROM python:3.10-slim

WORKDIR /stomavision

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./app app

RUN pip3 install -r app/requirements.txt

RUN mkdir /stomavision/app/.streamlit

ARG API_TOKEN
RUN bash -c 'echo "INSTILL_API_TOKEN = \"$API_TOKEN\"" > app/.streamlit/secrets.toml'
RUN bash -c 'cat app/config.toml >> app/.streamlit/secrets.toml'

WORKDIR /stomavision/app

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "/stomavision/app/streamlit_app.py", "--server.port=8501", "--browser.gatherUsageStats=false"]
