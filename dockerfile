FROM python:3.10-slim

WORKDIR /stomavision

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/heiruwu/StomaVision.git .

RUN pip3 install -r requirements.txt
RUN pip3 install -r app/requirements.txt

RUN mkdir /stomavision/app/.streamlit

ARG API_TOKEN
RUN bash -c 'echo "instill_api_key = \"$API_TOKEN\"" > app/.streamlit/secrets.toml'
RUN bash -c 'cat app/config.toml >> app/.streamlit/secrets.toml'

WORKDIR /stomavision/app

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "/stomavision/app/stomata_pipeline.py", "--server.port=8501", "--browser.gatherUsageStats=false"]
