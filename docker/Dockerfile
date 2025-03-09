FROM python:3.12

RUN apt-get update && apt-get install -y \
    xvfb \
    python3-tk \
    x11-apps

WORKDIR /app

# Skopiowanie plików projektu
COPY . /app

# Instalacja zależności
RUN pip install --no-cache-dir -r requirements.txt

# Uruchomienie aplikacji GUI
CMD ["bash", "-c", "Xvfb :99 -screen 0 1024x768x24 & sleep 2 && export DISPLAY=:99 && python3 main.py"]

