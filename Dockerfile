FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Backend sem tela: as figuras são salvas em Results/
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
