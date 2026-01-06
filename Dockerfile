FROM prefecthq/prefect:2-latest 

WORKDIR /app
COPY flow.py /app/flow.py

CMD ["python", "/app/flow.py"]