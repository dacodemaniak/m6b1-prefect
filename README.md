# Implementing "prefect" monitoring solution

## Install prefect
```shell
pip install prefect
```

## Run prefect

```shell
prefect server start
```

Then open a new terminal and run :

```shell
python flow.py
```

## Main parts

```python
@task(retries=2, retry_delay_seconds=1)
def check_random() -> None:
    import random
    logger = get_run_logger()
    value = random.randint(0, 1)
    logger.info(f"Generated value: {value}")
    if value < 0.5:
        logger.warning(f"Value {value} is less than 0.5, retrain is required.")
        raise ValueError("Value is less than 0.5, retrying...")
    logger.info(f"Value {value} is acceptable.")
```

Generate a random value, then check if the value is under 0.5. If it is raise an error, logging the value.

## Viewing

```shell
http://localhost:42000
```

## Dockerization

A containered version is available :

```
docker compose up -d
```

[UPDATE - 2026-01-07]

## First model training

```shell
python ./scripts/train_and_save.py
```

**Run only once** just build a simple MNIST Keras model ready for improvement

## Generate sqlite database

```shell
python ./scripts/sqlite_init.py
```

It will generate a simple table to log corrections.

## Mount API and IHM

```shell
uvicorn api.main:app --reload --port 8000
```
API will listen on http://127.0.0.1:8000

```shell
streamlit run ihm/app.py
```

IHM will exposed on default streamlit port

## Retrain orchestration

```shell
# Prefect server
prefect server start

# Local deployment
prefect deployment build scripts/train_pipeline.py:mnist_retraining_flow -n "Hourly-Retrain" --interval 3600

# Apply deployment
prefect deployment apply mnist_retraining_flow-deployment.yaml

# Launch agent
prefect worker start --pool 'default-agent-pool'

