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
