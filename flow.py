from prefect import flow, task
from prefect.logging import get_run_logger

@task(retries=2, retry_delay_seconds=1)
def check_random() -> None:
    import random
    logger = get_run_logger()
    value = random.randint(0, 1)
    logger.info(f"Generated value: {value}")
    if value < 0.5:
        logger.warning("Value is less than 0.5, retrain is required.")
        raise ValueError("Value is less than 0.5, retrying...")
    logger.info("Value is acceptable.")

@flow
def periodic_check() -> None:
    check_random()

if __name__ == "__main__":
    periodic_check.serve(
        name="every_10s",
        interval=10,
    )
