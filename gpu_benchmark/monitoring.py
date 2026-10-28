import logging
from logging.handlers import RotatingFileHandler
import datetime
import time
import GPUtil
import psutil


def setup_monitoring_logger(log_file: str):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a rotating file handler which logs even debug messages
    handler = RotatingFileHandler(log_file, maxBytes=100 * 1024 * 1024, backupCount=1)
    handler.setLevel(logging.INFO)

    # Create a formatter and set the formatter for the handler
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    logger.info(
        "Timestamp,RAM_Usage (%),CPU_Usage (%),"
        + ",".join(
            [
                f"GPU{i}_Usage (%),GPU{i}_Memory_Usage (%),GPU{i}_Temperature (C)"
                for i in range(len(GPUtil.getGPUs()))
            ]
        )
    )

    return logger


def monitor_resources_logging(stop_event, logger, update_time: float = 1) -> None:
    while not stop_event.is_set():
        # timestamp with milliseconds precision
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        ram_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)

        gpu_usage = [gpu.load * 100 for gpu in GPUtil.getGPUs()]
        gpu_memory_usage = [gpu.memoryUtil * 100 for gpu in GPUtil.getGPUs()]
        gpu_temperature = [gpu.temperature for gpu in GPUtil.getGPUs()]

        # Construct the log message
        log_message = f"{timestamp},{ram_usage},{cpu_usage}," + ",".join(
            [
                f"{value}"
                for pair in zip(gpu_usage, gpu_memory_usage, gpu_temperature)
                for value in pair
            ]
        )

        # Log the message
        logger.info(log_message)

        time.sleep(update_time)
