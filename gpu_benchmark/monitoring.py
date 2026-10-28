"""One NVML snapshot per interval, preserving legacy CSV columns."""
import csv
import logging
import time
import datetime
import psutil
import pynvml as nv


def setup_monitoring_logger(log_file):
    logger=logging.getLogger(log_file);logger.setLevel(logging.INFO);logger.propagate=False
    for handler in logger.handlers[:]:handler.close();logger.removeHandler(handler)
    logger.addHandler(logging.FileHandler(log_file))
    nv.nvmlInit()
    columns=['Timestamp','RAM_Usage (%)','CPU_Usage (%)']
    for i in range(nv.nvmlDeviceGetCount()):
        columns += [f'GPU{i}_Usage (%)',f'GPU{i}_Memory_Usage (%)',f'GPU{i}_Temperature (C)',f'GPU{i}_Power (W)',f'GPU{i}_Clock (MHz)']
    logger.info(','.join(columns));return logger


def monitor_resources_logging(stop_event, logger, update_time=1):
    handles=[nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
    while not stop_event.is_set():
        row=[datetime.datetime.now().isoformat(),psutil.virtual_memory().percent,psutil.cpu_percent()]
        for handle in handles:
            try:
                util=nv.nvmlDeviceGetUtilizationRates(handle);mem=nv.nvmlDeviceGetMemoryInfo(handle)
                row += [util.gpu,100*mem.used/mem.total,nv.nvmlDeviceGetTemperature(handle,nv.NVML_TEMPERATURE_GPU),nv.nvmlDeviceGetPowerUsage(handle)/1000,nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_SM)]
            except nv.NVMLError:row += ['']*5
        logger.info(','.join(map(str,row)))
        if stop_event.wait(update_time):break
