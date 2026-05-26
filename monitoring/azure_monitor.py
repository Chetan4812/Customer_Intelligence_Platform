"""
azure_monitor.py – Azure Monitor / OpenTelemetry Helper
=======================================================
Centralised helper to emit custom metrics from both services
to Azure Monitor via OpenTelemetry.

Usage (in your FastAPI service):
    from monitoring.azure_monitor import init_telemetry, record_prediction, record_rag_query
    init_telemetry()   # call once at startup
"""

from __future__ import annotations

import logging
import os
import time
from functools import wraps
from typing import Callable, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenTelemetry setup
# ---------------------------------------------------------------------------
_meter = None
_request_duration_histogram = None
_prediction_confidence_histogram = None
_request_counter = None
_error_counter = None


def init_telemetry(service_name: Optional[str] = None) -> None:
    """
    Initialise OpenTelemetry with Azure Monitor OTLP exporter.
    Set APPLICATIONINSIGHTS_CONNECTION_STRING env var to enable
    telemetry export to Azure. Works without the env var (no-op).
    """
    global _meter, _request_duration_histogram, _prediction_confidence_histogram
    global _request_counter, _error_counter

    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
    svc = service_name or os.getenv("SERVICE_NAME", "weel13-service")

    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": svc})
        readers = []

        if connection_string:
            # Azure Monitor OTLP exporter
            from azure.monitor.opentelemetry.exporter import AzureMonitorMetricExporter
            exporter = AzureMonitorMetricExporter(connection_string=connection_string)
            readers.append(PeriodicExportingMetricReader(exporter, export_interval_millis=30_000))
            log.info("Azure Monitor telemetry enabled for service '%s'.", svc)
        else:
            log.warning(
                "APPLICATIONINSIGHTS_CONNECTION_STRING not set. "
                "Metrics will only be logged locally."
            )

        provider = MeterProvider(resource=resource, metric_readers=readers)
        metrics.set_meter_provider(provider)
        _meter = metrics.get_meter(svc)

        # Define instruments
        _request_duration_histogram = _meter.create_histogram(
            "request_duration_ms",
            description="Request processing time in milliseconds",
            unit="ms",
        )
        _prediction_confidence_histogram = _meter.create_histogram(
            "prediction_confidence",
            description="XGBoost model predicted conversion probability",
            unit="1",
        )
        _request_counter = _meter.create_counter(
            "requests_total",
            description="Total number of requests served",
        )
        _error_counter = _meter.create_counter(
            "errors_total",
            description="Total number of request errors",
        )

    except ImportError as exc:
        log.warning("OpenTelemetry not installed (%s). Telemetry disabled.", exc)
    except Exception as exc:
        log.warning("Failed to initialise telemetry: %s", exc)


# ---------------------------------------------------------------------------
# Metric emitters
# ---------------------------------------------------------------------------

def record_request(endpoint: str, status: str = "ok", duration_ms: float = 0.0) -> None:
    """Record a completed request (duration + counter)."""
    attrs = {"endpoint": endpoint, "status": status}
    if _request_counter:
        _request_counter.add(1, attrs)
    if _request_duration_histogram:
        _request_duration_histogram.record(duration_ms, attrs)
    log.debug("Metric: endpoint=%s  status=%s  duration_ms=%.1f", endpoint, status, duration_ms)


def record_prediction(prob: float) -> None:
    """Record the conversion probability predicted by the ML model."""
    if _prediction_confidence_histogram:
        _prediction_confidence_histogram.record(prob, {"model": "xgboost_conversion"})
    log.debug("Metric: conversion_prob=%.4f", prob)


def record_error(endpoint: str) -> None:
    """Increment the error counter for an endpoint."""
    if _error_counter:
        _error_counter.add(1, {"endpoint": endpoint})


# ---------------------------------------------------------------------------
# Decorator helper
# ---------------------------------------------------------------------------

def timed_endpoint(endpoint_name: str) -> Callable:
    """
    Decorator that automatically records request duration and errors.

    Example:
        @timed_endpoint("predict")
        async def predict(request: PredictRequest): ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            status = "ok"
            try:
                return await func(*args, **kwargs)
            except Exception:
                status = "error"
                record_error(endpoint_name)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                record_request(endpoint_name, status, elapsed_ms)
        return wrapper
    return decorator
