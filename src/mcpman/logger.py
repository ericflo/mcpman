"""
Enhanced structured logging system for mcpman.

This module provides comprehensive logging capabilities including:
- Structured JSON logs with consistent fields
- Full HTTP request/response capture
- Complete tool execution details
- LLM interaction logging
- Performance metrics
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

# Configure default logger
DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(message)s"
DEFAULT_LOGGER_NAME = "mcpman"

# Log categories
CATEGORY_SYSTEM = "system"
CATEGORY_HTTP = "http"
CATEGORY_LLM = "llm"
CATEGORY_TOOL = "tool"
CATEGORY_VERIFICATION = "verification"
CATEGORY_EXECUTION = "execution_flow"
CATEGORY_PERFORMANCE = "performance"


class StructuredLogRecord(logging.LogRecord):
    """Enhanced LogRecord with structured data support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = None
        self.category = None
        self.payload = {}
        self.event_type = None


class StructuredLogger(logging.Logger):
    """Logger that creates StructuredLogRecord instances."""

    def makeRecord(
        self,
        name,
        level,
        fn,
        lno,
        msg,
        args,
        exc_info,
        func=None,
        extra=None,
        sinfo=None,
    ):
        """Create a StructuredLogRecord instance."""
        record = StructuredLogRecord(
            name, level, fn, lno, msg, args, exc_info, func, sinfo
        )
        if extra:
            for key, value in extra.items():
                setattr(record, key, value)
        return record


class JSONFormatter(logging.Formatter):
    """Format log records as JSON objects."""

    def format(self, record):
        """Format the log record as a JSON string."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add source information
        log_data["source"] = f"{record.pathname}:{record.lineno}"
        log_data["process_id"] = os.getpid()

        # Add structured data if available
        if hasattr(record, "run_id") and record.run_id:
            log_data["run_id"] = record.run_id

        if hasattr(record, "event_type") and record.event_type:
            log_data["event_type"] = record.event_type

        if hasattr(record, "category") and record.category:
            log_data["category"] = record.category

        if hasattr(record, "payload") and record.payload:
            log_data["payload"] = record.payload

        # Special handling for executions
        if hasattr(record, "execution") and record.execution:
            log_data["execution"] = record.execution

        return json.dumps(log_data)


# Configure logging system
logging.setLoggerClass(StructuredLogger)


def get_logger(name=DEFAULT_LOGGER_NAME) -> StructuredLogger:
    """Get a configured logger instance."""
    return logging.getLogger(name)


def setup_logging(
    log_file=None,
    level=DEFAULT_LOG_LEVEL,
    run_id=None,
    quiet_console=True,
    output_only=False,
):
    """
    Set up the logging system with handlers for console and file output.

    Args:
        log_file: Path to log file
        level: Log level
        run_id: Optional run ID for tracking
        quiet_console: Whether to suppress detailed logs from console
        output_only: Whether to suppress all console logs except the final output

    Returns:
        The run ID for this session
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create a unique run ID if not provided
    if not run_id:
        run_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.getpid()}"

    # Console handler setup
    console_handler = logging.StreamHandler(sys.stderr)

    if output_only:
        # In output_only mode, suppress all console logs except critical errors
        # This effectively makes the console only show the final output
        console_handler.setLevel(logging.CRITICAL)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
    elif quiet_console:
        # In quiet mode (but not output_only):
        # Only show warnings and errors with simple format
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
    else:
        # In debug mode, show all logs with JSON formatting
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(JSONFormatter())

    root_logger.addHandler(console_handler)

    # File handler if log file provided - always full JSON details
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        file_handler.setLevel(level)  # File gets all logs at specified level
        root_logger.addHandler(file_handler)

        # Log the setup - this will go to both console and file
        # Create a special logger just for this initial message to console
        setup_logger = logging.getLogger("setup")
        setup_logger.setLevel(logging.INFO)
        setup_logger.addHandler(file_handler)

        # Log setup info to file only
        setup_logger.info(
            f"Logging to file: {log_file}",
            extra={
                "run_id": run_id,
                "category": CATEGORY_SYSTEM,
                "payload": {"taskName": f"Task-{os.getpid()}", "log_file": log_file},
            },
        )

    return run_id


# Enhanced logging methods
def log_http_request(logger, url, method, headers, body=None, extra=None):
    """Log an HTTP request."""
    payload = {
        "url": url,
        "method": method,
        "headers": sanitize_headers(headers),
    }

    if body:
        payload["body"] = body

    if extra:
        payload.update(extra)

    logger.info(
        f"HTTP Request: {method} {url}",
        extra={
            "category": CATEGORY_HTTP,
            "event_type": "http_request",
            "payload": payload,
        },
    )


def log_http_response(
    logger, url, status_code, headers, body=None, response_time=None, extra=None
):
    """Log an HTTP response."""
    payload = {
        "url": url,
        "status_code": status_code,
        "headers": sanitize_headers(headers),
    }

    if body:
        payload["body"] = body

    if response_time:
        payload["response_time"] = response_time

    if extra:
        payload.update(extra)

    # Format the log message based on whether response_time is available
    if response_time is not None:
        log_message = f"HTTP Response: {status_code} from {url} ({response_time:.2f}s)"
    else:
        log_message = f"HTTP Response: {status_code} from {url}"

    logger.info(
        log_message,
        extra={
            "category": CATEGORY_HTTP,
            "event_type": "http_response",
            "payload": payload,
        },
    )


def log_llm_request(
    logger, provider, model, messages, tools=None, temperature=None, extra=None
):
    """Log a request to an LLM."""
    payload = {
        "provider": provider,
        "model": model,
        "messages": messages,
    }

    if tools:
        payload["tools"] = tools

    if temperature is not None:
        payload["temperature"] = temperature

    if extra:
        payload.update(extra)

    logger.info(
        f"LLM Request: {provider}/{model}",
        extra={
            "category": CATEGORY_LLM,
            "event_type": "llm_request",
            "payload": payload,
        },
    )


def log_llm_response(logger, provider, model, response, response_time=None, extra=None):
    """Log a response from an LLM."""
    payload = {
        "provider": provider,
        "model": model,
        "response": response,
    }

    if response_time:
        payload["response_time"] = response_time

    if extra:
        payload.update(extra)

    # Format the log message based on whether response_time is available
    if response_time is not None:
        log_message = f"LLM Response: {provider}/{model} ({response_time:.2f}s)"
    else:
        log_message = f"LLM Response: {provider}/{model}"

    logger.info(
        log_message,
        extra={
            "category": CATEGORY_LLM,
            "event_type": "llm_response",
            "payload": payload,
        },
    )


def log_api_error(
    logger,
    provider,
    error_type,
    status_code=None,
    error_details=None,
    run_id=None,
    extra=None,
):
    """Log an API error from an LLM provider in a consistent way."""
    payload = {
        "provider": provider,
        "error_type": error_type,
    }

    if status_code:
        payload["status_code"] = status_code

    if error_details:
        payload["error_details"] = error_details

    if run_id:
        payload["run_id"] = run_id

    if extra:
        payload.update(extra)

    error_message = f"{provider.capitalize()} API Error"
    if status_code:
        error_message += f" ({status_code})"
    if error_type:
        error_message += f": {error_type}"

    logger.error(
        error_message,
        extra={"category": CATEGORY_LLM, "event_type": "llm_error", "payload": payload},
    )


class LLMClientLogger:
    """
    Helper class for standardized logging across all LLM client implementations.

    This provides a consistent interface for logging operations in all LLM clients
    without duplicating code.
    """

    def __init__(self, provider_name, model_name=None):
        """
        Initialize the logger with provider and model information.

        Args:
            provider_name: Name of the LLM provider (e.g., 'openai', 'anthropic')
            model_name: Name of the model being used
        """
        self.provider = provider_name
        self.model = model_name
        self.logger = get_logger()

    def set_model(self, model_name):
        """Set or update the model name."""
        self.model = model_name

    def log_request(
        self, messages, tools=None, temperature=None, run_id=None, extra=None
    ):
        """Log an LLM request in a standardized format."""
        request_extra = {"run_id": run_id} if run_id else {}
        if extra:
            request_extra.update(extra)

        log_llm_request(
            self.logger,
            provider=self.provider,
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            extra=request_extra,
        )

    def log_response(self, response, response_time=None, run_id=None, extra=None):
        """Log an LLM response in a standardized format."""
        response_extra = {"run_id": run_id} if run_id else {}
        if extra:
            response_extra.update(extra)

        log_llm_response(
            self.logger,
            provider=self.provider,
            model=self.model,
            response=response,
            response_time=response_time,
            extra=response_extra,
        )

    def log_http_req(self, url, method, headers, body=None, run_id=None, extra=None):
        """Log an HTTP request in a standardized format."""
        request_extra = (
            {"run_id": run_id, "provider": self.provider, "model": self.model}
            if run_id
            else {"provider": self.provider, "model": self.model}
        )
        if extra:
            request_extra.update(extra)

        log_http_request(
            self.logger,
            url=url,
            method=method,
            headers=headers,
            body=body,
            extra=request_extra,
        )

    def log_http_resp(
        self,
        url,
        status_code,
        headers,
        body=None,
        response_time=None,
        run_id=None,
        extra=None,
    ):
        """Log an HTTP response in a standardized format."""
        response_extra = (
            {"run_id": run_id, "provider": self.provider, "model": self.model}
            if run_id
            else {"provider": self.provider, "model": self.model}
        )
        if extra:
            response_extra.update(extra)

        log_http_response(
            self.logger,
            url=url,
            status_code=status_code,
            headers=headers,
            body=body,
            response_time=response_time,
            extra=response_extra,
        )

    def log_error(
        self, error_type, status_code=None, error_details=None, run_id=None, extra=None
    ):
        """Log an API error in a standardized format."""
        log_api_error(
            self.logger,
            provider=self.provider,
            error_type=error_type,
            status_code=status_code,
            error_details=error_details,
            run_id=run_id,
            extra=extra,
        )


def log_tool_call(logger, tool_name, parameters, taskName=None, extra=None):
    """Log a tool call."""
    payload = {
        "tool": tool_name,
        "parameters": parameters,
    }

    if taskName:
        payload["taskName"] = taskName

    if extra:
        payload.update(extra)

    logger.info(
        f"Executing tool: {tool_name}",
        extra={
            "category": CATEGORY_TOOL,
            "event_type": "tool_call",
            "payload": payload,
        },
    )


def log_tool_response(
    logger, tool_name, response, success=True, time_ms=None, taskName=None, extra=None
):
    """Log a tool response."""
    payload = {
        "tool": tool_name,
        "success": success,
        "response": response,
    }

    if time_ms:
        payload["time_ms"] = time_ms

    if taskName:
        payload["taskName"] = taskName

    if extra:
        payload.update(extra)

    logger.info(
        f"Tool response: {tool_name}",
        extra={
            "category": CATEGORY_TOOL,
            "event_type": "tool_response",
            "payload": payload,
        },
    )


def log_verification(logger, is_complete, feedback=None, taskName=None, extra=None):
    """Log a verification result."""
    payload = {
        "is_complete": is_complete,
    }

    if feedback:
        payload["feedback"] = feedback

    if taskName:
        payload["taskName"] = taskName

    if extra:
        payload.update(extra)

    logger.info(
        "Task verification result",
        extra={
            "category": CATEGORY_VERIFICATION,
            "event_type": "verification",
            "payload": payload,
        },
    )


def log_execution_start(logger, taskName=None, extra=None):
    """Log the start of execution."""
    payload = {}

    if taskName:
        payload["taskName"] = taskName

    if extra:
        payload.update(extra)

    logger.info(
        "MCPMan execution started",
        extra={
            "category": CATEGORY_EXECUTION,
            "event_type": "execution_start",
            "payload": payload,
        },
    )


def log_execution_complete(logger, config, taskName=None, extra=None):
    """Log the completion of execution."""
    payload = {}

    if taskName:
        payload["taskName"] = taskName

    if extra:
        payload.update(extra)

    # Add configuration details
    for key, value in config.items():
        payload[key] = value

    logger.info(
        "MCPMan execution completed",
        extra={
            "category": CATEGORY_EXECUTION,
            "event_type": "execution_complete",
            "payload": payload,
            "execution": {k: v for k, v in config.items() if k != "run_id"},
        },
    )


def sanitize_headers(headers):
    """Remove sensitive information from headers."""
    if not headers:
        return {}

    sanitized = dict(headers)
    sensitive_keys = [
        "authorization",
        "x-api-key",
        "api-key",
        "x-openai-api-key",
        "anthropic-api-key",
    ]

    for key in sanitized:
        if key.lower() in sensitive_keys:
            sanitized[key] = "[REDACTED]"

    return sanitized


class LoggingTimer:
    """Context manager for timing operations."""

    def __init__(
        self, logger, operation_name, category=CATEGORY_PERFORMANCE, extra=None
    ):
        self.logger = logger
        self.operation_name = operation_name
        self.category = category
        self.extra = extra or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time

        payload = {
            "operation": self.operation_name,
            "elapsed_seconds": elapsed,
            **self.extra,
        }

        self.logger.info(
            f"Operation timing: {self.operation_name} took {elapsed:.4f}s",
            extra={
                "category": self.category,
                "event_type": "timing",
                "payload": payload,
            },
        )

        return False  # Don't suppress exceptions
