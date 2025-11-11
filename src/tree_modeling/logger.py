"""Logger module."""

import logging
import os
import sys

# Get the main script's filename without extension dynamically
main_script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# Ensure logs directory exists
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define the log format
log_format = "%(levelname)s %(asctime)s - %(message)s"

# Create the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear any existing handlers to avoid conflicts
if logger.hasHandlers():
    logger.handlers.clear()

# Create handlers
log_filename = f"{log_dir}/{main_script_name}.log"
file_handler = logging.FileHandler(log_filename, mode="a")
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatters and add them to the handlers
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Avoid duplicate logs
logger.propagate = False

# Debugging logs
logger.info("Writing logs to %s", log_filename)

# Log attached handlers for verification
for handler in logger.handlers:
    logger.info("Handler: %s, Level: %s", type(handler).__name__, handler.level)

# Flush handlers to ensure messages are written immediately
for handler in logger.handlers:
    handler.flush()
