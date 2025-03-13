import os

global LOGS_PATH, OUTPUT_PATH, LOGS_PATH

LOCAL_PATH = os.path.abspath(os.path.dirname(__file__))
OUTPUT_PATH = os.path.join(LOCAL_PATH, "output")
LOGS_PATH = os.path.join(OUTPUT_PATH, "logs")
LIBS_PATH = os.path.join(LOCAL_PATH, "libs")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)
