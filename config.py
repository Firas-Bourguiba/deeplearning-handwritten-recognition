from os import environ as env
import multiprocessing

PORT = int(8081)
DEBUG_MODE = int(1)

# Gunicorn config
bind = ":" + str(PORT)
workers = multiprocessing.cpu_count() * 2 + 1
threads = 2 * multiprocessing.cpu_count()