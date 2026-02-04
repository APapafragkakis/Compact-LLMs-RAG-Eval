import logging
import sys
import time
import functools
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

def call_in_batches(func, array, batch_size=1, log_every=10):
    result = []
    total = len(array)

    for i in range(0, total, batch_size):
        batch = array[i:i + batch_size]
        tmp = func(batch)
        result.extend(tmp)
        if (i // batch_size) % log_every == 0 or i + batch_size >= total:
            logging.info(f"Processed {min(i + batch_size, total)}/{total}")
    return result


LOG_FILE = os.path.join(os.path.dirname(__file__), 'benchmark.log')

def benchmark(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start

        with open(LOG_FILE, 'a') as f:
            f.write(f"{func.__name__} took {duration:.6f} seconds\n")

        return result
    return wrapper