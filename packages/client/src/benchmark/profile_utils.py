import cProfile
import pstats
import functools
import json
import time
from contextlib import contextmanager
from dataclasses import asdict

from .benchmark_config import BenchmarkConfig


LOG_FILE = "./profile_logs.jsonl"


def cprofile_function_and_save(profile_filename="profile.prof"):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()

            result = f(*args, **kwargs)

            pr.disable()
            pr.dump_stats(profile_filename)
            
            print(f"Profile data saved to {profile_filename}")
            return result

        return wrapper
    return decorator


@contextmanager
def time_and_log(section_name: str, benchmark_config: BenchmarkConfig):
    start_time = time.time()

    yield
    
    end_time = time.time()
    duration = end_time - start_time
    
    log_entry = asdict(benchmark_config)
    log_entry["section_name"] = section_name
    log_entry["duration"] = duration
    
    with open(LOG_FILE, "a") as log_file:
        print(json.dumps(log_entry), file=log_file)