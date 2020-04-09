import time

class Timer:
    times = {}

    @staticmethod
    def timer(name: str):
        def decorator(func):
            def wrapper(*args, **kw):
                start_time = time.time()
                ret = func(*args, **kw)
                time_spent = time.time() - start_time
                Timer.times[name] = time_spent * 1000
                return ret

            return wrapper

        return decorator
