import time


class Timer:
    times = {}
    avg_times = {}

    @staticmethod
    def timer(name: str):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                ret = func(*args, **kwargs)
                time_spent = time.time() - start_time
                Timer.times[name] = time_spent * 1000
                return ret

            return wrapper

        return decorator

    @staticmethod
    def avg_timer(name: str):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                ret = func(*args, **kwargs)
                time_spent = time.time() - start_time
                if name not in Timer.avg_times:
                    Timer.avg_times[name] = (0, 0)
                avg_time, count = Timer.avg_times[name]
                Timer.avg_times[name] = (avg_time * count / (count + 1) + time_spent * 1000 / (count + 1), count + 1)
                return ret

            return wrapper

        return decorator

    @staticmethod
    def clear_avg(name: str):
        if name in Timer.avg_times.keys():
            Timer.avg_times.pop(name)

    @staticmethod
    def logstr():
        log = 'Time spent: | '
        # Total time spent
        if 'all' in Timer.times.keys():
            total_time = Timer.times['all']
            log += '{:.2f}ms ({:.2f} fps) | '.format(total_time, 1000 / total_time)

        # Time spent on each step
        for k, v in Timer.times.items():
            if k != 'all':
                log += '{}: {:.2f}ms | '.format(k, v)

        # Average time spent on repeated steps
        for k, (v, count) in Timer.avg_times.items():
            if count > 0:
                log += '{}(avg): {:.2f}ms | '.format(k, v)

        return log
