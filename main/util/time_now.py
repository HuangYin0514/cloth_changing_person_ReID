import time


def time_now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
