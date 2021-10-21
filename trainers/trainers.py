trainers = {}


def register(name):
    def decorator(cls):
        trainers[name] = cls
        return cls
    return decorator
