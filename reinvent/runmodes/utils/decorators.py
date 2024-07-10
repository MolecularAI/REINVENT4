import importlib


def extra_dependencies(*library_names):
    def decorator(func):
        def wrapper(*args, **kwargs):
            execute_function = True
            for lib in library_names:
                try:
                    importlib.import_module(lib)
                except ImportError:
                    execute_function = False
            if execute_function:
                return func(*args, **kwargs)
            return

        return wrapper

    return decorator
