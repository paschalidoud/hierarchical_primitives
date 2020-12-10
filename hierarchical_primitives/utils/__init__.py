import os


def compose(*funcs):
    """Compose any number of functions by passing the output of one as argument
    to the other.
    
    TODO: Do we need support for multiple outputs? For instance in the case of
          f(g(*args)), can g return multiple values?
    """
    def inner(*args, **kwargs):
        r = funcs[-1](*args, **kwargs)
        for f in reversed(funcs[:-1]):
            r = f(r)
        return r

    return inner


def ensure_parent_directory_exists(path_to_file):
    directory = os.path.dirname(path_to_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
