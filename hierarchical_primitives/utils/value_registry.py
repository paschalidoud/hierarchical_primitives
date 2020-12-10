"""Value registry provides instances of global dictionary like variables.
Although an antipattern it allows for fast prototyping and experimenting with
rapidly changing interfaces."""


class ValueRegistry(object):
    _instances = {}

    @staticmethod
    def get_instance(key):
        if key not in ValueRegistry._instances:
            ValueRegistry._instances[key] = ValueRegistry(key)
        return ValueRegistry._instances[key]

    def __init__(self, key):
        if key in self._instances:
            raise RuntimeError("There can be only one! (imdb:0091203)")
        self._data = {}

    def clear(self):
        self._data.clear()

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        return self.update(key, value)

    def update(self, key, value):
        self._data[key] = value

    def increment(self, key, value):
        self._data[key] = self.get(key, 0) + value

    def get(self, key, value):
        return self._data.get(key, value)
