"""Stats logger provides a value registry for logging training stats. It is a
separate object for backwards compatibility."""

from .value_registry import ValueRegistry


class StatsLogger(object):
    @staticmethod
    def instance():
        return ValueRegistry.get_instance("stats_logger")
