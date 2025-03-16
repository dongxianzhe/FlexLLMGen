from typing import Any

class ValueHolder:
    def __init__(self):
        self.val: Any = None

    def store(self, val: Any):
        assert self.val is None
        self.val = val

    def pop(self: Any):
        ret = self.val
        self.val = None
        return ret

    def clear(self: Any):
        self.val = None


def array_1d(a):
    return [ValueHolder() for _ in range(a)]


def array_2d(a, b):
    return [[ValueHolder() for _ in range(b)] for _ in range(a)]


def array_3d(a, b, c):
    return [[[ValueHolder() for _ in range(c)] for _ in range(b)] for _ in range(a)]


def array_4d(a, b, c, d):
    return [[[[ValueHolder() for _ in range(d)] for _ in range(c)] for _ in range(b)] for _ in range(a)]