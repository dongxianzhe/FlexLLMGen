class A:
    def __init__(self):
        pass

    def fun(self):
        a()
        pass


def a():
    b()
    print('a is called')

def b():
    print('b is called')

obj = A()
a()