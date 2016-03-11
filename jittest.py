from numba import jit, void, float_, int32, jitclass


def buzz(x):
    return x

spec = [('X',int32)]

@jitclass(spec)
class FooClass(object):

    #@void()
    def __init__(self):
        self.X = 10

    #@float_(float_)
    def floaty(self, x):
        return x * self.X

    #@void()
    def printy(self):
        print self.X


    def foo(self, x):
        sum = 0
        for c in range(x):
            sum += c
            sum += self.bar(c)
        return buzz(sum)

    def bar(self, z):
        a = 0
        for c in range(z*100):
            a += c
        return 0


gg = FooClass()

print gg.floaty(72.0)
print gg.floaty(1)
gg.printy()

print gg.foo(1000)
