import time

from rock_texture_analyzer.utils.method_cache import MethodCache, method_cache


class A(MethodCache):
    @method_cache
    def b(self):
        return time.time()


def test_method_cache():
    assert A().b() == A().b()
