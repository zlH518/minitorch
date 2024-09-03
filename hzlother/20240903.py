import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple,Callable, Iterable

class A:
    def __init__(self, a: Dict[int,int], b: Iterable[float]):
        self.a = a
        self.b = b

    def geta(self):
        m: Dict[int, int] = self.__dict__["a"]
        return m.values()

    def getb(self):
        m: Iterable[float] = self.__dict__["b"]
        return m

a={1 : 1, 2: 2}
b=[1, 2, 3, 4]
test1=A(a,b)
print(test1.geta())
print(test1.getb())
