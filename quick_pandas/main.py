import time
from typing import List

import numpy as np
from numba import njit


@njit()
def t1(a: np.ndarray):
    return np.sum(a)


@njit()
def t2(a: List[np.ndarray]):
    res = 0
    for i in range(len(a)):
        res += np.sum(a[i])
    return res


@njit()
def t3(a: List[np.ndarray], index, s):
    if index >= len(a):
        return s
    s += np.sum(a[index])
    return t3(a, index + 1, s)


@njit()
def calc4(a: np.ndarray):
    return np.sum(a)


@njit()
def t4(a: List[np.ndarray], index, s):
    if index >= len(a):
        return s
    s += calc4(a[index])
    return t4(a, index + 1, s)


@njit()
def calc5(a: np.ndarray):
    return np.sum(a)


@njit()
def t5(a: List[np.ndarray], types, index, s):
    if index >= len(a):
        return s
    t = types[index]
    if t == 0:
        return t5(a, types, index + 1, s + calc5(a[index].view(np.int64)))
    if t == 1:
        return t5(a, types, index + 1, s + calc5(a[index].view(np.float64)))
    if t == 2:
        return t5(a, types, index + 1, s + calc5(a[index].view(np.int32)))


@njit()
def calc6(a: np.ndarray, arrays, types, index, s):
    return t6(arrays, types, index + 1, s + np.sum(a))


@njit()
def t6(a: List[np.ndarray], types, index, s):
    if index >= len(a):
        return s
    t = types[index]
    if t == 0:
        return calc6(a[index].view(np.int64), a, types, index, s)
    if t == 1:
        return calc6(a[index].view(np.float64), a, types, index, s)
    if t == 2:
        return calc6(a[index].view(np.int32), a, types, index, s)
    return 0


@njit()
def calc7(a: np.ndarray, arrays, types, index, s):
    return t7(arrays, types, index + 1, s + np.sum(a))


@njit()
def t7(a: List[np.ndarray], types, index, s):
    return t70(a, types, index, s)


@njit()
def t70(a: List[np.ndarray], types, index, s):
    if index >= len(a):
        return s
    t = types[index]
    if t == 0:
        return calc7(a[index].view(np.int64), a, types, index, s)
    if t == 1:
        return calc7(a[index].view(np.float64), a, types, index, s)
    if t == 2:
        return calc7(a[index].view(np.int32), a, types, index, s)
    return 0


def main():
    a1 = np.array([1, 2], dtype=np.int64)
    a2 = np.array([3, 4], dtype=np.float64)
    a3 = np.array([1, 2], dtype=np.int32)
    arrays1 = [a1, a2, a3]
    arrays2 = [a.view(np.uint8) for a in arrays1]
    types = [0, 1, 2]

    ts = time.time()
    t1(a1)
    print('[latency] t1: %fs' % (time.time() - ts))
    ts = time.time()
    t2(arrays2)
    print('[latency] t2: %fs' % (time.time() - ts))
    ts = time.time()
    t3(arrays2, 0, 0)
    print('[latency] t3: %fs' % (time.time() - ts))
    ts = time.time()
    t4(arrays2, 0, 0)
    print('[latency] t4: %fs' % (time.time() - ts))
    ts = time.time()
    t5(arrays2, types, 0, 0)
    print('[latency] t5: %fs' % (time.time() - ts))
    ts = time.time()
    t6(arrays2, types, 0, 0)
    print('[latency] t6: %fs' % (time.time() - ts))
    ts = time.time()
    t7(arrays2, types, 0, 0)
    print('[latency] t7: %fs' % (time.time() - ts))


if __name__ == '__main__':
    main()
