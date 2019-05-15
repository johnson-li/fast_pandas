import fast_pandas.main

def test_main():
    a = 1
    b = 2
    c = fast_pandas.main.just_for_test(a, b)
    assert c == 3

