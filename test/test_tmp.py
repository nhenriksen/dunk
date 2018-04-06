import pytest

def compare_lists(list1, list2):
    match = True
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            match = False
            break
    return match

def test_lists():
    list1 = 'a 3 c'.split()
    list2 = ['a', 'b', 'c']
    assert list1 == list2
