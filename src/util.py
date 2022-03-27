#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Tuple


def list_tuple_transpose(input_list: List[tuple]) -> Tuple[list, ...]:
    if len(input_list) <= 0:
        return tuple()

    if len(input_list[0]) <= 0:
        return tuple([] for _ in range(len(input_list)))

    outer_list = [[] for _ in range(len(input_list[0]))]

    for inner_tuple in input_list:

        assert len(input_list[0]) == len(inner_tuple)

        for elem_idx, elem in enumerate(inner_tuple):
            outer_list[elem_idx].append(elem)

    return tuple(outer_list)


def no_op() -> None:
    """
    A no-op function, for use with the `if __name__ == "__main__"` idiom.
    """


if __name__ == "__main__":
    no_op()
