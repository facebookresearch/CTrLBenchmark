import pytest

import ctrl


def test_s_plus():
    task_gen = ctrl.get_stream('s_plus')
    tasks = list(task_gen)
    assert len(tasks) == 6
    assert tasks[0].src_concepts == tasks[-1].src_concepts
    assert len(tasks[0].datasets[0]) < len(tasks[-1].datasets[0])


def test_s_minus():
    task_gen = ctrl.get_stream('s_minus')
    tasks = list(task_gen)
    assert len(tasks) == 6
    assert tasks[0].src_concepts == tasks[-1].src_concepts
    assert len(tasks[0].datasets[0]) > len(tasks[-1].datasets[0])


stream_lengths = [
    ('s_in', 6),
    ('s_out', 6),
    ('s_pl', 5)
]


@pytest.mark.parametrize("stream_name, expected_len", stream_lengths)
def test_get_stream_len(stream_name, expected_len):
    task_gen = ctrl.get_stream(stream_name)

    tasks = list(task_gen)
    assert len(tasks) == expected_len
