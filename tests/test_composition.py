import re

from parking_assistant.composition import new_reservation_reference


def test_reservation_references_have_a_readable_format():
    assert re.fullmatch(r"PK-[0-9A-F]{6}", new_reservation_reference())


def test_reservation_references_are_not_repeated():
    assert len({new_reservation_reference() for _ in range(50)}) == 50
