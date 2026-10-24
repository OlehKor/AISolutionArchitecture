from fakes import NOW
from parking_assistant.database import WorkingHours, Zone
from parking_assistant.live_data import format_live_data

ZONES = [
    Zone("A", "Level 0, short stay", 10, 4, 0, 3.5, 28.0),
    Zone("B", "Level -1, standard", 20, 20, 5, 2.5, 20.0),
]


def test_live_data_lists_hours_prices_and_availability():
    text = format_live_data([WorkingHours(0, "06:00", "23:00")], ZONES, NOW)

    assert "Monday: 06:00-23:00" in text
    assert "Zone A (Level 0, short stay): 6 of 10 spaces free now" in text
    assert "3.50 EUR/hour, 28.00 EUR/day maximum" in text


def test_live_data_sums_free_spaces_over_zones():
    text = format_live_data([], ZONES, NOW)
    assert "Total free spaces now: 6" in text
    assert "2026-10-24 10:00" in text
