from datetime import datetime

from parking_assistant.database import WorkingHours, Zone

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def format_live_data(hours: list[WorkingHours], zones: list[Zone], now: datetime) -> str:
    hours_lines = [f"- {WEEKDAYS[h.weekday]}: {h.open_time}-{h.close_time}" for h in hours]
    zone_lines = [
        f"- Zone {z.code} ({z.description}): {z.free_spaces} of {z.total_spaces} spaces free now,"
        f" {z.hourly_price:.2f} EUR/hour, {z.daily_price:.2f} EUR/day maximum"
        for z in zones
    ]
    total_free = sum(z.free_spaces for z in zones)
    return "\n".join(
        [
            f"Live garage data as of {now:%A %Y-%m-%d %H:%M}:",
            "Opening hours:",
            *hours_lines,
            "Zones, current availability and prices:",
            *zone_lines,
            f"Total free spaces now: {total_free}",
        ]
    )
