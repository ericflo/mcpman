# /// script
# requires-python = ">=3.9"  # Using zoneinfo, timezone features, and fromisoformat
# dependencies = [
#     "mcp>=1.6.0",
# ]
# ///

import argparse
import datetime
import time
import calendar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # For timezone support
from typing import Dict, List, Optional, Any, Union  # Added Union

from mcp.server.fastmcp import FastMCP

app = FastMCP("DateTimeUtils")

# --- Helper Functions ---


def _to_isoformat_z(dt: datetime.datetime) -> str:
    """Converts a datetime object to ISO 8601 format ending with 'Z' for UTC.

    Ensures the datetime is UTC before formatting.
    Drops microseconds for cleaner Z format compatibility.

    Args:
        dt (datetime.datetime): The datetime object to convert.

    Returns:
        str: The datetime formatted as YYYY-MM-DDTHH:MM:SZ.
    """
    if dt.tzinfo is None:
        # Assume naive datetimes represent UTC for internal consistency
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    elif dt.tzinfo != datetime.timezone.utc:
        dt = dt.astimezone(datetime.timezone.utc)

    # Format as YYYY-MM-DDTHH:MM:SSZ, dropping microseconds
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso_utc_or_offset(timestamp_iso: str) -> datetime.datetime:
    """Parses an ISO 8601 timestamp string (expecting Z or offset) into a timezone-aware UTC datetime.

    Args:
        timestamp_iso (str): The ISO 8601 timestamp string.

    Returns:
        datetime.datetime: The parsed datetime object, normalized to UTC.

    Raises:
        ValueError: For invalid formats or naive timestamps (lacking Z or offset).
    """
    try:
        # Handle 'Z' explicitly for broader compatibility before fromisoformat
        if timestamp_iso.endswith("Z"):
            timestamp_iso = timestamp_iso[:-1] + "+00:00"

        dt = datetime.datetime.fromisoformat(timestamp_iso)

        # Ensure the datetime is timezone-aware
        if dt.tzinfo is None:
            # ISO format standard implies offset or Z should be present
            raise ValueError("Timestamp lacks timezone information (offset or Z).")

        # Convert to UTC
        return dt.astimezone(datetime.timezone.utc)
    except ValueError as e:
        raise ValueError(
            f"Invalid ISO 8601 timestamp format or value: '{timestamp_iso}'. Error: {e}"
        )
    except Exception as e:  # Catch potential broader parsing issues
        raise ValueError(f"Could not parse timestamp '{timestamp_iso}': {e}")


def _get_zone_info(tz_name: str) -> ZoneInfo:
    """Gets a ZoneInfo object for the given timezone name.

    Args:
        tz_name (str): The IANA timezone name (e.g., 'America/New_York').

    Returns:
        ZoneInfo: The corresponding zoneinfo object.

    Raises:
        ValueError: If the timezone name is invalid or not found.
    """
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        raise ValueError(f"Invalid or unknown timezone name: '{tz_name}'")


# --- Core DateTime Tools ---


@app.tool()
def get_current_time_utc() -> str:
    """Returns the current time in UTC, formatted as an ISO 8601 string (YYYY-MM-DDTHH:MM:SZ).

    Returns:
        str: The current UTC time in ISO 8601 format.
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    return _to_isoformat_z(now_utc)


@app.tool()
def get_current_time_local(timezone_name: str) -> str:
    """Returns the current time in the specified timezone, formatted as an ISO 8601 string with offset.

    Args:
        timezone_name (str): The IANA timezone name (e.g., 'America/New_York', 'Europe/Paris', 'UTC').

    Returns:
        str: The current time in the specified timezone as YYYY-MM-DDTHH:MM:SS+HH:MM format.

    Raises:
        ValueError: If the timezone_name is invalid.
    """
    target_tz = _get_zone_info(timezone_name)
    now_local = datetime.datetime.now(target_tz)
    # Format with offset using isoformat()
    return now_local.isoformat(timespec="seconds")


@app.tool()
def convert_timezone(timestamp_iso: str, target_timezone: str) -> str:
    """Converts a given ISO 8601 timestamp (with Z or offset) to another timezone.

    Args:
        timestamp_iso (str): The timestamp string in ISO 8601 format (must include Z or offset, e.g., '2023-10-27T10:30:00Z' or '2023-10-27T05:30:00-05:00').
        target_timezone (str): The target IANA timezone name (e.g., 'America/Los_Angeles').

    Returns:
        str: The converted timestamp in the target timezone, formatted as ISO 8601 with offset (YYYY-MM-DDTHH:MM:SS+HH:MM).

    Raises:
        ValueError: If the timestamp format is invalid, lacks timezone info, or the target timezone is invalid.
    """
    dt_utc = _parse_iso_utc_or_offset(timestamp_iso)
    target_tz = _get_zone_info(target_timezone)
    dt_target = dt_utc.astimezone(target_tz)
    # Format with offset using isoformat()
    return dt_target.isoformat(timespec="seconds")


@app.tool()
def format_datetime(
    timestamp_iso: str, format_string: str, timezone_name: str = "UTC"
) -> str:
    """Formats a given ISO 8601 timestamp (with Z or offset) according to a Python strftime string, optionally in a specific timezone.

    Args:
        timestamp_iso (str): The timestamp string in ISO 8601 format (must include Z or offset).
        format_string (str): The Python strftime format code (e.g., '%Y-%m-%d %H:%M:%S', '%A, %d %B %Y %Z%z').
        timezone_name (str, optional): Optional IANA timezone name to format the time in. Defaults to 'UTC'.

    Returns:
        str: The formatted date/time string.

    Raises:
        ValueError: If timestamp format, format string, or timezone name are invalid.
    """
    dt_utc = _parse_iso_utc_or_offset(timestamp_iso)
    target_tz = _get_zone_info(timezone_name)
    dt_local = dt_utc.astimezone(target_tz)
    try:
        # %Z and %z formatting requires a timezone-aware datetime
        return dt_local.strftime(format_string)
    except ValueError as e:
        raise ValueError(f"Invalid strftime format string '{format_string}': {e}")


@app.tool()
def parse_datetime(date_string: str, format_string: str) -> str:
    """Parses a date string using a Python strptime format string and returns it as a UTC ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SZ).

    IMPORTANT: The parsed datetime is assumed to represent UTC if no timezone info is parsed via %z/%Z.

    Args:
        date_string (str): The date/time string to parse (e.g., '2023-10-27 10:30:00').
        format_string (str): The Python strptime format string matching the date_string (e.g., '%Y-%m-%d %H:%M:%S').

    Returns:
        str: The parsed timestamp normalized to UTC in ISO 8601 format (YYYY-MM-DDTHH:MM:SZ).

    Raises:
        ValueError: If the date_string does not match the format_string.
    """
    try:
        # strptime usually creates naive datetime unless %z/%Z is successfully parsed
        dt_parsed = datetime.datetime.strptime(date_string, format_string)

        # If it's still naive, assume it was intended to be UTC
        if dt_parsed.tzinfo is None:
            dt_utc = dt_parsed.replace(tzinfo=datetime.timezone.utc)
        else:
            # If timezone info was parsed, convert to UTC for standard output
            dt_utc = dt_parsed.astimezone(datetime.timezone.utc)

        return _to_isoformat_z(dt_utc)
    except ValueError as e:
        raise ValueError(
            f"Could not parse '{date_string}' with format '{format_string}': {e}"
        )


# --- Time Delta and Comparison Tools ---


@app.tool()
def add_timedelta(
    timestamp_iso: str,
    days: float = 0,
    hours: float = 0,
    minutes: float = 0,
    seconds: float = 0,
) -> str:
    """Adds a duration to a given UTC ISO 8601 timestamp.

    Args:
        timestamp_iso (str): The starting timestamp string in ISO 8601 format (must include Z or offset).
        days (float, optional): Number of days to add (can be fractional). Defaults to 0.
        hours (float, optional): Number of hours to add (can be fractional). Defaults to 0.
        minutes (float, optional): Number of minutes to add (can be fractional). Defaults to 0.
        seconds (float, optional): Number of seconds to add (can be fractional). Defaults to 0.

    Returns:
        str: The resulting timestamp, normalized to UTC, in ISO 8601 format (YYYY-MM-DDTHH:MM:SZ).

    Raises:
        ValueError: If the timestamp_iso format is invalid or lacks timezone info.
    """
    dt_utc = _parse_iso_utc_or_offset(timestamp_iso)
    delta = datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    result_dt_utc = dt_utc + delta
    return _to_isoformat_z(result_dt_utc)


@app.tool()
def time_difference(timestamp1_iso: str, timestamp2_iso: str) -> dict:
    """Calculates the precise difference between two ISO 8601 timestamps (with Z or offset).

    Calculates timestamp1 - timestamp2.

    Args:
        timestamp1_iso (str): The first timestamp string (e.g., the later one).
        timestamp2_iso (str): The second timestamp string (e.g., the earlier one).

    Returns:
        dict: A dictionary detailing the difference:
            - 'total_seconds' (float): Total difference in seconds (can be negative).
            - 'days' (int): Whole number of days in the difference.
            - 'seconds' (int): Remaining seconds part (0 <= seconds < 86400).
            - 'microseconds' (int): Remaining microseconds part (0 <= microseconds < 1,000,000).

    Raises:
        ValueError: If either timestamp format is invalid or lacks timezone info.
    """
    dt1_utc = _parse_iso_utc_or_offset(timestamp1_iso)
    dt2_utc = _parse_iso_utc_or_offset(timestamp2_iso)
    delta = dt1_utc - dt2_utc

    return {
        "total_seconds": delta.total_seconds(),
        "days": delta.days,
        "seconds": delta.seconds,
        "microseconds": delta.microseconds,
    }


@app.tool()
def compare_timestamps(timestamp1_iso: str, timestamp2_iso: str) -> str:
    """Compares two ISO 8601 timestamps (with Z or offset).

    Args:
        timestamp1_iso (str): The first timestamp string.
        timestamp2_iso (str): The second timestamp string.

    Returns:
        str: 'before' if timestamp1 < timestamp2, 'after' if timestamp1 > timestamp2, 'equal' otherwise.

    Raises:
        ValueError: If either timestamp format is invalid or lacks timezone info.
    """
    dt1_utc = _parse_iso_utc_or_offset(timestamp1_iso)
    dt2_utc = _parse_iso_utc_or_offset(timestamp2_iso)
    if dt1_utc < dt2_utc:
        return "before"
    elif dt1_utc > dt2_utc:
        return "after"
    else:
        return "equal"


# --- Timestamp Component and Property Tools ---


@app.tool()
def get_timestamp_parts(timestamp_iso: str, timezone_name: str = "UTC") -> dict:
    """Extracts individual components from an ISO 8601 timestamp, optionally in a specific timezone.

    Args:
        timestamp_iso (str): The timestamp string in ISO 8601 format (must include Z or offset).
        timezone_name (str, optional): IANA timezone name to get parts relative to. Defaults to 'UTC'.

    Returns:
        dict: A dictionary containing components:
            - 'year' (int)
            - 'month' (int)
            - 'day' (int)
            - 'hour' (int)
            - 'minute' (int)
            - 'second' (int)
            - 'microsecond' (int)
            - 'day_of_week' (int): 0=Monday, 6=Sunday
            - 'day_of_year' (int): 1-366
            - 'timezone' (str): The name of the timezone used.
            - 'utc_offset_seconds' (int): Offset from UTC in seconds.

    Raises:
        ValueError: If timestamp or timezone name are invalid.
    """
    dt_utc = _parse_iso_utc_or_offset(timestamp_iso)
    target_tz = _get_zone_info(timezone_name)
    dt_local = dt_utc.astimezone(target_tz)
    utc_offset = dt_local.utcoffset()

    return {
        "year": dt_local.year,
        "month": dt_local.month,
        "day": dt_local.day,
        "hour": dt_local.hour,
        "minute": dt_local.minute,
        "second": dt_local.second,
        "microsecond": dt_local.microsecond,
        "day_of_week": dt_local.weekday(),
        "day_of_year": dt_local.timetuple().tm_yday,
        "timezone": timezone_name,
        "utc_offset_seconds": int(utc_offset.total_seconds()) if utc_offset else 0,
    }


@app.tool()
def is_leap_year(year: int) -> bool:
    """Checks if a given year is a leap year according to the Gregorian calendar rules.

    Args:
        year (int): The year to check.

    Returns:
        bool: True if the year is a leap year, False otherwise.
    """
    return calendar.isleap(year)


@app.tool()
def is_valid_date(year: int, month: int, day: int) -> bool:
    """Checks if the given year, month, and day form a valid date in the Gregorian calendar.

    Args:
        year (int): The year.
        month (int): The month (1-12).
        day (int): The day.

    Returns:
        bool: True if the date is valid, False otherwise.
    """
    if not (1 <= month <= 12):
        return False
    try:
        datetime.date(year, month, day)
        return True
    except ValueError:
        return False


# --- Conversion Tools ---


@app.tool()
def iso_to_unix(timestamp_iso: str) -> float:
    """Converts a UTC ISO 8601 timestamp (with Z or offset) to a Unix timestamp (seconds since epoch).

    Args:
        timestamp_iso (str): The timestamp string in ISO 8601 format (must include Z or offset).

    Returns:
        float: The Unix timestamp (seconds since 1970-01-01 00:00:00 UTC).

    Raises:
        ValueError: If the timestamp format is invalid or lacks timezone info.
    """
    dt_utc = _parse_iso_utc_or_offset(timestamp_iso)
    # datetime.timestamp() correctly handles timezone-aware objects
    return dt_utc.timestamp()


@app.tool()
def unix_to_iso(unix_timestamp: float) -> str:
    """Converts a Unix timestamp (seconds since epoch) to a UTC ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SZ).

    Args:
        unix_timestamp (float): The Unix timestamp (seconds since 1970-01-01 00:00:00 UTC).

    Returns:
        str: The timestamp normalized to UTC in ISO 8601 format (YYYY-MM-DDTHH:MM:SZ).

    Raises:
        ValueError: If the Unix timestamp is invalid (e.g., causes an out-of-range error).
    """
    try:
        # Create datetime object directly as UTC
        dt_utc = datetime.datetime.fromtimestamp(
            unix_timestamp, tz=datetime.timezone.utc
        )
        return _to_isoformat_z(dt_utc)
    except (OSError, ValueError) as e:  # Catch potential range errors
        raise ValueError(f"Invalid Unix timestamp {unix_timestamp}: {e}")


# --- Period Boundary Tools ---


@app.tool()
def get_period_boundaries(
    timestamp_iso: str, period: str, timezone_name: str = "UTC"
) -> dict:
    """Calculates the start and end timestamps for a given period containing the specified timestamp, relative to a timezone.

    Periods:
        - 'day': Starts at 00:00:00, ends at 23:59:59.999999.
        - 'week': Starts Monday 00:00:00, ends Sunday 23:59:59.999999.
        - 'month': Starts 1st 00:00:00, ends last day 23:59:59.999999.
        - 'year': Starts Jan 1st 00:00:00, ends Dec 31st 23:59:59.999999.

    Args:
        timestamp_iso (str): The reference timestamp string (ISO 8601 with Z or offset).
        period (str): The period type ('day', 'week', 'month', 'year').
        timezone_name (str, optional): IANA timezone for period boundaries. Defaults to 'UTC'.

    Returns:
        dict: A dictionary with 'start_iso' (str) and 'end_iso' (str) keys containing the boundary timestamps
              normalized to UTC (YYYY-MM-DDTHH:MM:SZ format).

    Raises:
        ValueError: If timestamp, period name, or timezone name are invalid.
    """
    dt_utc = _parse_iso_utc_or_offset(timestamp_iso)
    target_tz = _get_zone_info(timezone_name)
    dt_local = dt_utc.astimezone(target_tz)

    start_dt_local: datetime.datetime
    end_dt_local: datetime.datetime

    if period == "day":
        start_dt_local = dt_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt_local = (
            start_dt_local
            + datetime.timedelta(days=1)
            - datetime.timedelta(microseconds=1)
        )
    elif period == "week":  # Week starts on Monday (weekday()==0)
        start_dt_local = dt_local - datetime.timedelta(days=dt_local.weekday())
        start_dt_local = start_dt_local.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end_dt_local = (
            start_dt_local
            + datetime.timedelta(days=7)
            - datetime.timedelta(microseconds=1)
        )
    elif period == "month":
        start_dt_local = dt_local.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        next_month_year = start_dt_local.year + (start_dt_local.month // 12)
        next_month = (start_dt_local.month % 12) + 1
        end_dt_local = start_dt_local.replace(
            year=next_month_year, month=next_month
        ) - datetime.timedelta(microseconds=1)
    elif period == "year":
        start_dt_local = dt_local.replace(
            month=1, day=1, hour=0, minute=0, second=0, microsecond=0
        )
        end_dt_local = start_dt_local.replace(
            year=start_dt_local.year + 1
        ) - datetime.timedelta(microseconds=1)
    else:
        raise ValueError(
            f"Invalid period: '{period}'. Choose from 'day', 'week', 'month', 'year'."
        )

    # Convert boundaries back to UTC for the return value
    start_utc = start_dt_local.astimezone(datetime.timezone.utc)
    end_utc = end_dt_local.astimezone(datetime.timezone.utc)

    return {
        "start_iso": _to_isoformat_z(start_utc),
        "end_iso": _to_isoformat_z(end_utc),
    }


# --- ISO Week Date Tools ---


@app.tool()
def get_iso_week_date(timestamp_iso: str) -> Dict[str, int]:
    """Gets the ISO 8601 week date components for a given timestamp (UTC).

    Args:
        timestamp_iso (str): The timestamp string in ISO 8601 format (must include Z or offset).

    Returns:
        Dict[str, int]: A dictionary with:
            - 'iso_year' (int): The ISO 8601 week-numbering year.
            - 'iso_week' (int): The ISO 8601 week number (1-53).
            - 'iso_weekday' (int): The ISO 8601 weekday (1=Monday, 7=Sunday).

    Raises:
        ValueError: If the timestamp format is invalid or lacks timezone info.
    """
    dt_utc = _parse_iso_utc_or_offset(timestamp_iso)
    iso_year, iso_week, iso_weekday = dt_utc.isocalendar()
    return {"iso_year": iso_year, "iso_week": iso_week, "iso_weekday": iso_weekday}


@app.tool()
def from_iso_week_date(iso_year: int, iso_week: int, iso_weekday: int) -> str:
    """Constructs a date (represented as UTC timestamp) from ISO 8601 week date components.

    Args:
        iso_year (int): The ISO 8601 week-numbering year.
        iso_week (int): The ISO 8601 week number (1-53).
        iso_weekday (int): The ISO 8601 weekday (1=Monday to 7=Sunday).

    Returns:
        str: The corresponding date as a UTC ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SZ, time part will be 00:00:00).

    Raises:
        ValueError: If the provided ISO week date components are invalid.
    """
    try:
        # fromisocalendar is available directly on datetime.date in Python 3.8+
        # Since we require 3.9, this is fine.
        target_date = datetime.date.fromisocalendar(iso_year, iso_week, iso_weekday)
        # Convert to datetime at midnight UTC
        dt_utc = datetime.datetime.combine(
            target_date, datetime.time.min, tzinfo=datetime.timezone.utc
        )
        return _to_isoformat_z(dt_utc)
    except ValueError as e:
        raise ValueError(
            f"Invalid ISO week date components: year={iso_year}, week={iso_week}, weekday={iso_weekday}. Error: {e}"
        )


# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the DateTimeUtils MCP server.",
        formatter_class=argparse.RawTextHelpFormatter,  # Keep description formatting
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["sse", "stdio"],
        default="stdio",
        help="Transport method to use (sse or stdio).",
    )
    # Future: Add --port for SSE if needed

    args = parser.parse_args()

    print(f"Starting DateTimeUtils server (Transport: {args.transport})")
    app.run(transport=args.transport)
