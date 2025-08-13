import argparse
import datetime
import json
import logging
import re
import typing

import matplotlib
import matplotlib.pyplot
import mplcursors
import polars


def setup_logging(verbose_count: int) -> logging.Logger:
    """
    Setup logging based on verbosity level
    0: WARNING and above (silence is golden)
    1: INFO and above
    2: DEBUG and above
    """
    if verbose_count == 0:
        level = logging.WARNING
    elif verbose_count == 1:
        level = logging.INFO
    else:  # 2 or more
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    return logging.getLogger(__name__)


def parse_duration_string(duration_str: str, allow_zero: bool = False) -> int:
    """
    Parse user-friendly duration strings like '1h', '20m', '1d30m', '2h45m'
    Returns duration in minutes
    allow_zero: If True, allows zero duration (useful for waitAfterPrevious)
    """
    duration_str = duration_str.lower().strip()

    # Pattern to match combinations like: 1d30m, 2h45m, 1h, 30m, 1d, etc.
    pattern = r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?"
    match = re.match(f"^{pattern}$", duration_str)

    if not match:
        raise ValueError(
            f"Invalid duration format: {duration_str}. Use formats like '1h', '20m', '1d30m', '2h45m'"
        )

    days = int(match.group(1)) if match.group(1) else 0
    hours = int(match.group(2)) if match.group(2) else 0
    minutes = int(match.group(3)) if match.group(3) else 0

    total_minutes = days * 24 * 60 + hours * 60 + minutes

    if total_minutes == 0 and not allow_zero:
        raise ValueError(f"Duration cannot be zero: {duration_str}")

    return total_minutes


def parse_time_string(time_str: str) -> datetime.time:
    """
    Parse user-friendly time strings like '12pm', '1am', '2:10am', '14:30'
    Returns a datetime.time object
    """
    time_str = time_str.lower().strip()

    # Handle 24-hour format (e.g., '14:30', '09:15')
    if ":" in time_str and ("am" not in time_str and "pm" not in time_str):
        try:
            hour, minute = map(int, time_str.split(":"))
            return datetime.time(hour, minute)
        except ValueError:
            raise ValueError(f"Invalid 24-hour time format: {time_str}")

    # Handle 12-hour format with am/pm
    # Match patterns like '12pm', '1am', '2:10am', '11:30pm'
    pattern = r"^(\d{1,2})(?::(\d{2}))?(am|pm)$"
    match = re.match(pattern, time_str)

    if not match:
        raise ValueError(
            f"Invalid time format: {time_str}. Use formats like '12pm', '1am', '2:10am', or '14:30'"
        )

    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    period = match.group(3)

    # Validate minute
    if minute >= 60:
        raise ValueError(f"Invalid minute value: {minute}")

    # Convert to 24-hour format
    if period == "am":
        if hour == 12:
            hour = 0
        elif hour > 12:
            raise ValueError(f"Invalid hour for AM: {hour}")
    else:  # pm
        if hour != 12:
            hour += 12
        elif hour > 12:
            raise ValueError(f"Invalid hour for PM: {hour}")

    return datetime.time(hour, minute)


def parse_date_string(date_str: typing.Optional[str]) -> datetime.date:
    """
    Parse date strings in relative or absolute format
    Relative: 'today', 'tomorrow', 'in 2 days', 'day after tomorrow'
    Absolute: 'mm-dd' format like '12-25' or '01-15'
    Returns a datetime.date object
    """
    if not date_str:
        return datetime.date.today()

    date_str = date_str.lower().strip()
    today = datetime.date.today()

    # Handle relative dates
    if date_str == "today":
        return today
    elif date_str == "tomorrow":
        return today + datetime.timedelta(days=1)
    elif date_str == "day after tomorrow":
        return today + datetime.timedelta(days=2)
    elif date_str.startswith("in ") and date_str.endswith(" days"):
        # Handle "in X days"
        try:
            days_match = re.search(r"in (\d+) days?", date_str)
            if days_match:
                days = int(days_match.group(1))
                return today + datetime.timedelta(days=days)
        except (ValueError, AttributeError):
            pass

    # Handle absolute dates in mm-dd format
    if re.match(r"^\d{1,2}-\d{1,2}$", date_str):
        try:
            month, day = map(int, date_str.split("-"))
            # Assume current year, but if the date has already passed this year, use next year
            current_year = today.year
            target_date = datetime.date(current_year, month, day)

            if target_date < today:
                target_date = datetime.date(current_year + 1, month, day)

            return target_date
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}")

    raise ValueError(
        f"Invalid date format: {date_str}. Use 'today', 'tomorrow', 'in X days', or 'mm-dd' format"
    )


def load_timeline_config(
    config_path: str, logger: logging.Logger
) -> typing.Dict[str, typing.Any]:
    """
    Load timeline configuration from JSON file
    """
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.debug("Successfully loaded JSON configuration")

        # Validate required fields
        if "title" not in config:
            raise ValueError("Missing 'title' field in configuration")
        if "steps" not in config:
            raise ValueError("Missing 'steps' field in configuration")

        logger.info(f"Configuration title: {config['title']}")
        logger.info(f"Number of steps: {len(config['steps'])}")

        # Validate each step and convert duration to minutes
        for i, step in enumerate(config["steps"]):
            logger.debug(f"Processing step {i}: {step.get('name', 'unnamed')}")

            required_fields = ["name", "duration", "color"]
            for field in required_fields:
                if field not in step:
                    raise ValueError(f"Step {i}: Missing required field '{field}'")

            # Convert duration from user-friendly format to minutes
            if isinstance(step["duration"], str):
                try:
                    original_duration = step["duration"]
                    step["duration"] = parse_duration_string(
                        step["duration"], allow_zero=False
                    )
                    logger.debug(
                        f"Step {i} duration: {original_duration} -> {step['duration']} minutes"
                    )
                except ValueError as e:
                    raise ValueError(f"Step {i} ({step['name']}): {e}")

            # Set default waitAfterPrevious if not specified
            if "waitAfterPrevious" not in step:
                step["waitAfterPrevious"] = 0
                logger.debug(f"Step {i}: Set default waitAfterPrevious to 0")
            elif isinstance(step["waitAfterPrevious"], str):
                try:
                    original_wait = step["waitAfterPrevious"]
                    # Allow zero for waitAfterPrevious since "start immediately" is valid
                    step["waitAfterPrevious"] = parse_duration_string(
                        step["waitAfterPrevious"], allow_zero=True
                    )
                    logger.debug(
                        f"Step {i} waitAfterPrevious: {original_wait} -> {step['waitAfterPrevious']} minutes"
                    )
                except ValueError as e:
                    raise ValueError(
                        f"Step {i} ({step['name']}) waitAfterPrevious: {e}"
                    )

        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def create_timeline_from_config(
    config: typing.Dict[str, typing.Any],
    start_time_str: typing.Optional[str],
    start_date_str: typing.Optional[str],
    logger: logging.Logger,
) -> typing.Tuple[typing.List[typing.Dict[str, typing.Any]], str]:
    """
    Create timeline data from JSON configuration with FIXED dependency handling
    """
    # Parse start time (default to current time if not provided)
    if start_time_str:
        try:
            start_time_obj = parse_time_string(start_time_str)
            logger.info(
                f"Using specified start time: {start_time_obj.strftime('%I:%M %p')}"
            )
        except ValueError as e:
            raise ValueError(f"Invalid start time: {e}")
    else:
        start_time_obj = datetime.datetime.now().time().replace(second=0, microsecond=0)
        logger.info(
            f"Using current time as start: {start_time_obj.strftime('%I:%M %p')}"
        )

    # Parse start date
    try:
        start_date_obj = parse_date_string(start_date_str)
        logger.info(f"Using start date: {start_date_obj.strftime('%Y-%m-%d')}")
    except ValueError as e:
        raise ValueError(f"Invalid start date: {e}")

    base_datetime = datetime.datetime.combine(start_date_obj, start_time_obj)
    logger.info(f"Base datetime: {base_datetime}")

    # Track task completion times for dependency resolution
    task_end_times: typing.Dict[str, datetime.datetime] = {}
    tasks_data: typing.List[typing.Dict[str, typing.Any]] = []

    for i, step in enumerate(config["steps"]):
        logger.debug(f"Processing step {i}: {step['name']}")

        # Determine start time based on dependencies
        if "dependsOn" in step and step["dependsOn"]:
            # DEPENDENCY CASE: Start after the dependency finishes
            dependency_name = step["dependsOn"]
            logger.debug(f"{step['name']} depends on '{dependency_name}'")

            if dependency_name not in task_end_times:
                raise ValueError(
                    f"Step '{step['name']}' depends on '{dependency_name}' which hasn't been defined yet"
                )

            # Start exactly when the dependency finishes
            start_time = task_end_times[dependency_name]
            logger.debug(f"Dependency '{dependency_name}' ends at {start_time}")

            # Add any additional wait time
            if step["waitAfterPrevious"] > 0:
                start_time += datetime.timedelta(minutes=step["waitAfterPrevious"])
                logger.debug(
                    f"Added {step['waitAfterPrevious']} minutes wait, new start: {start_time}"
                )

        else:
            # SEQUENTIAL CASE: Start after previous task or at base time
            if i == 0:
                # First task starts at base time
                start_time = base_datetime
                logger.debug(f"First task starts at base time: {start_time}")
            else:
                # Start after previous task + wait time
                prev_end = tasks_data[-1]["end_time"]
                start_time = prev_end + datetime.timedelta(
                    minutes=step["waitAfterPrevious"]
                )
                logger.debug(
                    f"Sequential start after previous task end ({prev_end}) + {step['waitAfterPrevious']} min wait = {start_time}"
                )

        # Calculate end time
        end_time = start_time + datetime.timedelta(minutes=step["duration"])
        logger.debug(
            f"{step['name']}: {start_time} -> {end_time} (duration: {step['duration']} min)"
        )

        task = {
            "task_id": i + 1,
            "task_name": step["name"],
            "start_time": start_time,
            "end_time": end_time,
            "category": step.get("category", "General"),
            "description": step.get(
                "description", f"{step['name']} - Duration: {step['duration']} minutes"
            ),
            "color": step["color"],
            "order": i + 1,
            "duration_minutes": step["duration"],
        }

        tasks_data.append(task)
        # CRITICAL: Store the end time for this task so other tasks can depend on it
        task_end_times[step["name"]] = end_time
        logger.debug(f"Stored end time for '{step['name']}': {end_time}")

    logger.info(f"Generated timeline with {len(tasks_data)} tasks")
    return tasks_data, config["title"]


def format_duration(minutes: int) -> str:
    """
    Format duration in minutes to a compact string like '2h30m', '45m', or '1h'
    """
    hours = minutes // 60
    remaining_minutes = minutes % 60

    if hours > 0:
        if remaining_minutes > 0:
            return f"{hours}h{remaining_minutes}m"
        else:
            return f"{hours}h"
    else:
        return f"{remaining_minutes}m"


def print_schedule_table(
    tasks_data: typing.List[typing.Dict[str, typing.Any]], logger: logging.Logger
) -> None:
    """
    Print a formatted schedule table to stdout
    """
    logger.debug("Generating schedule table")

    print("\n" + "=" * 80)
    print("TIMELINE SCHEDULE")
    print("=" * 80)

    # Calculate column widths
    max_task_width = max(len(task["task_name"]) for task in tasks_data)
    task_width = max(max_task_width, 20)

    # Header
    print(
        f"{'Task':<{task_width}} {'Start Time':<12} {'End Time':<12} {'Duration':<10} {'Description'}"
    )
    print("-" * 80)

    # Rows
    for task in tasks_data:
        duration_hours = (task["end_time"] - task["start_time"]).total_seconds() / 3600
        if duration_hours < 1:
            duration_str = f"{int(duration_hours * 60)}min"
        else:
            hours = int(duration_hours)
            minutes = int((duration_hours - hours) * 60)
            if minutes == 0:
                duration_str = f"{hours}h"
            else:
                duration_str = f"{hours}h{minutes}m"

        start_str = task["start_time"].strftime("%I:%M %p")
        end_str = task["end_time"].strftime("%I:%M %p")

        # Truncate description if too long
        desc = task["description"]
        if len(desc) > 30:
            desc = desc[:27] + "..."

        print(
            f"{task['task_name']:<{task_width}} {start_str:<12} {end_str:<12} {duration_str:<10} {desc}"
        )

    print("=" * 80)


def create_gantt_chart_from_config(
    config_path: str,
    start_time_str: typing.Optional[str],
    start_date_str: typing.Optional[str],
    output_path: typing.Optional[str] = None,
    no_dates: bool = False,
    logger: typing.Optional[logging.Logger] = None,
) -> typing.Tuple[typing.Any, typing.Any]:
    """
    Creates a Gantt chart from JSON configuration
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Starting Gantt chart generation")

    # Load configuration
    config = load_timeline_config(config_path, logger)
    tasks_data, chart_title = create_timeline_from_config(
        config, start_time_str, start_date_str, logger
    )

    # Print schedule table to stdout
    print_schedule_table(tasks_data, logger)

    logger.info("Configuring matplotlib")
    # Configure plot style
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    matplotlib.rcParams["axes.titlesize"] = 16
    matplotlib.rcParams["axes.labelsize"] = 12

    # Convert to Polars DataFrame
    logger.debug("Converting tasks data to Polars DataFrame")
    df = polars.DataFrame(tasks_data)
    # Calculate duration in hours
    df = df.with_columns([
        (
            (polars.col("end_time") - polars.col("start_time")).dt.total_seconds()
            / 3600
        ).alias("duration_hours")
    ])

    # Create compact plot with narrow height
    logger.debug("Creating matplotlib figure")
    fig, ax = matplotlib.pyplot.subplots(figsize=(18, 6))

    # Create y-positions for tasks in chronological order
    task_names = df["task_name"].to_list()
    y_positions = {name: i for i, name in enumerate(task_names)}

    # Plot task bars
    logger.debug("Plotting task bars")
    bars = []
    for row in df.iter_rows(named=True):
        color = row["color"]
        y_pos = y_positions[row["task_name"]]

        # CRITICAL FIX: Convert duration from hours to timedelta for proper matplotlib rendering
        duration_timedelta = datetime.timedelta(hours=row["duration_hours"])

        logger.debug(
            f"Bar for {row['task_name']}: {row['start_time']} to {row['end_time']}"
        )

        bar = ax.barh(
            y_pos,
            duration_timedelta,  # Use timedelta for proper datetime scaling
            left=row["start_time"],
            height=0.8,
            color=color,
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
        )

        # Add task name label with duration in the center of each bar
        bar_center_x = row["start_time"] + duration_timedelta / 2

        # Format the duration for display
        duration_str = format_duration(row["duration_minutes"])
        task_label = f"{row['task_name']} ({duration_str})"

        # Choose text color based on bar color brightness
        # Convert hex color to RGB and calculate brightness
        hex_color = color.lstrip("#")
        rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
        text_color = "white" if brightness < 128 else "black"

        ax.text(
            bar_center_x,
            y_pos,
            task_label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=text_color,
        )

        # Add annotation data to each bar
        for rect in bar:
            rect.task_info = (
                f"Task: {row['task_name']}\n"
                f"Start: {row['start_time'].strftime('%I:%M %p')}\n"
                f"End: {row['end_time'].strftime('%I:%M %p')}\n"
                f"Duration: {row['duration_hours']:.1f} hours\n"
                f"Description: {row['description']}"
            )

        bars.extend(bar)

    # Add start time labels for each task
    logger.debug("Adding time labels")
    for row in df.iter_rows(named=True):
        y_pos = y_positions[row["task_name"]]
        start_time = row["start_time"]

        # Format time as 1pm, 2:35pm, etc. (no date info)
        time_label = start_time.strftime(
            "%-I:%M %p" if start_time.minute != 0 else "%-I %p"
        ).lower()

        # Position label to the left of the bar to avoid collisions
        label_x = start_time - datetime.timedelta(minutes=45)

        ax.text(
            label_x,
            y_pos,
            time_label,
            fontsize=9,
            fontweight="bold",
            color="black",
            ha="right",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"
            ),
        )

    # Add hover annotations
    logger.debug("Setting up hover annotations")
    cursor = mplcursors.cursor(bars, hover=True)

    @cursor.connect("add")
    def on_hover(sel):
        sel.annotation.set_text(sel.artist.task_info)
        sel.annotation.get_bbox_patch().set(
            facecolor="white", alpha=0.9, edgecolor="black"
        )
        sel.annotation.set_fontsize(10)

    # Format time axis - IMPROVED to prevent overlap
    start_time = df["start_time"].min()
    end_time = df["end_time"].max()

    # Create strategic time ticks with better spacing
    total_hours = (end_time - start_time).total_seconds() / 3600
    if total_hours <= 8:
        tick_interval = 2
    elif total_hours <= 16:
        tick_interval = 3
    elif total_hours <= 24:
        tick_interval = 4
    else:
        tick_interval = 6

    logger.debug(
        f"Timeline spans {total_hours:.1f} hours, using {tick_interval}h tick intervals"
    )

    key_times = []
    current_tick = start_time
    while current_tick <= end_time:
        key_times.append(current_tick)
        current_tick += datetime.timedelta(hours=tick_interval)

    # Ensure end time is included if not too close to last tick
    if (
        key_times
        and (end_time - key_times[-1]).total_seconds() / 3600 >= tick_interval / 2
    ):
        key_times.append(end_time)

    # Set the main time ticks
    ax.set_xticks(key_times)

    # Format labels based on no_dates option
    time_labels = []
    if no_dates:
        logger.debug("Using time-only labels (no dates)")
        # Show only times, no dates
        for time in key_times:
            if time.minute == 0:
                time_labels.append(time.strftime("%-I %p"))
            else:
                time_labels.append(time.strftime("%-I:%M %p"))
    else:
        logger.debug("Using time labels with date information")
        # Show times with date transitions (original behavior)
        base_date = start_time.date()
        for time in key_times:
            if time.date() == base_date:
                if time.minute == 0:
                    time_labels.append(time.strftime("%-I %p"))
                else:
                    time_labels.append(time.strftime("%-I:%M %p"))
            else:
                days_diff = (time.date() - base_date).days
                if time.minute == 0:
                    time_labels.append(
                        f"{time.strftime('%-I %p')}\n(Day {days_diff + 1})"
                    )
                else:
                    time_labels.append(
                        f"{time.strftime('%-I:%M %p')}\n(Day {days_diff + 1})"
                    )

    ax.set_xticklabels(
        time_labels, fontsize=10, ha="center", rotation=0, verticalalignment="top"
    )

    # Add more spacing for x-axis labels
    ax.tick_params(axis="x", pad=20)

    # Add grid lines
    ax.grid(axis="x", linestyle="--", alpha=0.3, color="gray")

    # Add day transition lines and secondary date axis only if not no_dates
    if not no_dates:
        logger.debug("Adding day transition markers")
        current_date = start_time.date()
        end_date = end_time.date()

        # Add vertical dotted lines for new day transitions
        while current_date < end_date:
            next_day = current_date + datetime.timedelta(days=1)
            day_transition = datetime.datetime.combine(next_day, datetime.time(0, 0))

            if start_time <= day_transition <= end_time:
                ax.axvline(
                    x=day_transition, color="red", linestyle=":", alpha=0.8, linewidth=2
                )
                ax.text(
                    day_transition,
                    len(df),
                    f" New Day\n {next_day.strftime('%b %d')}",
                    rotation=90,
                    verticalalignment="bottom",
                    fontsize=10,
                    color="red",
                    fontweight="bold",
                    ha="left",
                )

            current_date = next_day

        # Create secondary axis for date display
        sec_ax = ax.secondary_xaxis("bottom")

        # Calculate day boundaries
        current_date = start_time.date()
        end_date = end_time.date()
        date_ticks = []
        date_labels = []

        day_num = 1
        while current_date <= end_date:
            # Position tick in the middle of each day's timeline
            day_start = datetime.datetime.combine(current_date, datetime.time(0, 0))
            day_end = datetime.datetime.combine(current_date, datetime.time(23, 59))

            # Clamp to actual timeline bounds
            day_start = max(day_start, start_time)
            day_end = min(day_end, end_time)

            if day_start < day_end:
                middle_of_day = day_start + (day_end - day_start) / 2
                date_ticks.append(middle_of_day)
                date_labels.append(f"Day {day_num} ({current_date.strftime('%b %d')})")

            current_date += datetime.timedelta(days=1)
            day_num += 1

        if date_ticks:
            sec_ax.set_xticks(date_ticks)
            sec_ax.set_xticklabels(
                date_labels, fontsize=12, color="gray", weight="bold"
            )
            sec_ax.tick_params(axis="x", labelsize=12, colors="gray", pad=30)
            sec_ax.spines["bottom"].set_position(("outward", 60))

        # Hide unwanted spines
        sec_ax.spines["top"].set_visible(False)
        sec_ax.spines["right"].set_visible(False)

    # Set y-axis (remove y-axis labels since we have task names on bars)
    ax.set_yticks([])

    # Customize the plot
    logger.debug("Finalizing plot styling")
    ax.set_title(chart_title, fontsize=20, fontweight="bold", pad=40)
    ax.set_xlabel("Time", fontsize=14, fontweight="bold", labelpad=40)
    ax.set_ylabel("")  # Remove y-axis label

    # Style the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)  # Hide left spine since no y-axis labels
    ax.spines["bottom"].set_linewidth(0.5)

    # Set axis limits with padding for labels
    ax.set_xlim(
        start_time - datetime.timedelta(hours=1.5),
        end_time + datetime.timedelta(hours=0.5),
    )
    # Set tight y-axis limits for compact layout
    ax.set_ylim(-0.3, len(df) - 0.7)

    # Invert y-axis so chronological order goes top to bottom
    ax.invert_yaxis()

    # Apply layout adjustments
    matplotlib.pyplot.tight_layout(pad=2.0)

    # Adjust bottom padding based on whether we have date axis
    if no_dates:
        matplotlib.pyplot.subplots_adjust(bottom=0.15)
    else:
        matplotlib.pyplot.subplots_adjust(bottom=0.2)

    # Save or show
    if output_path:
        logger.info(f"Saving chart to: {output_path}")
        matplotlib.pyplot.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.3,
        )
        print(f"Timeline chart saved as '{output_path}'")
    else:
        logger.info("Displaying chart")
        matplotlib.pyplot.show()

    return fig, ax


def main() -> int:
    """
    Main function with command-line argument parsing
    """
    parser = argparse.ArgumentParser(
        description="Generate timeline Gantt charts from JSON configuration"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="timeline.json",
        help="Path to JSON configuration file (default: timeline.json)",
    )
    parser.add_argument(
        "--start-time",
        "-s",
        help="Start time in user-friendly format (e.g., 7am, 2:30pm, 14:30). Defaults to current time.",
    )
    parser.add_argument(
        "--start-date",
        "-d",
        help="Start date in relative format (today, tomorrow, in 2 days) or absolute (mm-dd). Defaults to today.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (if not specified, chart will be displayed)",
    )
    parser.add_argument(
        "--no-dates",
        action="store_true",
        help="Remove date information from display (keeps time of day)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (use multiple times for more verbose output: -v for INFO, -vv for DEBUG)",
    )

    args = parser.parse_args()

    # Setup logging based on verbosity
    logger = setup_logging(args.verbose)

    try:
        if args.verbose > 0:
            if args.start_time:
                print(f"Start time: {args.start_time}")
            else:
                current_time = (
                    datetime.datetime.now().time().replace(second=0, microsecond=0)
                )
                print(f"Start time: {current_time.strftime('%I:%M %p')} (current time)")

            if args.start_date:
                print(f"Start date: {args.start_date}")
            else:
                print("Start date: today")

        create_gantt_chart_from_config(
            args.config,
            args.start_time,
            args.start_date,
            args.output,
            args.no_dates,
            logger,
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
