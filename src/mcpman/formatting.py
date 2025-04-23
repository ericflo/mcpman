"""
Formatting utilities for MCPMan console output.

This module provides consistent formatting for all console output,
including boxes, colors, text alignment, and other visual elements.
"""

import re
import os
import textwrap
import itertools
import threading
import sys
import time
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


# Terminal width detection
def get_terminal_width():
    """Get terminal width or a reasonable default"""
    try:
        # Get actual terminal width but cap it to avoid extreme stretching
        width = os.get_terminal_size().columns
        return min(width, 120)  # Cap at 120 chars for readability
    except (OSError, AttributeError):
        return 80


def visible_length(s):
    """
    Calculate the visible length of a string, excluding ANSI color codes.

    Args:
        s: The string to measure

    Returns:
        The visible length of the string
    """
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return len(ansi_escape.sub("", s))


def normalize_text(text):
    """
    Normalize text by handling whitespace, control characters, and line endings.

    Args:
        text: The text to normalize

    Returns:
        Normalized text
    """
    # Remove control characters but preserve tabs and newlines
    clean_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Normalize line endings
    clean_text = re.sub(r"\r\n?", "\n", clean_text)

    # Replace tabs with spaces
    clean_text = clean_text.replace("\t", "    ")

    # Collapse multiple newlines and strip
    clean_text = re.sub(r"\n\s*\n+", "\n\n", clean_text).strip()

    return clean_text


def format_single_line(text, max_width=None):
    """
    Format text as a single line, collapsing whitespace and truncating if needed.

    Args:
        text: The text to format
        max_width: Maximum width (optional)

    Returns:
        Formatted single line
    """
    # Normalize the text for display - collapse all whitespace
    clean_text = re.sub(r"\s+", " ", normalize_text(text)).strip()

    # Truncate if needed
    if max_width and len(clean_text) > max_width:
        return clean_text[: max_width - 3] + "..."

    return clean_text


def format_paragraphs(text, width=None):
    """
    Format text into paragraphs with proper line wrapping.

    Args:
        text: The text to format
        width: Maximum width per line (optional)

    Returns:
        List of wrapped paragraph lines
    """
    if width is None:
        width = get_terminal_width() - 10

    # First normalize the text
    clean_text = normalize_text(text)

    # Split into paragraphs and wrap each one
    paragraphs = clean_text.split("\n\n")

    all_lines = []
    for para in paragraphs:
        if para.strip():
            # Clean internal whitespace for wrapping
            clean_para = re.sub(r"\s+", " ", para).strip()
            # Wrap the paragraph
            wrapped = textwrap.fill(clean_para, width=width)
            all_lines.extend(wrapped.split("\n"))
        all_lines.append("")  # Empty line between paragraphs

    # Remove last empty line
    if all_lines and not all_lines[-1]:
        all_lines.pop()

    return all_lines


# Box styles
class BoxStyle:
    """Box style definitions for different types of boxes"""

    STANDARD = {
        "top_left": "╔",
        "top_right": "╗",
        "bottom_left": "╚",
        "bottom_right": "╝",
        "horizontal": "═",
        "vertical": "║",
        "left_t": "╠",
        "right_t": "╣",
        "color": Fore.CYAN,
    }

    LIGHT = {
        "top_left": "╭",
        "top_right": "╮",
        "bottom_left": "╰",
        "bottom_right": "╯",
        "horizontal": "─",
        "vertical": "│",
        "left_t": "├",
        "right_t": "┤",
        "color": Fore.YELLOW,
    }

    SUCCESS = {
        "top_left": "╔",
        "top_right": "╗",
        "bottom_left": "╚",
        "bottom_right": "╝",
        "horizontal": "═",
        "vertical": "║",
        "left_t": "╠",
        "right_t": "╣",
        "color": Fore.GREEN,
    }

    WARNING = {
        "top_left": "╔",
        "top_right": "╗",
        "bottom_left": "╚",
        "bottom_right": "╝",
        "horizontal": "═",
        "vertical": "║",
        "left_t": "╠",
        "right_t": "╣",
        "color": Fore.RED,
    }

    SERVER = {
        "top_left": "╔",
        "top_right": "╗",
        "bottom_left": "╚",
        "bottom_right": "╝",
        "horizontal": "═",
        "vertical": "║",
        "left_t": "╠",
        "right_t": "╣",
        "color": Fore.MAGENTA,
        "indent": 2,  # Special handling for server boxes
    }

    PROMPT = {
        "top_left": "┌",
        "top_right": "┐",
        "bottom_left": "└",
        "bottom_right": "┘",
        "horizontal": "─",
        "vertical": "│",
        "left_t": "├",
        "right_t": "┤",
        "color": Fore.MAGENTA,
    }


def draw_box(title, content, style=BoxStyle.STANDARD, width=None, indent=0):
    """
    Draw a formatted box with title and content.

    Args:
        title: The box title
        content: Content lines (string or list of strings)
        style: Box style to use
        width: Width of the box (auto-detected if None)
        indent: Additional indentation

    Returns:
        List of formatted box lines
    """
    # Determine box width
    if width is None:
        term_width = get_terminal_width()
        width = min(term_width - 6, 80)  # Reasonable default

    # Apply box indentation
    box_indent = " " * indent
    if "indent" in style:
        box_indent = " " * style["indent"]

    # Extract style elements with proper color formatting
    color = style["color"]
    top_left = f"{color}{style['top_left']}{Style.RESET_ALL}"
    top_right = f"{color}{style['top_right']}{Style.RESET_ALL}"
    bottom_left = f"{color}{style['bottom_left']}{Style.RESET_ALL}"
    bottom_right = f"{color}{style['bottom_right']}{Style.RESET_ALL}"
    horizontal = f"{color}{style['horizontal']}"  # No reset to allow continuous lines
    vertical = f"{color}{style['vertical']}{Style.RESET_ALL}"
    left_t = f"{color}{style['left_t']}{Style.RESET_ALL}"
    right_t = f"{color}{style['right_t']}{Style.RESET_ALL}"

    # Calculate usable width inside the box (between borders)
    content_width = (
        width - 4
    )  # -4 for borders and padding (left border, space, content, space, right border)

    # Create the horizontal lines with exact width
    horizontal_line = horizontal * content_width
    top_border = f"{box_indent}{color}{style['top_left']}{horizontal_line}{style['top_right']}{Style.RESET_ALL}"
    bottom_border = f"{box_indent}{color}{style['bottom_left']}{horizontal_line}{style['bottom_right']}{Style.RESET_ALL}"

    # Create title section if provided
    if title:
        # Create colored title
        colored_title = f"{Fore.WHITE}{title}{Style.RESET_ALL}"
        title_visible_len = visible_length(colored_title)

        # Calculate padding for exact centering
        padding_total = content_width - title_visible_len
        left_padding = padding_total // 2
        right_padding = padding_total - left_padding

        # Format the title line with exact centering
        title_line = (
            f"{box_indent}{color}{style['vertical']}{Style.RESET_ALL}"
            f"{' ' * left_padding}{colored_title}{' ' * right_padding}"
            f"{color}{style['vertical']}{Style.RESET_ALL}"
        )

        # Create separator line with exact width
        separator_line = f"{box_indent}{color}{style['left_t']}{horizontal_line}{style['right_t']}{Style.RESET_ALL}"
    else:
        title_line = None
        separator_line = None

    # Process and format content
    if isinstance(content, str):
        # Wrap text content into paragraphs with proper width
        content_lines = format_paragraphs(content, width=content_width)
    else:
        # Already a list of lines
        content_lines = content

    # Format each content line with proper alignment
    formatted_lines = []
    for line in content_lines:
        # Clean up the line (remove trailing whitespace)
        clean_line = line.rstrip()

        # Add color to content
        colored_line = f"{Fore.WHITE}{clean_line}{Style.RESET_ALL}"

        # Calculate the visible length (excluding ANSI codes)
        visible_len = visible_length(colored_line)

        # Calculate exact padding for right border alignment
        right_padding = content_width - visible_len
        padding = " " * max(0, right_padding)

        # Format with precise alignment - content should be aligned with exact right border
        formatted_line = (
            f"{box_indent}{color}{style['vertical']}{Style.RESET_ALL}"
            f"{colored_line}{padding}"
            f"{color}{style['vertical']}{Style.RESET_ALL}"
        )
        formatted_lines.append(formatted_line)

    # Assemble the complete box with all elements
    result = [top_border]
    if title_line:
        result.append(title_line)
    if separator_line:
        result.append(separator_line)
    result.extend(formatted_lines)
    result.append(bottom_border)

    return result


def print_box(title, content, style=BoxStyle.STANDARD, width=None, indent=0):
    """
    Print a formatted box with title and content.

    Args:
        title: The box title
        content: Content lines (string or list of strings)
        style: Box style to use
        width: Width of the box (auto-detected if None)
        indent: Additional indentation
    """
    lines = draw_box(title, content, style, width, indent)
    for line in lines:
        print(line)


def format_tool_call(name, args):
    """
    Format a tool call with colors and indentation.

    Args:
        name: Tool name
        args: Tool arguments as string

    Returns:
        Formatted tool call string
    """
    terminal_width = get_terminal_width()

    # Truncate arguments if too long
    max_args_length = terminal_width - len(name) - 20
    if len(args) > max_args_length and max_args_length > 30:
        display_args = args[: max_args_length - 20] + "..." + args[-15:]
    else:
        display_args = args

    return f"{Fore.CYAN}➤ {Fore.GREEN}{name}{Style.RESET_ALL}({Fore.YELLOW}{display_args}{Style.RESET_ALL})"


def format_tool_response(name, response):
    """
    Format a tool response with colors and indentation.

    Args:
        name: Tool name
        response: Tool response

    Returns:
        Formatted tool response string
    """
    terminal_width = get_terminal_width()

    # Clean and normalize the response
    clean_response = normalize_text(response)

    # For display in one line, replace newlines with spaces and collapse whitespace
    clean_response = re.sub(r"\s+", " ", clean_response).strip()

    # Determine if response needs truncation
    is_error = clean_response.startswith("Error:")
    color = Fore.RED if is_error else Fore.WHITE

    # Calculate available width for response text
    prefix = f"{Fore.BLUE}← {Fore.GREEN}{name}{Style.RESET_ALL}: "
    prefix_visible_length = len(name) + 4  # Account for arrow, colon and spaces

    # Calculate max response length to display
    max_resp_length = terminal_width - prefix_visible_length - 5  # Leave some margin

    # Handle JSON responses specially for better display
    is_json = False
    if (clean_response.startswith("{") and clean_response.endswith("}")) or (
        clean_response.startswith("[") and clean_response.endswith("]")
    ):
        is_json = True

    # Apply intelligent truncation
    if len(clean_response) > max_resp_length:
        if is_json:
            # For JSON, truncate to show structure but replace middle content with ellipsis
            if max_resp_length > 60:
                # Example input: {"foo": "bar", "baz": 123, "qux": [1,2,3]}
                # Example output: {"foo": "bar", ... "qux": [1,2,3]}

                # Find a point roughly 2/3 through the visible length to place the ellipsis
                split_point = int(max_resp_length * 0.6)
                # For JSON, make sure we don't split in the middle of a quoted string or property name
                # Find the last comma before the split point
                last_comma = clean_response.rfind(",", 0, split_point)
                if last_comma > 0:
                    # Found a good splitting point at a comma
                    display_resp = (
                        clean_response[: last_comma + 1]
                        + " ... "
                        + clean_response[-(max_resp_length - split_point - 5) :]
                    )
                else:
                    # No comma found, just do a clean split
                    display_resp = (
                        clean_response[:split_point] + "..." + clean_response[-20:]
                    )
            else:
                # For narrow terminals with JSON, show beginning and type indicator
                display_resp = clean_response[: max_resp_length - 5] + "..."
        else:
            # For normal text
            if max_resp_length > 60:
                # For longer terminals, show beginning and end
                start_len = int(max_resp_length * 0.6)  # Show more of the beginning
                end_len = max_resp_length - start_len - 5  # Space for ellipsis
                display_resp = (
                    clean_response[:start_len] + "..." + clean_response[-end_len:]
                )
            else:
                # For narrow terminals, just show beginning with ellipsis
                display_resp = clean_response[: max_resp_length - 5] + "..."
    else:
        display_resp = clean_response

    # Format the final response
    return f"{prefix}{color}{display_resp}{Style.RESET_ALL}"


def format_llm_response(content, is_final=False):
    """
    Format an LLM response with proper indentation and wrapping.

    Args:
        content: The LLM response content
        is_final: Whether this is a final answer (vs potential answer)

    Returns:
        Formatted LLM response string with box
    """
    # Normalize the content text
    clean_content = normalize_text(content)

    if is_final:
        # For final answers, use the success style for the table header
        # but present the actual content without the box for easy copy/paste
        style = BoxStyle.SUCCESS
        title = "FINAL ANSWER"

        # Create simplified header (top and bottom only, no middle separator)
        color = style["color"]
        # Get terminal width for consistent sizing but make it narrower
        width = get_terminal_width() - 16  # Reduce width by 10 more chars than standard
        content_width = width - 4

        # Create horizontal lines with exact width
        horizontal_line = f"{style['horizontal'] * content_width}"
        top_border = f"{color}{style['top_left']}{horizontal_line}{style['top_right']}{Style.RESET_ALL}"
        bottom_border = f"{color}{style['bottom_left']}{horizontal_line}{style['bottom_right']}{Style.RESET_ALL}"

        # Create colored title
        colored_title = f"{Fore.WHITE}{title}{Style.RESET_ALL}"
        title_visible_len = visible_length(colored_title)

        # Calculate padding for exact centering
        padding_total = content_width - title_visible_len
        left_padding = padding_total // 2
        right_padding = padding_total - left_padding

        # Format the title line with exact centering
        title_line = (
            f"{color}{style['vertical']}{Style.RESET_ALL}"
            f"{' ' * left_padding}{colored_title}{' ' * right_padding}"
            f"{color}{style['vertical']}{Style.RESET_ALL}"
        )

        # Return only top border, title line, bottom border and the content
        header = f"{top_border}\n{title_line}\n{bottom_border}"

        # Return the header followed by the clean content with proper spacing
        return header + "\n\n" + clean_content
    else:
        # For potential answers, keep the existing box format
        style = BoxStyle.LIGHT
        title = "POTENTIAL ANSWER"

        # Use the box drawing function to create the full boxed content
        box_lines = draw_box(title, clean_content, style=style)

        # Join and return the complete box
        return "\n".join(box_lines)


def format_tool_list(server_name, tools, indent=2):
    """
    Format a list of tools in a box.

    Args:
        server_name: Name of the server
        tools: List of tool objects with 'name' attribute
        indent: Indentation level

    Returns:
        List of formatted box lines
    """
    # Calculate box dimensions
    terminal_width = get_terminal_width()
    tool_names = [tool.name for tool in tools]
    max_name_len = max(len(name) for name in tool_names) if tool_names else 0

    # Title text (uncolored) for width calculation
    plain_title = f"Server '{server_name}' initialized with {len(tools)} tools:"

    # Determine minimum width needed for content
    min_width = max(max_name_len + 10, len(plain_title) + 4)
    # Cap at terminal width minus indentation and some margin
    box_width = min(terminal_width - indent - 4, max(min_width, 60))

    # Calculate usable width inside the box (between borders)
    content_width = box_width - 4  # -4 for left border, space, space, right border

    # Create the box decorations
    horizontal_line = "═" * content_width
    top_border = f"{' ' * indent}{Fore.MAGENTA}╔{horizontal_line}╗{Style.RESET_ALL}"
    bottom_border = f"{' ' * indent}{Fore.MAGENTA}╚{horizontal_line}╝{Style.RESET_ALL}"
    separator = f"{' ' * indent}{Fore.MAGENTA}╠{horizontal_line}╣{Style.RESET_ALL}"

    # Create the title with proper centering
    colored_title = f"{Fore.GREEN}Server '{server_name}'{Style.RESET_ALL} initialized with {Fore.CYAN}{len(tools)}{Style.RESET_ALL} tools:"

    # Center the title
    title_visible_length = visible_length(colored_title)
    padding_total = content_width - title_visible_length
    left_padding = padding_total // 2
    right_padding = padding_total - left_padding

    # Format properly aligned title line with exact centered positioning
    title_line = (
        f"{' ' * indent}{Fore.MAGENTA}║{Style.RESET_ALL}"
        f"{' ' * left_padding}{colored_title}{' ' * right_padding}"
        f"{Fore.MAGENTA}║{Style.RESET_ALL}"
    )

    # Format each tool with proper spacing and consistent alignment
    content_lines = []
    for name in tool_names:
        # Create the tool line with arrow indicator
        tool_display = f"{Fore.CYAN}▸ {Fore.WHITE}{name}{Style.RESET_ALL}"
        tool_visible_len = visible_length(tool_display)

        # Calculate right padding exactly
        right_padding = content_width - tool_visible_len
        padding = " " * max(0, right_padding)

        # Format with precise alignment
        line = f"{' ' * indent}{Fore.MAGENTA}║{Style.RESET_ALL}{tool_display}{padding}{Fore.MAGENTA}║{Style.RESET_ALL}"
        content_lines.append(line)

    # Assemble the full box with perfect alignment
    lines = [top_border, title_line, separator] + content_lines + [bottom_border]

    return lines


def format_verification_result(passed, feedback):
    """
    Format verification result with colors.

    Args:
        passed: Whether verification passed
        feedback: Feedback text

    Returns:
        Formatted verification result string
    """
    if passed:
        return f"\n{Fore.GREEN}✓ VERIFICATION PASSED:{Style.RESET_ALL} {feedback}"
    else:
        return f"\n{Fore.RED}✗ VERIFICATION FAILED:{Style.RESET_ALL} {feedback}"


def format_processing_step(step):
    """
    Format a processing step with subtle styling.

    Args:
        step: Step description

    Returns:
        Formatted processing step string
    """
    return f"{Fore.BLUE}■ {Fore.CYAN}{step}...{Style.RESET_ALL}"


def format_short_prompt(prompt, max_length=70):
    """
    Format a short prompt display for regular mode (not debug).

    Args:
        prompt: The user prompt to display
        max_length: Maximum length to display before truncating

    Returns:
        List with two lines for display
    """
    # Clean and normalize the prompt
    clean_prompt = normalize_text(prompt)

    # Truncate if needed
    if len(clean_prompt) > max_length:
        short_prompt = clean_prompt[: max_length - 3] + "..."
    else:
        short_prompt = clean_prompt

    # Format with nice styling
    header = f"{Fore.CYAN}┌─{Style.RESET_ALL} {Fore.YELLOW}Processing request:{Style.RESET_ALL}"
    detail = (
        f"{Fore.CYAN}└─►{Style.RESET_ALL} {Fore.WHITE}{short_prompt}{Style.RESET_ALL}"
    )

    return [header, detail]


def print_short_prompt(prompt, max_length=70):
    """
    Print a short prompt display for regular mode.

    Args:
        prompt: The user prompt to display
        max_length: Maximum length to display before truncating
    """
    lines = format_short_prompt(prompt, max_length)
    for line in lines:
        print(line)


def format_value(value, max_width):
    """
    Format a value with intelligent truncation if needed.

    Args:
        value: The value to format
        max_width: Maximum width

    Returns:
        Formatted value string
    """
    if len(value) > max_width:
        # For URLs, find a good breaking point for readability
        if "/" in value and "http" in value:
            # For URLs, try to break at a logical point like after a domain
            parts = value.split("/")
            if len(parts) > 3:  # http://domain.com/path
                # Keep protocol and domain, then add ...
                base_url = "/".join(parts[:3]) + "/"
                if len(base_url) <= max_width - 3:
                    return base_url + "..."

        # For general values, truncate at a clean point
        return value[: max_width - 3] + "..."
    return value


class ProgressSpinner:
    """Displays an animated spinner in the console during long operations"""

    def __init__(self, message, colors=True):
        self.message = message
        self.stop_event = threading.Event()
        self.spinner = itertools.cycle(
            ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        )
        self.spinner_thread = None
        self.colors = colors

    def spin(self):
        while not self.stop_event.is_set():
            prefix = f"{Fore.BLUE}" if self.colors else ""
            suffix = f"{Style.RESET_ALL}" if self.colors else ""
            sys.stdout.write(f"\r{prefix}{next(self.spinner)} {self.message}{suffix} ")
            sys.stdout.flush()
            time.sleep(0.1)

    def __enter__(self):
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        if self.spinner_thread:
            self.spinner_thread.join(timeout=0.5)
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")  # Clear the line
        sys.stdout.flush()


def print_llm_config(config_data, config_path):
    """
    Print the LLM configuration in a nice box.

    Args:
        config_data: Dictionary with configuration data
        config_path: Path to the config file
    """
    # Create rows for the configuration box
    rows = [
        f"Implementation:  {config_data.get('impl', 'custom')}",
        f"Model:           {config_data.get('model', '')}",
        f"API URL:         {format_value(config_data.get('url', ''), 45)}",
        f"Timeout:         {config_data.get('timeout', 180.0)}s",
        f"Strict Tools:    {config_data.get('strict_tools', 'default')}",
        f"Server Config:   {config_path}",
    ]

    # Print the configuration box
    print_box("LLM CONFIGURATION", rows, style=BoxStyle.STANDARD)
