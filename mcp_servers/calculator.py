# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "mcp>=1.6.0",
# ]
# ///

import math
import argparse
from mcp.server.fastmcp import FastMCP
from typing import Union

app = FastMCP("Calculator")


@app.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The sum of a and b.
    """
    return a + b


@app.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a.

    Args:
        a (float): The number to subtract from.
        b (float): The number to subtract.

    Returns:
        float: The result of a - b.
    """
    return a - b


@app.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The product of a and b.
    """
    return a * b


@app.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b.

    Args:
        a (float): The dividend.
        b (float): The divisor.

    Returns:
        float: The result of a / b.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@app.tool()
def power(base: float, exponent: float) -> float:
    """Calculate base raised to the power of exponent.

    Args:
        base (float): The base number.
        exponent (float): The exponent.

    Returns:
        float: base raised to the power of exponent.
    """
    return math.pow(base, exponent)


@app.tool()
def sqrt(a: float) -> float:
    """Calculate the square root of a.

    Args:
        a (float): The number to find the square root of.

    Returns:
        float: The square root of a.

    Raises:
        ValueError: For negative input.
    """
    if a < 0:
        raise ValueError("Cannot calculate the square root of a negative number.")
    return math.sqrt(a)


@app.tool()
def log(a: float, base: float = math.e) -> float:
    """Calculate the logarithm of a with the specified base.

    Args:
        a (float): The number to calculate the logarithm of.
        base (float, optional): The base of the logarithm. Defaults to math.e (natural log).

    Returns:
        float: The logarithm of a with the given base.

    Raises:
        ValueError: For non-positive input for a.
    """
    if a <= 0:
        raise ValueError("Cannot calculate the logarithm of a non-positive number.")
    return math.log(a, base)


@app.tool()
def sin(angle: float) -> float:
    """Calculate the sine of an angle.

    Args:
        angle (float): The angle in radians.

    Returns:
        float: The sine of the angle.
    """
    return math.sin(angle)


@app.tool()
def cos(angle: float) -> float:
    """Calculate the cosine of an angle.

    Args:
        angle (float): The angle in radians.

    Returns:
        float: The cosine of the angle.
    """
    return math.cos(angle)


@app.tool()
def tan(angle: float) -> float:
    """Calculate the tangent of an angle.

    Args:
        angle (float): The angle in radians.

    Returns:
        float: The tangent of the angle.
    """
    # Optional: Add handling for angles where tan is undefined (e.g., pi/2, 3*pi/2, ...)
    # if math.isclose(math.cos(angle), 0):
    #     raise ValueError("Tangent is undefined for this angle.")
    return math.tan(angle)


@app.tool()
def sind(angle_degrees: float) -> float:
    """Calculate the sine of an angle given in degrees.

    Args:
        angle_degrees (float): The angle in degrees.

    Returns:
        float: The sine of the angle.
    """
    return math.sin(math.radians(angle_degrees))


@app.tool()
def cosd(angle_degrees: float) -> float:
    """Calculate the cosine of an angle given in degrees.

    Args:
        angle_degrees (float): The angle in degrees.

    Returns:
        float: The cosine of the angle.
    """
    return math.cos(math.radians(angle_degrees))


@app.tool()
def tand(angle_degrees: float) -> float:
    """Calculate the tangent of an angle given in degrees.

    Args:
        angle_degrees (float): The angle in degrees.

    Returns:
        float: The tangent of the angle.
    """
    return math.tan(math.radians(angle_degrees))


@app.tool()
def asin(value: float) -> float:
    """Calculate the arcsine (inverse sine) in radians.

    Args:
        value (float): The value whose arcsine is to be computed, must be between -1 and 1.

    Returns:
        float: The principal value of the arcsine in radians, between -pi/2 and pi/2.

    Raises:
        ValueError: If the input value is outside the range [-1, 1].
    """
    if not -1.0 <= value <= 1.0:
        raise ValueError("Input for asin must be between -1 and 1.")
    return math.asin(value)


@app.tool()
def acos(value: float) -> float:
    """Calculate the arccosine (inverse cosine) in radians.

    Args:
        value (float): The value whose arccosine is to be computed, must be between -1 and 1.

    Returns:
        float: The principal value of the arccosine in radians, between 0 and pi.

    Raises:
        ValueError: If the input value is outside the range [-1, 1].
    """
    if not -1.0 <= value <= 1.0:
        raise ValueError("Input for acos must be between -1 and 1.")
    return math.acos(value)


@app.tool()
def atan(value: float) -> float:
    """Calculate the arctangent (inverse tangent) in radians.

    Args:
        value (float): The value whose arctangent is to be computed.

    Returns:
        float: The principal value of the arctangent in radians, between -pi/2 and pi/2.
    """
    return math.atan(value)


@app.tool()
def atan2(y: float, x: float) -> float:
    """Calculate the arctangent of y/x in radians, considering the signs of x and y to determine the quadrant.

    Args:
        y (float): The y-coordinate.
        x (float): The x-coordinate.

    Returns:
        float: The angle in radians between the positive x-axis and the point (x, y), between -pi and pi.
    """
    return math.atan2(y, x)


@app.tool()
def ceil(a: float) -> float:
    """Calculate the ceiling of a number.

    Args:
        a (float): The number.

    Returns:
        float: The smallest integer greater than or equal to a, returned as a float.
    """
    return float(math.ceil(a))


@app.tool()
def floor(a: float) -> float:
    """Calculate the floor of a number.

    Args:
        a (float): The number.

    Returns:
        float: The largest integer less than or equal to a, returned as a float.
    """
    return float(math.floor(a))


@app.tool()
def absolute(a: float) -> float:
    """Calculate the absolute value of a number.

    Args:
        a (float): The number.

    Returns:
        float: The absolute value of a.
    """
    return math.fabs(a)


@app.tool()
def factorial(n: int) -> int:
    """Calculate the factorial of a non-negative integer.

    Args:
        n (int): The non-negative integer.

    Returns:
        int: The factorial of n (n!).

    Raises:
        ValueError: For negative or non-integer input.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("Factorial is only defined for non-negative integers.")
    return math.factorial(n)


@app.tool()
def to_degrees(radians: float) -> float:
    """Convert angle from radians to degrees.

    Args:
        radians (float): The angle in radians.

    Returns:
        float: The angle converted to degrees.
    """
    return math.degrees(radians)


@app.tool()
def to_radians(degrees: float) -> float:
    """Convert angle from degrees to radians.

    Args:
        degrees (float): The angle in degrees.

    Returns:
        float: The angle converted to radians.
    """
    return math.radians(degrees)


@app.tool()
def pi() -> float:
    """Return the value of the mathematical constant pi.

    Returns:
        float: The value of pi.
    """
    return math.pi


@app.tool()
def e() -> float:
    """Return the value of the mathematical constant e.

    Returns:
        float: The value of e.
    """
    return math.e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Calculator MCP server.")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["sse", "stdio"],
        default="stdio",
        help="Transport method to use (sse or stdio).",
    )
    args = parser.parse_args()

    app.run(transport=args.transport)
