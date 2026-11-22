"""The loop the robot drives in the hall `open` (30 x 20 m): a rounded rectangle, counter-clockwise.

    lower straight   y = 7,  x 10 -> 20   (the start, heading 0)
    right straight   x = 21, y 8 -> 12
    upper straight   y = 13, x 20 -> 10   (the tunnel of config/tunnel.json begins at x = 12)
    left straight    x = 9,  y 12 -> 8    (the robot stops at y = 10 every lap, still in the tunnel)

The same numbers are in config/*.json (spawn, zones); test/test_lab.py checks that they agree.
"""
import math

CENTRE = (15.0, 10.0)
SIZE = (12.0, 6.0)                      # m, outer width and height of the loop
RADIUS = 1.0                            # m, corner radius
START = (12.0, 7.0, 0.0)                # x, y, heading of the spawn, on the lower straight
STOP = (9.0, 10.0)                      # middle of the left straight


def loop(step: float = 0.05) -> list:
    """The loop as a sequence of points.

    Args:
        step: Distance between two points in m.

    Returns:
        List of (x, y, heading): position in m, direction of travel in rad, starting at START and going
        counter-clockwise.
    """
    (cx, cy), (w, h), r = CENTRE, SIZE, RADIUS
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    pieces = [("line", (x0 + r, y0), (x1 - r, y0)), ("arc", (x1 - r, y0 + r), -math.pi / 2),
              ("line", (x1, y0 + r), (x1, y1 - r)), ("arc", (x1 - r, y1 - r), 0.0),
              ("line", (x1 - r, y1), (x0 + r, y1)), ("arc", (x0 + r, y1 - r), math.pi / 2),
              ("line", (x0, y1 - r), (x0, y0 + r)), ("arc", (x0 + r, y0 + r), math.pi)]
    points = []
    for kind, a, b in pieces:
        if kind == "line":
            length = math.dist(a, b)
            heading = math.atan2(b[1] - a[1], b[0] - a[0])
            n = max(1, round(length / step))
            points += [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, heading) for i in range(n)]
        else:
            n = max(1, round(r * math.pi / 2 / step))
            for i in range(n):
                phi = b + math.pi / 2 * i / n
                points.append((a[0] + r * math.cos(phi), a[1] + r * math.sin(phi), phi + math.pi / 2))
    first = min(range(len(points)), key=lambda i: math.dist(points[i][:2], START[:2]))
    return points[first:] + points[:first]


def length() -> float:
    """Length of the loop.

    Returns:
        Length in m: four straights and four quarter circles of RADIUS.
    """
    (w, h), r = SIZE, RADIUS
    return 2 * (w + h) - 8 * r + 2 * math.pi * r
