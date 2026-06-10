import math
from shapely.geometry import Polygon, box
from shapely.validation import make_valid
from shapely.affinity import rotate

def get_largest_inscribed_rectangle(geom, angle):
    """
    Given a polygon (geom) and its dominant angle, find the largest
    inscribed rectangle aligned with that angle.
    """
    if geom.is_empty:
        return geom
        
    centroid = geom.centroid
    cx, cy = centroid.x, centroid.y

    def rot(x, y, a):
        c, s = math.cos(a), math.sin(a)
        return c * (x - cx) - s * (y - cy) + cx, s * (x - cx) + c * (y - cy) + cy

    # Rotate polygon to axis-aligned frame
    rot_coords = [rot(x, y, -angle) for x, y in geom.exterior.coords]
    rot_geom = make_valid(Polygon(rot_coords))
    
    if rot_geom.is_empty:
        return geom

    # Get bounds
    minx, miny, maxx, maxy = rot_geom.bounds
    
    # Collect all unique x/y coords from polygon vertices
    xs = set([minx, maxx])
    ys = set([miny, maxy])
    for x, y in rot_geom.exterior.coords:
        xs.add(round(x, 6))
        ys.add(round(y, 6))
        
    xs_sorted = sorted(list(xs))
    ys_sorted = sorted(list(ys))
    
    nx = len(xs_sorted) - 1
    ny = len(ys_sorted) - 1
    
    if nx <= 0 or ny <= 0:
        return geom

    # Build occupancy grid
    grid = [[False] * nx for _ in range(ny)]
    from shapely.geometry import Point
    for row in range(ny):
        for col in range(nx):
            cx_cell = (xs_sorted[col] + xs_sorted[col + 1]) / 2
            cy_cell = (ys_sorted[row] + ys_sorted[row + 1]) / 2
            if rot_geom.contains(Point(cx_cell, cy_cell)):
                grid[row][col] = True

    # DP for largest rectangle
    best_area = 0
    best_rect = None

    heights = [0] * nx
    for row in range(ny):
        for col in range(nx):
            if grid[row][col]:
                heights[col] += (ys_sorted[row+1] - ys_sorted[row])
            else:
                heights[col] = 0

        stack = []
        for col in range(nx + 1):
            h = heights[col] if col < nx else 0
            start_col = col
            while stack and stack[-1][1] > h:
                prev_col, prev_h = stack.pop()
                w = xs_sorted[col] - xs_sorted[prev_col]
                area = prev_h * w
                if area > best_area:
                    best_area = area
                    best_rect = (xs_sorted[prev_col], ys_sorted[row+1] - prev_h, xs_sorted[col], ys_sorted[row+1])
                start_col = prev_col
            stack.append((start_col, h))

    if best_rect is None:
        return geom

    rx1, ry1, rx2, ry2 = best_rect
    corners = [(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2), (rx1, ry1)]
    back_corners = [rot(x, y, angle) for x, y in corners]
    return Polygon(back_corners)

if __name__ == "__main__":
    p1 = box(0, 0, 10, 10)
    p2 = box(5, 5, 15, 15)
    diff = p1.difference(p2) # L-shape: 0,0 to 10,10 minus 5,5 to 15,15
    rect = get_largest_inscribed_rectangle(diff, 0)
    print("Diff Area:", diff.area)
    print("Largest Inscribed Rect Area:", rect.area)
    print("Largest Inscribed Rect Bounds:", rect.bounds)
