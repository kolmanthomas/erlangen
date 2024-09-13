import numpy as np
import matplotlib.pyplot as plt
import affine_transform as at
import general_linear_group as glg

import draw
import shapes
import mobius
import conics
import util

"""
def first_optical_property_of_ellipse(ellipse: conics.Ellipse):
    fig, ax = plt.subplots()
    # Plot the ellipse
    t = np.linspace(-np.pi, np.pi, 100)
    x, y = ellipse.x(t), ellipse.y(t)
    plt.plot(x, y, 'r')
    # Plot the foci
    plt.plot(ellipse.f1[0], ellipse.f1[1], 'ro')
    plt.plot(ellipse.f2[0], ellipse.f1[1], 'ro')
    # Pick a point p_0 on the ellipse
    p0 = ellipse(0.5)
    # Create a line from p_0 to one of the foci
    shapes.Line(p0, ellipse.f1)
    # Plot line
    draw.plot_line(ax, shapes.Line(np.array([1, 1]), np.array([2, 2])))
    # Find intersection of line and ellipse
    points = conics.Ellipse(2, 1).intersection(shapes.Line(np.array([1, 1]), np.array([2, 2])))
    draw.plot_point(ax, points[0])
    draw.plot_point(ax, points[1], 'go')

    # Plot tangent line of ellipse
    m = ellipse.y_prime(points[1]) / ellipse.x_prime(points[1])
    shapes.Line.point_slope_form(points[1], m)
    plt.show()
"""


def shear_equilateral_triangle_into_right_angled_triangle():
    fig, ax = plt.subplots()
    p = np.array([[1, 1], [5, 3], [7, 1]])
    triangle = np.array([[1, 5, 7, 1], [1, 3, 1, 1]])

    a = at.AffineTransform(np.array([[1, -2], [0, 1]]), np.array([0, 0]))
    t = np.array([a(p[0]), a(p[1]), a(p[2])])
    new_triangle = np.array([[t[0, 0], t[1, 0], t[2, 0], t[0, 0]], [t[0, 1], t[1, 1], t[2, 1], t[0, 1]]])

    plt.plot(triangle[0], triangle[1], 'r')
    plt.plot(new_triangle[0], new_triangle[1], 'r')
    plt.show()

"""
def pappus_chain():
    fig, ax = plt.subplots() 
    # Create
    circ_1 = Circle(0, 0, 1)
    draw.plot_circle(ax, circ_1)

    a = InversiveTransform(1, 1)
    plt.show()
"""

def shoemakers_knife(fig, ax):
    # Create circle
    c1 = conics.Circle(xpos=0, ypos=0, r=1)

    #a1 = draw.present(fig, ax, c1)

    # Pick arbitrary point A
    A = np.array([-1, 0])
    B = np.array([0.5, 0])
    C = np.array([1, 0])
    #ax.plot(B[0], B[1], 'ro')
    c2 = conics.Circle.from_two_points(A, B)
    c3 = conics.Circle.from_two_points(B, C)
    #draw.plot_conic(ax, c2, 'r')
    #draw.plot_conic(ax, c3, 'r')

    inv_ref_circ = conics.Circle(A[0], A[1], 1)
    inv = mobius.InversiveTransform.from_circle(inv_ref_circ)

    # These are lines
    c1_inv = inv(c1)
    c2_inv = inv(c2)
    # and these are circles
    c3_inv = inv(c3)

    # Get the radius and center from c3_inv
    x_center, y_center = c3_inv.xpos, c3_inv.ypos
    r = c3_inv.r

    # distance 
    p1 = c1_inv.x(0)
    p2 = c2_inv.x(0)

    f = lambda t : conics.Circle(x_center, y_center + t * 2 * r, r)
    t_np = np.arange(-5, 5, 1)
    circles = []
    for t in t_np:
        circles.append(f(t))

    circles_inv = [inv(c) for c in circles]
    """
    [draw.plot_conic(ax, c, 'g') for c in circles]
    [draw.plot_conic(ax, c_inv, 'r') for c_inv in circles_inv]

    draw.draw(ax, c1_inv, 'g')
    draw.draw(ax, c2_inv, 'g')
    draw.draw(ax, c3_inv, 'g')
    """

    anim = draw.present(fig, ax, [[c1, c2, c3], [c1_inv, c2_inv, c3_inv]])
