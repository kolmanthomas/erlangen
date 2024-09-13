from manim import *
import numpy as np
import numpy.typing as npt

from typing import List
from time import sleep

import mobius
import util

def three_points_from_line(line: Line) -> npt.NDArray[np.complex_]:
    x = lambda t : t * line.get_unit_vector()[0] + line.get_x()
    y = lambda t : t * line.get_unit_vector()[1] + line.get_y()
    return np.array([complex(x(0), y(0)), complex(x(1), y(1)), complex(x(2), y(2))])

def three_points_from_circle(circle: Circle) -> npt.NDArray[np.complex_]:
    return np.array([
        complex(circle.point_at_angle(0)[0], circle.point_at_angle(0)[1]), 
        complex(circle.point_at_angle(PI/2)[0], circle.point_at_angle(PI/2)[1]), 
        complex(circle.point_at_angle(PI)[0], circle.point_at_angle(PI)[1]) 
        ])

def get_three_points(shape: Circle | Line | List[Circle | Line] | List[Circle] | List[Line] ) -> npt.NDArray[np.complex_]:
    """
        Returns a point tensor.        
    """
    p = np.array([three_points_from_circle(s) if isinstance(s, Circle) else three_points_from_line(s) for s in shape])
    return p

def get_shape_from_three_points(points: npt.NDArray) -> List[Circle | Line]:
    """
        Args:
            points: 
    """
    util.assert_is_3D(points)

    shapes = []

    for p in points:
        if util.is_collinear_3D(p[0][0:2], p[1][0:2], p[2][0:2]):
            shapes.append(Line(start=p[0], end=p[1]))
        else:
            shapes.append(Circle.from_three_points(p[0], p[1], p[2]))

    return shapes

def inversive_transform(ref_circle: Circle, shapes: Circle | Line | List[Circle] | List[Circle | Line]) -> List[Circle | Line]:
    ref_circle_x = ref_circle.get_center()[0]
    ref_circle_y = ref_circle.get_center()[1]
    inv = mobius.InversiveTransform(complex(ref_circle_x, ref_circle_y), ref_circle.radius)

    p = get_three_points(shapes) # 2D complex tensor
    p = np.ndarray.flatten(p) # 1D complex tensor
    p = inv(p) # 1D complex tensor
    p = util.complex_to_real(p) # 2D real tensor
    p = util.point_2D_to_3D(p) # 2D real tensor
    p = np.reshape(p, (-1, 3, 3)) # 3D real tensor

    # Find if shape converts to circle or line
    shape_mask = []
    for shape in shapes:
        if isinstance(shape, Circle):
            print(np.linalg.norm(np.array([shape.get_x(), shape.get_y()])))
            if np.isclose(np.linalg.norm(np.array([shape.get_x() - ref_circle_x, shape.get_y() - ref_circle_y])), shape.radius):
                shape_mask.append(1)
            else:
                shape_mask.append(0)
        else:
            if np.isclose(shape.get_y() - shape.get_slope() * shape.get_x(), 0):
                shape_mask.append(1)
            else:
                shape_mask.append(0)
    print(shape_mask)

    new_shapes = []
    for i in range(0, p.shape[0]):
        if shape_mask[i] == 0:
            new_shapes.append(Circle.from_three_points(p[i][0], p[i][1], p[i][2]))
        else:
            new_shapes.append(Line(start=p[i][0], end=p[i][1]).set_length(20))

    return new_shapes


class CreateCircle(Scene):
    def construct(self):
        r0 = 5
        r2 = r0 * 2/5
        r3 = r0 = 9/10
        # Circle that inversion is with respect to
        c = Circle(color=BLUE, radius=2).shift(4 * LEFT)
        # These are the three "starting" circles
        c1 = Circle(color=RED, radius=4)
        c1_copy = c1.copy()
        c2 = Circle(color=RED, radius=2.5).shift(1.5*LEFT)
        c2_copy = c2.copy()
        c3 = Circle(color=RED, radius=1.5).shift(2.5*RIGHT)

        # And these are the ith circles that touch c1, c2, and c(i - 1)
        c_invs = inversive_transform(ref_circle=c, shapes=[c1, c2, c3])
        # c_inv[2] here is the circle under inversion that becomes a circle, unlike c_invs[0] and c_invs[1] which become lines
        center = c_invs[2].get_center()
        r = c_invs[2].radius
        c_invs[2].set_color(WHITE)
        t_list = np.arange(-15, 15, 1)
        t_list = np.delete(t_list, 15)
        print(t_list)
        new_circles = [Circle(color=WHITE, radius=r).move_to(center + UP*2*t*r) for t in t_list]
        inv_new_circles = inversive_transform(c, new_circles)

        # Create the starting circles
        g1 = VGroup(c1, c2, c3, *inv_new_circles)
        self.play(Create(g1))

        # Create the reference inversion circle
        self.play(Create(c))
        self.play(Transform(c1, c_invs[0]))
        self.play(Transform(c2, c_invs[1]))
        c_invs_copy = [i.copy() for i in c_invs]

        g2 = VGroup(c3, *inv_new_circles)
        g2_copy = g2.copy()
        g3 = VGroup(c_invs[2], *new_circles)
        self.play(Transform(g2, g3))

        #l = Line(start=center, end=center + UP*2*r).set_length(20)
        #self.play(Create(l))

        #inv_l = inversive_transform(c, l)
        #self.play(Transform(l, *inv_l))
        self.play(Transform(c1, c1_copy))
        self.play(Transform(c2, c2_copy))
        self.play(Transform(g2, g2_copy))

        self.wait(3)

        """
        self.play(Transform(c_invs[0], c1))
        self.play(Transform(c_invs[1], c2))

        for circ in new_circles:
            self.play(Create(circ))

        for i in range(0, len(new_circles)):
            self.play(Transform(new_circles[i], inv_new_circles[i]))
        """

        """
        self.play(Create(c4))
        self.play(Transform(c4, c4_inv))

        c1 = Circle(color=RED, radius=2)
        c2 = Circle(color=RED, radius=1.5).shift(1/2 * LEFT)
        c3 = Circle(color=RED, radius=0.5).shift(1.5 * RIGHT)
        self.play(Transform(l1, c1))
        self.play(Transform(l2, c2))
        self.play(Transform(c3_inv, c3))
        """

        

"""
with tempconfig({"quality" : "medium_quality", "disable_caching" : True}):
    scene = CreateCircle()
    scene.render()
"""
