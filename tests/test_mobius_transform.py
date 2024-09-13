import numpy as np
from erlangen.mobius import MobiusTransform, InversiveTransform
import erlangen.erlangen as erl

def test_call_mobius_transform():
    mobius = lambda z : (2 * z + 1j) / (3 * z + 2j)
    mobius_class = MobiusTransform(2, 1j, 3, 2j)
    assert np.isclose(mobius(1), mobius_class(1))
     
    
def test_inversive_transform_on_line_not_through_origin():
    circ = erl.Circle(0, 0, 1)
    line = erl.Line(
        np.array([0, 1]),
        np.array([1, 1])
    )
    inv = InversiveTransform.from_circle(circ)
    inv_circ = inv(line)
    inv_circ_expected = erl.Circle(xpos=0, ypos=0.5, r=0.5)
    assert inv_circ == inv_circ_expected
    
