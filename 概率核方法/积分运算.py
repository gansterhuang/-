import sympy

from sympy import integrate, cos, sin
from sympy.abc import a, x, y,w,u,theta
from sympy import *

con=integrate(E**((x**2+(w*x+u)**2) / -2*(theta**2)), (x, -float("inf"), float("inf")))

print(con)