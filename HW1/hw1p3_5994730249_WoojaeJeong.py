import sys
from func import f

try:
    a = float(sys.argv[1])
    b = float(sys.argv[2])

except ValueError:
    print("Range error", file = sys.stderr)
    sys.exit(1)

if a >= b:
    print("Range error", file = sys.stderr)
    sys.exit(1)

elif f(a)*f(b) >= 0:
    print("Range error", file = sys.stderr)
    sys.exit(1)

tol = 1e-10

def secant_method(f, x0, x1, tol):
    N = 0
    idx = 0

    while idx == 0:
         N += 1
         x2 = x1 - (f(x1)*(x1 - x0)) / (f(x1)-f(x0))
         
         if abs(x2-x1) < tol:
             idx = 1
             return N, x0, x1, x2
         x0, x1 = x1, x2

N, x, y, z = secant_method(f, a, b, tol)

print(N)
print(x)
print(y)
print(z)
