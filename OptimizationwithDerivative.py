import matplotlib.pyplot as plt
import numpy as np

"""
1. Function Definition and Data Generation:

    def f(x):: Defines a function f(x) that returns 2 times the square of the input value x.
    x = np.arange(0,50,0.0001): Creates a NumPy array x containing values ranging from 0 to 49 (exclusive) with a very small step size of 0.0001. This generates a high-resolution set of x-coordinates for the function.
    y = f(x): Calculates the corresponding y-values for each x-value in the x array using the f(x) function. This creates the data points for the function.
    colors = ['k','g','r','b','c']: Defines a list of colors to be used for plotting different tangent lines.

2. Plotting the Original Function:

    plt.plot(x,y): Creates a line plot of the function f(x) using the x and y arrays. This visually represents the function's behavior.

3. Tangent Line Approximation Function:

    def approximate_tangent_line(x, approximate_derivatives, b): This function defines how to calculate the equation of an approximated tangent line for a function f(x) at a given point x.
        approximate_derivatives: Represents the slope of the approximated tangent line.
        b: Represents the y-intercept of the approximated tangent line.
        The function returns the equation of the tangent line: approximate_derivatives * x + b.

4. Looping for Multiple Approximations:

    The for loop iterates five times (range(5)):
        p2_delta = 0.001: Defines a small delta value used to calculate the derivative numerically.
        In each iteration, x1 is set to a different value (i) ranging from 0 to 4 (since the loop iterates five times).
        x2 = x1 + p2_delta: Calculates a slightly higher x-value (x2) based on x1 and the delta.
        y1 = f(x1) and y2 = f(x2): Calculates the corresponding y-values for x1 and x2 using the f(x) function. These points lie on the curve of the function.
        Prints the coordinates of the two points used for the approximation: (x1, y1), (x2, y2).
        approximate_derivatives = (y2 - y1) / (x2 - x1): Calculates the approximate slope of the tangent line by finding the slope between the two chosen points. This is a numerical approximation of the derivative.
        b = y2 - approximate_derivatives * x2: Calculates the y-intercept of the tangent line using the point-slope form (y = mx + b). It leverages the approximate slope (approximate_derivatives) and one of the points (x2, y2).
        to_plot = [x1-0.9, x1, x1+0.9]: Creates an array of x-values to be used for plotting the tangent line. This extends the line slightly beyond the chosen point (x1).
        plt.scatter(x1, y1, c=colors[i]): Plots a scatter point at the chosen point (x1, y1) on the function's curve using the color from the colors list (different color for each iteration).
        plt.plot(to_plot, [approximate_tangent_line(point, approximate_derivatives, b) for point in to_plot], c= colors[i]): Plots the approximated tangent line using the calculated slope (approximate_derivatives), y-intercept (b), and the array of x-values (to_plot). It uses the same color as the corresponding scatter point.
        Prints the approximate derivative for f(x) at the chosen point x1.

5. Displaying the Plot:

    plt.show(): Displays the generated plot, which shows the original function, the scattered points where the tangent lines are approximated, and the approximated tangent lines themselves in different colors.
"""

def f(x):
    return 2*x**2

x = np.arange(0,50,0.0001)
y = f(x)
colors = ['k','g','r','b','c']
plt.plot(x,y)

def approximate_tangent_line(x, approximate_derivatives, b):
    return approximate_derivatives*x+b

for i in range(5):

    p2_delta = 0.001
    x1 =i
    x2 = x1+p2_delta

    y1 = f(x1)
    y2 = f(x2)

    print((x1,y1), (x2,y2))

    approximate_derivatives = (y2-y1)/(x2-x1)
    b = y2-approximate_derivatives*x2
    to_plot = [x1-0.9, x1, x1+0.9]

    plt.scatter(x1,y1, c=colors[i])
    plt.plot(to_plot, [approximate_tangent_line(point, approximate_derivatives, b) for point in to_plot], c= colors[i])

    print("Approximate derivative for f(x)", f"where x= {x1} is {approximate_derivatives}")

plt.show()