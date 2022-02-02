import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
import gif


def bernstein_poly(i, m, t):
    """
     The Bernstein polynomial b_i,m(t)
    """
    return comb(m, i) * (t ** i) * (1 - t) ** (m - i)


p_start = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
           [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
           [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
           [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
           [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
           [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
p_circle = [[0, 0, 1], [0, 0.4, 1], [0, 0.8, 1], [0, 1.2, 1], [0, 1.6, 1], [0, 2, 1],
            [0, 0, 1], [-0.6736422, 0.4, 1], [-1.3472844, 0.8, 1], [-2.0209266, 1.2, 1], [-2.6945688, 1.6, 1],
            [-3.368211, 2, 1],
            [0, 0, 1], [-0.7900032, -0.88181, 1], [-1.5800064, -1.76362, 1], [-2.3700096, -2.64543, 1],
            [-3.1600128, -3.52724, 1],
            [-3.950016, -4.40905, 1],
            [0, 0, 1], [0.7900032, -0.88181, 1], [1.5800064, -1.76362, 1], [2.3700096, -2.64543, 1],
            [3.1600128, -3.52724, 1],
            [3.950016, -4.40905, 1],
            [0, 0, 1], [0.6736422, 0.4, 1], [1.3472844, 0.8, 1], [2.0209266, 1.2, 1], [2.6945688, 1.6, 1],
            [3.368211, 2, 1],
            [0, 0, 1], [0, 0.4, 1], [0, 0.8, 1], [0, 1.2, 1], [0, 1.6, 1], [0, 2, 1]]
p_cylinder = [[0, 2, 1], [0, 2, 2], [0, 2, 3], [0, 2, 4], [0, 2, 5], [0, 2, 6],
              [-3.368211, 2, 1], [-3.368211, 2, 2], [-3.368211, 2, 3], [-3.368211, 2, 4], [-3.368211, 2, 1],
              [-3.368211, 2, 6],
              [-3.950016, -4.40905, 1], [-3.950016, -4.40905, 2], [-3.950016, -4.40905, 3], [-3.950016, -4.40905, 4],
              [-3.950016, -4.40905, 5], [-3.950016, -4.40905, 6],
              [3.950016, -4.40905, 1], [3.950016, -4.40905, 2], [3.950016, -4.40905, 3], [3.950016, -4.40905, 4],
              [3.950016, -4.40905, 5], [3.950016, -4.40905, 6],
              [3.368211, 2, 1], [3.368211, 2, 2], [3.368211, 2, 3], [3.368211, 2, 4], [3.368211, 2, 5],
              [3.368211, 2, 6],
              [0, 2, 1], [0, 2, 2], [0, 2, 3], [0, 2, 4], [0, 2, 5], [0, 2, 6]]
p_cone = [[0, 2, 1], [0, 2, 2], [0, 2, 3], [0, 2, 4], [0, 2, 5], [0, 2, 6],
          [-3.368211, 2, 1], [-3.368211, 2, 2], [-3.368211, 2, 3], [-3.368211, 2, 4], [-3.368211, 2, 5],
          [-3.368211, 2, 6],
          [-3.950016, -4.40905, 1], [-3.950016, -4.40905, 2], [-3.950016, -4.40905, 3], [-3.950016, -4.40905, 4],
          [-3.950016, -4.40905, 5], [-3.950016, -4.40905, 6],
          [3.950016, -4.40905, 1], [3.950016, -4.40905, 2], [3.950016, -4.40905, 3], [3.950016, -4.40905, 4],
          [3.950016, -4.40905, 5], [3.950016, -4.40905, 6],
          [3.368211, 2, 1], [3.368211, 2, 2], [3.368211, 2, 3], [3.368211, 2, 4], [3.368211, 2, 5], [3.368211, 2, 6],
          [0, 2, 1], [0, 2, 2], [0, 2, 3], [0, 2, 4], [0, 2, 5], [0, 2, 6]]
for num in range(len(p_cone)):
    if num in [1, 7, 13, 19, 25, 31]:
        p_cone[num][0] = p_cone[num][0] * 0.8
        p_cone[num][1] = p_cone[num][1] * 0.8
    elif num in [2, 8, 14, 20, 26, 32]:
        p_cone[num][0] = p_cone[num][0] * 0.6
        p_cone[num][1] = p_cone[num][1] * 0.6
    elif num in [3, 9, 15, 21, 27, 33]:
        p_cone[num][0] = p_cone[num][0] * 0.4
        p_cone[num][1] = p_cone[num][1] * 0.4
    elif num in [4, 10, 16, 22, 28, 34]:
        p_cone[num][0] = p_cone[num][0] * 0.2
        p_cone[num][1] = p_cone[num][1] * 0.2
    elif num in [5, 11, 17, 23, 29, 35]:
        p_cone[num][0] = p_cone[num][0] * 0
        p_cone[num][1] = p_cone[num][1] * 0
p_house = [[-2, -2, 1], [0, -2, 1], [0, -2, 2], [1, -2, 2], [1, -2, 1], [3, -2, 1],
           [-2, -2, 1], [-2, -2, 6], [0, -2, 6], [1, -2, 6], [3, -2, 6], [3, -2, 1],
           [-2, -1.5, 1], [-2, -1.5, 6], [0, -1.5, 6], [1, -1.5, 6], [3, -1.5, 6], [3, -1.5, 1],
           [-2, 2.5, 1], [-2, 2.5, 6], [0, 2.5, 6], [1, 2.5, 6], [3, 2.5, 6], [3, 2.5, 1],
           [-2, 3, 1], [-2, 3, 6], [0, 3, 6], [1, 3, 6], [3, 3, 6], [3, 3, 1],
           [-2, 3, 1], [-1, 3, 1], [0, 3, 1], [1, 3, 1], [2, 3, 1], [3, 3, 1]]


def bezier_surface(points):
    vv = np.arange(0, 1.01, 0.05)
    uu = np.arange(0, 1.01, 0.05)
    v, u = np.meshgrid(vv, uu)
    base = []
    for x in range(6):
        for y in range(6):
            base.append(bernstein_poly(x, 5, v) * bernstein_poly(y, 5, u))
    x1 = 0
    x2 = 0
    x3 = 0
    for kk in range(len(points)):
        x1 += base[kk] * points[kk][0]
        x2 += base[kk] * points[kk][1]
        x3 += base[kk] * points[kk][2]

    ax = plt.axes(projection='3d')
    ax.scatter([points[ww][0] for ww in range(len(points))], [points[ww][1] for ww in range(len(points))],
               [points[ww][2] for ww in range(len(points))], color="k")  # control points

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([1, 6])
    ax.set_xlabel('x1', labelpad=20)
    ax.set_ylabel('x2', labelpad=20)
    ax.set_zlabel('x3', labelpad=20)
    return ax.plot_surface(x1, x2, x3, cmap=plt.cm.magma)


def transitions(start, end):
    """
    Function generating control points when transitioning between surfaces
    """
    p_final = []
    for step in np.arange(0.1, 1.05, 0.1):
        p_transitionx = []
        p_transtiony = []
        p_transitionz = []
        p_transition = []
        for i in range(len(start)):
            p_transitionx.append(start[i][0] * (1 - step) + end[i][0] * step)
            p_transtiony.append(start[i][1] * (1 - step) + end[i][1] * step)
            p_transitionz.append(start[i][2] * (1 - step) + end[i][2] * step)
        for j in range(len(start)):
            p_transition.append([p_transitionx[j], p_transtiony[j], p_transitionz[j]])
        p_final.append(p_transition)
    return p_final


for num in [transitions(p_start, p_circle), p_circle, transitions(p_circle, p_cylinder), p_cylinder,
            transitions(p_cylinder, p_cone), p_cone,
            transitions(p_cone, p_house), p_house]:
    if num in [transitions(p_start, p_circle), transitions(p_circle, p_cylinder), transitions(p_cylinder, p_cone),
               transitions(p_cone, p_house)]:
        for mm in num:
            bezier_surface(mm)
            plt.pause(0.2)
            plt.clf()
    elif num == p_house:
        bezier_surface(num)
        plt.pause(2)
        plt.clf()
    else:
        bezier_surface(num)
        plt.pause(0.2)
        plt.clf()


@gif.frame  # saving into a gif
def bezier_surface(points):
    vv = np.arange(0, 1.01, 0.05)
    uu = np.arange(0, 1.01, 0.05)
    v, u = np.meshgrid(vv, uu)
    base = []
    for x in range(6):
        for y in range(6):
            base.append(bernstein_poly(x, 5, v) * bernstein_poly(y, 5, u))
    x1 = 0
    x2 = 0
    x3 = 0
    for kk in range(len(points)):
        x1 += base[kk] * points[kk][0]
        x2 += base[kk] * points[kk][1]
        x3 += base[kk] * points[kk][2]

    ax = plt.axes(projection='3d')
    ax.scatter([points[ww][0] for ww in range(len(points))], [points[ww][1] for ww in range(len(points))],
               [points[ww][2] for ww in range(len(points))], color="k")  # control points

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([1, 6])
    ax.set_xlabel('x1', labelpad=20)
    ax.set_ylabel('x2', labelpad=20)
    ax.set_zlabel('x3', labelpad=20)
    return ax.plot_surface(x1, x2, x3, cmap=plt.cm.magma)


frames = []
for num in [transitions(p_start, p_circle), p_circle, transitions(p_circle, p_cylinder), p_cylinder,
            transitions(p_cylinder, p_cone), p_cone,
            transitions(p_cone, p_house), p_house]:
    if num in [transitions(p_start, p_circle), transitions(p_circle, p_cylinder), transitions(p_cylinder, p_cone),
               transitions(p_cone, p_house)]:
        for mm in num:
            frame = bezier_surface(mm)
            frames.append(frame)
    elif num == p_house:
        for _ in range(10):
            frame = bezier_surface(num)
            frames.append(frame)
    else:
        frame = bezier_surface(num)
        frames.append(frame)

gif.save(frames, 'bezier.gif', duration=7, unit="session", between="startend")
