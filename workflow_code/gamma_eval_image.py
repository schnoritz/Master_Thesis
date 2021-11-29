import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


def latex(width, height, path):
    '''Decorator that sets latex parameters for plt'''
    def do_latex(func):
        def wrap(*args, **kwargs):

            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = "Palatino"
            plt.rcParams["font.size"] = 11
            fig = func(*args, **kwargs)
            cm = 1/2.54
            fig.set_size_inches(width*cm, height*cm, forward=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
        return wrap
    return do_latex


@latex(width=25, height=25, path="/Users/simongutwein/Desktop/gamma_plot.pdf")
def main():

    class Arrow3D(FancyArrowPatch):

        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    fig = plt.figure(1, figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=25., azim=45)

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    x_circ1 = np.cos(u)
    y_circ1 = np.sin(u)
    z_circ1 = np.zeros(len(y_circ1))

    ax.plot_surface(x, y, z, linewidth=1, alpha=0.1, color="darkblue")  # , antialiased=False)  # ,  alpha=0.5, color="red", zorder=-1)  # alpha=0.5

    ax.scatter3D(0, 0, 0, marker="8", color="k")
    a = Arrow3D([0, 0], [-2, 2], [0, 0], mutation_scale=10, lw=1, arrowstyle="<|-|>", color="k")
    ax.add_artist(a)
    a = Arrow3D([-2, 2], [0, 0], [0, 0], mutation_scale=10, lw=1, arrowstyle="<|-|>", color="k")
    ax.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [-1.7, 1.7], mutation_scale=10, lw=1, arrowstyle="<|-|>", color="k")
    ax.add_artist(a)

    pos = [0.2, -0.4, 0.6]
    pos2 = [-0.6, 0.6, 1.3]

    #ax.text3D(pos2[0], pos2[1]-0.08, pos2[2]+0.08, r'$r_{failed}$', ha="center", fontsize=11, zorder=10)
    #ax.scatter(pos2[0], pos2[1], pos2[2], marker="X", color="k")
    #a = Arrow3D([0, pos2[0]+0.01], [0, pos2[1]+0.01], [0, pos2[2]+0.01], mutation_scale=10, lw=2, arrowstyle="-|>", color="red")
    # ax.add_artist(a)

    ax.plot3D(x_circ1, y_circ1, z_circ1, zorder=1, color="Orange")
    ax.plot3D(z_circ1, x_circ1, y_circ1, zorder=2, color="Maroon")

    ax.text3D(0, 2.1, 0.2, "Y-Plane", ha="center", zorder=10)
    ax.text3D(2.1, 0, 0.2, "X-Plane", ha="center", zorder=10)
    ax.text3D(0, 0, 1.8, "Dose-Value", ha="center", zorder=10)
    ax.text3D(0.15, 0, -0.15, r'$r_m$', ha="center", fontsize=11)
    #ax.text3D(pos[0], pos[1]-0.08, pos[2]+0.08, r'$r_{passed}$', ha="left", fontsize=11, zorder=10)
    #ax.scatter(pos[0], pos[1], pos[2], marker="X", color="k")
    #a = Arrow3D([0, pos[0]+0.01], [0, pos[1]+0.01], [0, pos[2]+0.01], mutation_scale=10, lw=2, arrowstyle="-|>", color="green")
    # ax.add_artist(a)

    a = Arrow3D([0, np.cos(3*np.pi/4)], [0, np.sin(3*np.pi/4)], [0, 0], mutation_scale=10, lw=2, arrowstyle="-|>", linestyle='--', color="k")
    ax.add_artist(a)
    ax.text3D(np.cos(3*np.pi/4)-0.1, np.sin(3*np.pi/4)+0.1, 0, r'$\Delta d_M$', ha="center", fontsize=11)

    a = Arrow3D([0, 0],  [0, np.sin(3*np.pi/4)], [0, np.cos(3*np.pi/4)], mutation_scale=10, lw=2, arrowstyle="-|>", linestyle='--', color="k", zorder=0)
    ax.add_artist(a)
    ax.text3D(0, np.sin(3*np.pi/4)+0.1, np.cos(3*np.pi/4)-0.1, r'$\Delta D_M$', ha="center", fontsize=11)

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-1.5, 1.5])

    ax.axis("off")
    plt.tight_layout()
    return fig


if __name__ == "__main__":

    main()
