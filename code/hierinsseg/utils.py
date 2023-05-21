import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from colors import colors

def convert_color_to_hexcode(rgb):
    r, g, b = rgb
    return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

def render_pc_set(out_fn, pcs, \
        subplotsize=(1, 1), figsize=(8, 8), azim=60, elev=20, scale=0.3):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(subplotsize[0], subplotsize[1], 1, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    xs = []; ys = []; zs = [];
    for i in range(len(pcs)):
        x = pcs[i][:, 0]
        y = pcs[i][:, 2]
        z = pcs[i][:, 1]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        ax.scatter(x, y, z, marker='.', c=convert_color_to_hexcode(colors[i % len(colors)]))
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    zs = np.concatenate(zs, axis=0)
    miv = np.min([np.min(xs), np.min(ys), np.min(zs)])
    mav = np.max([np.max(xs), np.max(ys), np.max(zs)])
    ax.set_xlim(miv, mav)
    ax.set_ylim(miv, mav)
    ax.set_zlim(miv, mav)
    plt.tight_layout()
    fig.savefig(out_fn, bbox_inches='tight')
    plt.close(fig)

