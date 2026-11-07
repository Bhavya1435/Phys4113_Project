"""Shared full-orbit tools. Coordinates and parameters use normalized units."""
from pathlib import Path
import json
import platform
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOLERANCES = ((1e-8, 1e-10), (1e-10, 1e-12), (1e-12, 1e-14))

def zero_field(t, position):
    return np.zeros_like(position)

def bottle_field(t, position, B0=1., L=10.):
    """Cartesian form, regular on axis; div B = curl B = 0."""
    x, y, z = np.moveaxis(np.asarray(position), -1, 0)
    return B0 * np.stack((-x*z/L**2, -y*z/L**2,
                         1 + (z*z-(x*x+y*y)/2)/L**2), axis=-1)

def integrate(initial, duration, magnetic, electric=zero_field,
              rtol=1e-12, atol=1e-14, q=1., m=1., events=None):
    def rhs(t, state):
        return np.r_[state[3:], q/m * (electric(t, state[:3]) +
                                     np.cross(state[3:], magnetic(t, state[:3])))]
    sol = solve_ivp(rhs, (0., duration), initial, method='DOP853',
                    rtol=rtol, atol=atol, dense_output=True, events=events)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol

def diagnostics(states, magnetic, m=1., q=1.):
    x, v = states[:, :3], states[:, 3:]
    B = magnetic(0., x)
    strength = np.linalg.norm(B, axis=1)
    b = B / strength[:, None]
    u = np.sum(v*b, axis=1)
    vp = v - u[:, None]*b
    energy = m*np.sum(v*v, axis=1)/2
    mu = m*np.sum(vp*vp, axis=1)/(2*strength)
    # x = X + rho, rho = b x v_perp / Omega.
    center = x + m/q*np.cross(v, B)/strength[:, None]**2
    return energy, mu, center

def linear_fit(t, position):
    slope, intercept = np.polyfit(t, position, 1)
    return float(slope), float(intercept)

def style():
    plt.rcParams.update({'font.size': 15, 'axes.labelsize': 17,
                         'legend.fontsize': 12, 'axes.titlesize': 16,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'lines.linewidth': 1.7, 'figure.constrained_layout.use': True})

def save_figure(fig, name):
    folder = ROOT / 'figures'
    folder.mkdir(exist_ok=True)
    for extension in ('pdf', 'png'):
        fig.savefig(folder / f'{name}.{extension}', dpi=350, bbox_inches='tight')
    plt.close(fig)

def save_json(name, data):
    folder = ROOT / 'results'
    folder.mkdir(exist_ok=True)
    data['environment'] = dict(python=platform.python_version(), numpy=np.__version__,
                               scipy=scipy.__version__, matplotlib=matplotlib.__version__)
    (folder / name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(data, indent=2))
