"""
Polarization optical-train explorer
===================================

Define a starting polarization (H, V, A, D, R, L), stack linear polarizers and
waveplates, then see:
  1. a schematic of the optical train,
  2. a PyVista 3D render of the final polarization as a real-space E-field helix,
  3. a Poincaré (Bloch) sphere showing the final state.

-----------------------------------------------------------------------
RUNNING LOCALLY
-----------------------------------------------------------------------
    pip install streamlit pyvista stpyvista numpy matplotlib
    streamlit run polarization_app.py

-----------------------------------------------------------------------
RUNNING ON STREAMLIT CLOUD  (headless, no GPU, no X server)
-----------------------------------------------------------------------
The default `vtk` wheel from PyPI renders through X11/EGL, which does not
exist on Streamlit Cloud and segfaults. The fix is the OSMesa-built VTK wheel
(`vtk-osmesa`), which renders fully offscreen on CPU. Put these next to the
script in your repo:

  requirements.txt
      streamlit
      numpy
      matplotlib
      pyvista
      stpyvista
      vtk-osmesa          <-- replaces the default vtk wheel (do NOT also list vtk)

  packages.txt          <-- system (apt) packages
      libgl1
      libglx-mesa0
      libosmesa6          <-- the OSMesa runtime the wheel links against

No xvfb, no DISPLAY, no X server. The env vars at the top of this file select
the OSMesa render window before VTK imports.
-----------------------------------------------------------------------
"""

# --- headless offscreen rendering config: must run before pyvista / VTK import
# Uses the vtk-osmesa wheel (see requirements.txt) so VTK renders through OSMesa
# with no X server / display needed. Do NOT use xvfb here — on Streamlit Cloud
# the X path is unavailable and segfaults; OSMesa is the supported route.
import os
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow

# stpyvista is optional; fall back to screenshots if unavailable
try:
    from stpyvista import stpyvista
    HAVE_STPYVISTA = True
except Exception:
    HAVE_STPYVISTA = False


# ----------------------------------------------------------------------
# Jones calculus
# ----------------------------------------------------------------------

JONES_STATES = {
    "H  — horizontal":       [1, 0],
    "V  — vertical":         [0, 1],
    "D  — +45° diagonal":    [1, 1],
    "A  — -45° anti-diag.":  [1, -1],
    "R  — right circular":   [1, -1j],
    "L  — left circular":    [1, 1j],
}


def make_state(key):
    j = np.array(JONES_STATES[key], dtype=complex)
    return j / np.linalg.norm(j)


def polarizer(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c * c, c * s], [c * s, s * s]], dtype=complex)


def waveplate(retardance, theta):
    c, s = np.cos(theta), np.sin(theta)
    Rot = np.array([[c, -s], [s, c]])
    J = np.diag([np.exp(-1j * retardance / 2), np.exp(1j * retardance / 2)])
    return Rot @ J @ Rot.T


def stokes(j):
    Ex, Ey = j[0], j[1]
    S0 = (abs(Ex) ** 2 + abs(Ey) ** 2).real
    S1 = (abs(Ex) ** 2 - abs(Ey) ** 2).real
    S2 = (2 * np.real(np.conj(Ex) * Ey))
    S3 = (2 * np.imag(np.conj(Ex) * Ey))
    return float(S0), float(S1), float(S2), float(S3)


def classify(j, tol=1e-3):
    S0, S1, S2, S3 = stokes(j)
    if S0 < tol:
        return "extinguished (no light)"
    s1, s2, s3 = S1 / S0, S2 / S0, S3 / S0
    if abs(s3) > 1 - tol:
        return "right circular" if s3 < 0 else "left circular"
    if abs(s3) < tol:
        ang = 0.5 * np.degrees(np.arctan2(s2, s1))
        return f"linear at {ang:+.1f}°"
    hand = "right" if s3 < 0 else "left"
    ang = 0.5 * np.degrees(np.arctan2(s2, s1))
    return f"{hand}-elliptical (major axis {ang:+.1f}°)"


def propagate(elements, j0):
    j = j0.copy()
    intensities = [float(np.vdot(j, j).real)]
    for M, _ in elements:
        j = M @ j
        intensities.append(float(np.vdot(j, j).real))
    return j, intensities


# ----------------------------------------------------------------------
# Optical-train schematic (matplotlib)
# ----------------------------------------------------------------------

def draw_train(start_label, elements, intensities):
    n = len(elements)
    fig_w = max(6, 2.2 + 1.9 * (n + 1))
    fig, ax = plt.subplots(figsize=(fig_w, 2.6))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, 2.6)
    ax.axis("off")

    y = 1.3
    x = 0.4
    box_w, box_h, gap = 1.5, 1.0, 0.45

    def beam(x0, x1, I):
        ax.add_patch(FancyArrow(x0, y, x1 - x0 - 0.05, 0,
                                width=0.015 + 0.05 * I, head_width=0.18,
                                head_length=0.12, length_includes_head=True,
                                color="#C0392B", alpha=0.55 + 0.45 * I))

    ax.text(x, y + 0.78, "input", ha="center", fontsize=10, color="#555")
    ax.add_patch(FancyBboxPatch((x - 0.35, y - 0.45), 0.7, 0.9,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#FDEBD0", ec="#B9770E", lw=1.5))
    ax.text(x, y, start_label.split()[0], ha="center", va="center",
            fontsize=13, fontweight="bold", color="#7E5109")
    cursor = x + 0.35

    colors = {"polarizer": ("#D6EAF8", "#21618C", "#1B4F72"),
              "waveplate": ("#D5F5E3", "#196F3D", "#145A32")}

    for i, (M, label) in enumerate(elements):
        bx = cursor + gap
        beam(cursor, bx, intensities[i])
        kind = "polarizer" if label.lower().startswith("pol") else "waveplate"
        fc, ec, tc = colors[kind]
        ax.add_patch(FancyBboxPatch((bx, y - box_h / 2), box_w, box_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1.5))
        ax.text(bx + box_w / 2, y, label, ha="center", va="center",
                fontsize=9.5, color=tc, fontweight="bold")
        cursor = bx + box_w

    beam(cursor, cursor + gap + 0.5, intensities[-1])
    ax.text(cursor + gap + 0.55, y + 0.4,
            f"output\nI = {intensities[-1]*100:.0f}%",
            ha="left", va="center", fontsize=10, color="#555")

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# PyVista plotters
# ----------------------------------------------------------------------

def _new_plotter(size):
    import pyvista as pv
    pv.OFF_SCREEN = True
    # OSMesa (vtk-osmesa wheel) renders with no display; no xvfb needed.
    return pv.Plotter(window_size=list(size), off_screen=True)


def field_plotter(j, n_periods=3, k=2 * np.pi):
    import pyvista as pv
    norm = np.linalg.norm(j)
    if norm < 1e-9:
        return None
    Ex, Ey = j / norm

    z = np.linspace(0, n_periods, 800)
    ex = np.real(Ex * np.exp(1j * k * z))
    ey = np.real(Ey * np.exp(1j * k * z))

    pl = _new_plotter((760, 520))
    pl.set_background("white")

    tube = pv.Spline(np.column_stack([z, ex, ey]), len(z)).tube(radius=0.04)
    pl.add_mesh(tube, color="#2E6FB7", smooth_shading=True,
                specular=0.5, specular_power=15)

    pl.add_mesh(pv.Line((0, 0, 0), (n_periods, 0, 0)).tube(radius=0.007),
                color="#999999")

    for i in range(0, len(z), 28):
        v = pv.Line((z[i], 0, 0), (z[i], ex[i], ey[i])).tube(radius=0.009)
        pl.add_mesh(v, color="#2E6FB7", opacity=0.35)

    wt = np.linspace(0, 2 * np.pi, 200)
    pex = np.real(Ex * np.exp(1j * wt))
    pey = np.real(Ey * np.exp(1j * wt))
    ell = np.column_stack([np.full_like(pex, n_periods), pex, pey])
    pl.add_mesh(pv.Spline(ell, len(wt)).tube(radius=0.016), color="#C0392B")

    pl.camera_position = [(n_periods * 1.6, 2.6, 2.2),
                          (n_periods / 2, 0, 0), (0, 0, 1)]
    pl.add_axes(xlabel="z/lambda", ylabel="Ex", zlabel="Ey",
                color="black", line_width=3)
    return pl


def poincare_plotter(j):
    """Poincaré (Bloch) sphere. Bloch vector = normalized (S1, S2, S3)."""
    import pyvista as pv
    norm = np.linalg.norm(j)
    if norm < 1e-9:
        return None
    S0, S1, S2, S3 = stokes(j)
    if S0 < 1e-9:
        return None
    bloch = np.array([S1, S2, S3]) / S0  # pure state -> unit length

    pl = _new_plotter((560, 560))
    pl.set_background("white")

    pl.add_mesh(pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60),
                color="#DCE6F2", opacity=0.18, smooth_shading=True)

    # equator + two meridians as guide rings
    for normal in [(0, 0, 1), (1, 0, 0), (0, 1, 0)]:
        ring = pv.Circle(radius=1.0, resolution=120)
        if normal == (1, 0, 0):
            ring = ring.rotate_y(90, inplace=False)
        elif normal == (0, 1, 0):
            ring = ring.rotate_x(90, inplace=False)
        pl.add_mesh(ring.extract_feature_edges(), color="#9FB3C8", line_width=1)

    for a in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        a = np.array(a)
        pl.add_mesh(pv.Line(-1.15 * a, 1.15 * a), color="#9FB3C8", line_width=1)

    poles = {
        "H": (1, 0, 0), "V": (-1, 0, 0),
        "D": (0, 1, 0), "A": (0, -1, 0),
        "L": (0, 0, 1), "R": (0, 0, -1),
    }
    pts = np.array(list(poles.values())) * 1.25
    pl.add_point_labels(pts, list(poles.keys()),
                        font_size=18, text_color="#34495E",
                        shape=None, always_visible=True, show_points=False)

    pl.add_mesh(pv.Line((0, 0, 0), tuple(bloch)), color="#C0392B", line_width=5)
    pl.add_mesh(pv.Sphere(radius=0.05, center=tuple(bloch)), color="#C0392B")

    pl.camera_position = [(2.4, 1.8, 1.6), (0, 0, 0), (0, 0, 1)]
    pl.add_axes(xlabel="S1", ylabel="S2", zlabel="S3",
                color="black", line_width=3)
    return pl


def show_plotter(pl, key, caption=None, interactive=False):
    """Render a plotter.

    Default is a server-side screenshot (a PNG), which is the robust path on
    Streamlit Cloud. The stpyvista *interactive* (trame) backend serializes the
    scene to HTML and can fail with a pickling error on some plotter contents,
    so it is opt-in via `interactive=True`.
    """
    if pl is None:
        st.info("Nothing to render.")
        return

    if interactive and HAVE_STPYVISTA:
        try:
            stpyvista(pl, key=key)
            if caption:
                st.caption(caption)
            return
        except Exception:
            pass  # fall through to a static screenshot

    img = pl.screenshot(return_img=True)
    pl.close()
    st.image(img, use_container_width=True, caption=caption)


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------

st.set_page_config(page_title="Polarization optical train", layout="wide")
st.title("Polarization optical-train explorer")
st.caption("Jones calculus → schematic → field helix + Poincaré sphere")

if "elements" not in st.session_state:
    st.session_state.elements = []

with st.sidebar:
    st.header("Input beam")
    start_key = st.selectbox("Starting polarization",
                             list(JONES_STATES.keys()), index=0)

    st.divider()
    st.header("Add an element")
    add_kind = st.radio("Type", ["Linear polarizer", "Waveplate"])
    c1, c2 = st.columns(2)
    if c1.button("➕ Add", use_container_width=True):
        new = {"kind": add_kind, "angle": 0.0}
        if add_kind == "Waveplate":
            new["retard"] = np.pi / 2  # default λ/4
        else:
            new["retard"] = None
        st.session_state.elements.append(new)
        st.rerun()
    if c2.button("🗑 Clear all", use_container_width=True):
        st.session_state.elements = []
        st.rerun()

    if st.session_state.elements:
        st.divider()
        st.header("Edit train")
        st.caption("Light passes top → bottom. Changes apply live.")

        for i, e in enumerate(st.session_state.elements):
            is_pol = e["kind"].startswith("Linear")
            icon = "🟦 Polarizer" if is_pol else "🟩 Waveplate"
            with st.expander(f"{i+1}. {icon}", expanded=True):
                # editable axis angle (live)
                e["angle"] = np.radians(
                    st.slider("Axis angle (°)", -90, 90,
                              int(round(np.degrees(e["angle"]))), 1,
                              key=f"ang{i}"))

                # editable retardance for waveplates
                if not is_pol:
                    cur = np.degrees(e["retard"])
                    preset = ("Quarter-wave (λ/4)" if abs(cur - 90) < 1 else
                              "Half-wave (λ/2)" if abs(cur - 180) < 1 else "Custom")
                    choice = st.selectbox(
                        "Retardance",
                        ["Quarter-wave (λ/4)", "Half-wave (λ/2)", "Custom"],
                        index=["Quarter-wave (λ/4)", "Half-wave (λ/2)",
                               "Custom"].index(preset),
                        key=f"wp{i}")
                    if choice.startswith("Quarter"):
                        e["retard"] = np.pi / 2
                    elif choice.startswith("Half"):
                        e["retard"] = np.pi
                    else:
                        e["retard"] = np.radians(
                            st.slider("Retardance (°)", 0, 360,
                                      int(round(cur)), 5, key=f"ret{i}"))

                # reorder / remove controls
                b1, b2, b3 = st.columns(3)
                if b1.button("▲", key=f"up{i}", use_container_width=True,
                             disabled=(i == 0)):
                    st.session_state.elements[i-1], st.session_state.elements[i] = \
                        st.session_state.elements[i], st.session_state.elements[i-1]
                    st.rerun()
                if b2.button("▼", key=f"dn{i}", use_container_width=True,
                             disabled=(i == len(st.session_state.elements)-1)):
                    st.session_state.elements[i+1], st.session_state.elements[i] = \
                        st.session_state.elements[i], st.session_state.elements[i+1]
                    st.rerun()
                if b3.button("✕", key=f"rm{i}", use_container_width=True):
                    st.session_state.elements.pop(i)
                    st.rerun()

# Build element matrices + labels
elements = []
for e in st.session_state.elements:
    a = e["angle"]
    if e["kind"].startswith("Linear"):
        M = polarizer(a)
        label = f"Polarizer\n{np.degrees(a):.0f}°"
    else:
        M = waveplate(e["retard"], a)
        d = np.degrees(e["retard"])
        tag = "λ/4" if abs(d - 90) < 1 else ("λ/2" if abs(d - 180) < 1 else f"δ={d:.0f}°")
        label = f"Waveplate {tag}\n@ {np.degrees(a):.0f}°"
    elements.append((M, label))

j0 = make_state(start_key)
j_final, intensities = propagate(elements, j0)
I = intensities[-1]

# ---- Result 1: schematic ----
st.subheader("1 · Optical train")
st.pyplot(draw_train(start_key, elements, intensities), use_container_width=True)

# ---- summary row ----
m1, m2, m3 = st.columns(3)
m1.metric("Transmitted intensity", f"{I*100:.1f}%")
if I > 1e-6:
    jn = j_final / np.linalg.norm(j_final)
    m2.metric("State", classify(jn))
    S0, S1, S2, S3 = stokes(j_final)
    m3.metric("S₃ (handedness)", f"{S3/S0:+.3f}")
else:
    m2.metric("State", "extinguished")

# ---- Results 2 & 3: 3D views ----
st.subheader("2 · Final polarization — real-space field & Poincaré sphere")
colL, colR = st.columns(2)

if I > 1e-6:
    try:
        with colL:
            st.markdown("**E-field helix**")
            show_plotter(field_plotter(j_final), key="field",
                         caption="Blue tube: E along z. Red ring: temporal ellipse.")
        with colR:
            st.markdown("**Poincaré sphere**")
            show_plotter(poincare_plotter(j_final), key="poincare",
                         caption="Red vector: final state. Poles = H/V/D/A/R/L.")
    except ImportError:
        st.error("PyVista not installed. Run: pip install pyvista stpyvista")
    except Exception as ex:
        st.error(f"3D render failed (OpenGL/xvfb issue): {ex}")
        st.info("On Streamlit Cloud, use the OSMesa VTK build: put `vtk-osmesa` "
                "in requirements.txt and `libgl1, libglx-mesa0, libosmesa6` in "
                "packages.txt (see header).")
else:
    st.warning("Beam fully extinguished — no polarization to display.")