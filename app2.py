"""Wi-Fi Signal Coverage Mapper - starter version (PS-15)

Run:  pip install streamlit numpy matplotlib pandas
      streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

DEAD_ZONE_DBM = -80   # below this = dead zone
PATH_LOSS_N = 3.0     # indoor path-loss exponent (2 = open space, 3-4 = walls)

st.title("Real-Time Wi-Fi Signal Coverage Mapper")

# ---------- 1. Inputs ----------
st.sidebar.header("Room & router")
W = st.sidebar.number_input("Room width (m)", 5.0, 100.0, 12.0)
H = st.sidebar.number_input("Room height (m)", 5.0, 100.0, 8.0)
rx = st.sidebar.number_input("Current router X", 0.0, W, 1.0)
ry = st.sidebar.number_input("Current router Y", 0.0, H, 1.0)
floor = st.sidebar.file_uploader("Floor plan image (optional)", type=["png", "jpg", "jpeg"])
alpha = st.sidebar.slider("Heatmap opacity", 0.2, 1.0, 0.6) if floor else 1.0

st.subheader("Measurements (edit, add or delete rows)")
default = pd.DataFrame({
    "x": [1, 3, 6, 9, 11, 2, 6, 10, 3, 11],
    "y": [1, 2, 1, 2, 1, 5, 5, 6, 7, 7],
    "dbm": [-35, -45, -58, -72, -80, -55, -65, -82, -70, -88],
})
data = st.data_editor(default, num_rows="dynamic")
data = data.dropna()


# ---------- 2. Estimation (IDW interpolation) ----------
def idw(points, values, gx, gy, power=2):
    """Estimate signal at every grid cell from nearby measurements."""
    d = np.sqrt((gx[..., None] - points[:, 0]) ** 2 +
                (gy[..., None] - points[:, 1]) ** 2)
    d = np.maximum(d, 1e-6)
    w = 1 / d ** power
    return (w * values).sum(axis=-1) / w.sum(axis=-1)


if len(data) >= 3:
    xs = np.linspace(0, W, 120)
    ys = np.linspace(0, H, 80)
    gx, gy = np.meshgrid(xs, ys)
    pts = data[["x", "y"]].to_numpy(float)
    vals = data["dbm"].to_numpy(float)
    signal = idw(pts, vals, gx, gy)

    # ---------- 3. Router position suggestion ----------
    def dist(cx, cy):
        return np.maximum(np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2), 0.5)

    def predict(cx, cy):
        # shift signal by the change in distance to the router
        return signal + 10 * PATH_LOSS_N * np.log10(dist(rx, ry) / dist(cx, cy))

    def dead_pct(sig):
        return 100 * (sig < DEAD_ZONE_DBM).mean()

    best, best_dead, best_avg = (rx, ry), dead_pct(signal), signal.mean()
    for cx in np.linspace(0.5, W - 0.5, 20):
        for cy in np.linspace(0.5, H - 0.5, 14):
            s = predict(cx, cy)
            score = (dead_pct(s), -s.mean())
            if score < (best_dead, -best_avg):
                best, best_dead, best_avg = (cx, cy), score[0], s.mean()

    # ---------- 4. Heatmap ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    if floor:
        from PIL import Image
        ax.imshow(np.array(Image.open(floor).convert("RGB")),
                  extent=[0, W, 0, H], aspect="equal")
    im = ax.imshow(signal, extent=[0, W, 0, H], origin="lower",
                   cmap="RdYlGn", vmin=-90, vmax=-30, aspect="equal", alpha=alpha)
    ax.contourf(gx, gy, signal, levels=[-200, DEAD_ZONE_DBM],
                colors="none", hatches=["///"])          # dead zones hatched
    ax.contour(gx, gy, signal, levels=[DEAD_ZONE_DBM], colors="black")
    ax.scatter(pts[:, 0], pts[:, 1], c="white", edgecolors="black", label="Measured")
    ax.scatter(rx, ry, marker="*", s=300, c="blue", label="Router now")
    ax.scatter(*best, marker="*", s=300, c="magenta", label="Suggested router")
    fig.colorbar(im, label="Signal (dBm)")
    ax.legend(loc="upper left")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    st.pyplot(fig)

    # ---------- 5. Results ----------
    now_dead = dead_pct(signal)
    st.metric("Dead-zone area now", f"{now_dead:.1f}%")
    st.success(f"Move router to ({best[0]:.1f}, {best[1]:.1f}) -> "
               f"dead zone drops {now_dead:.1f}% -> {best_dead:.1f}%, "
               f"average signal {signal.mean():.1f} -> {best_avg:.1f} dBm")
    st.caption("Hatched area / black line = dead zones. Walls are ignored in this simple model.")
else:
    st.info("Add at least 3 measurements.")