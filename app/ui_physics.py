

import os
import math
import numpy as np
import plotly.graph_objects as go
from scipy.signal import sawtooth
import streamlit as st
import sys
import os
sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "magnet", "simplecs"))
if sim_path not in sys.path:
    sys.path.insert(0, sim_path)

from magnet.core import BH_Transformer
from plecs_io import get_plecs_server, load_model, close_model, simulate_bh_cycle

MATERIAL_PROPERTIES = {
    "3C90": {"Ms": 3.74e5},
    "3C94": {"Ms": 3.82e5},
    "3E6":  {"Ms": 3.50e5},
    "3F4":  {"Ms": 3.90e5},
    "N27":  {"Ms": 3.90e5},
    "N30":  {"Ms": 3.66e5},
    "N49":  {"Ms": 4.14e5},
    "N87":  {"Ms": 3.88e5},
}

MATERIAL_LIST = list(MATERIAL_PROPERTIES.keys())


def get_material_Ms(material):
    if material in MATERIAL_PROPERTIES:
        return MATERIAL_PROPERTIES[material]["Ms"]
    return MATERIAL_PROPERTIES["N87"]["Ms"]



def resample_to_N(x, N):
    x = np.asarray(x, dtype=float)
    if len(x) == N:
        return x
    u_old = np.linspace(0, 1, len(x), endpoint=False)
    u_new = np.linspace(0, 1, N, endpoint=False)
    return np.interp(u_new, u_old, x)


def compute_Hc(B, H):
    B = np.asarray(B, dtype=float)
    H = np.asarray(H, dtype=float)
    dB = np.gradient(B)
    Hc_pos, Hc_neg = np.nan, np.nan

    rising = dB > 0
    if np.any(rising):
        idx = np.where(rising)[0]
        for k in range(len(idx) - 1):
            i1, i2 = idx[k], idx[k + 1]
            if B[i1] <= 0 < B[i2]:
                Hc_pos = H[i1] + (H[i2] - H[i1]) * (-B[i1]) / (B[i2] - B[i1])
                break

    falling = dB < 0
    if np.any(falling):
        idx = np.where(falling)[0]
        for k in range(len(idx) - 1):
            i1, i2 = idx[k], idx[k + 1]
            if B[i1] >= 0 > B[i2]:
                Hc_neg = H[i1] + (H[i2] - H[i1]) * (-B[i1]) / (B[i2] - B[i1])
                break

    if np.isfinite(Hc_pos) and np.isfinite(Hc_neg):
        return 0.5 * (abs(Hc_pos) + abs(Hc_neg))
    elif np.isfinite(Hc_pos):
        return abs(Hc_pos)
    elif np.isfinite(Hc_neg):
        return abs(Hc_neg)
    return np.nan


def compute_anisotropy(Hc, theta_deg):
    theta_rad = math.radians(float(theta_deg))
    Hani_z = float(Hc)
    cos_t = math.cos(theta_rad)
    Hk = Hani_z / cos_t if abs(cos_t) >= 1e-6 else Hani_z * 1000
    Hani_y = Hk * math.sin(theta_rad)
    return Hani_z, Hani_y


def generate_B_waveform(freq_hz, amp, n_points, waveform="sine"):
    t = np.linspace(0, 1.0 / freq_hz, n_points, endpoint=False, dtype=np.float32)
    if waveform == "triangle":
        B = (amp * sawtooth(2 * np.pi * freq_hz * t, width=0.5)).astype(np.float32)
    elif waveform == "trapezoidal":
        B = (2 * amp * sawtooth(2 * np.pi * freq_hz * t, width=0.5)).astype(np.float32)
        B = np.clip(B, -amp, amp).astype(np.float32)
    else:
        B = (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    B = B - np.mean(B)
    return t, B


PLECS_MODEL = "PU_LLG_V3_v4,5"


def run_plecs_sim(server, H0, freq_hz, Hani_z, Hani_y, alpha,
                  material="N87", n_cycles=15, cycle_pick=8,
                  waveform="sine"):
    Ms = get_material_Ms(material)
    SW = {"sine": 1, "triangle": 2, "trapezoidal": 3}.get(waveform, 1)
    model_vars = {
        "Ms":     float(Ms),
        "alpha":  float(alpha),
        "gamma":  float(1.759e11),
        "mu0":    float(4 * math.pi * 1e-7),
        "Hani_z": float(Hani_z),
        "Hani_y": float(Hani_y),
        "f_ani":  float(freq_hz),
        "H0":     float(H0),
        "I0":     float(H0),
        "SW":     float(SW),
    }
    stop_time = float(n_cycles / freq_hz)
    t, H, B = simulate_bh_cycle(server, PLECS_MODEL, model_vars, stop_time, cycle_index=cycle_pick)
    B = B - np.mean(B)
    return H, B


def split_branches(B, H):
    dB = np.diff(B, prepend=B[0])
    asc = dB >= 0
    return (B[asc], H[asc]), (B[~asc], H[~asc])


def rrmse_on_common_B(B_ref, H_ref, B_cmp, H_cmp, nb=200):
    Bmin = max(np.min(B_ref), np.min(B_cmp))
    Bmax = min(np.max(B_ref), np.max(B_cmp))
    if not (np.isfinite(Bmin) and np.isfinite(Bmax)) or Bmax <= Bmin:
        return np.nan

    B_grid = np.linspace(Bmin, Bmax, nb)

    def interp_H(B, H):
        order = np.argsort(B)
        Bm, Hm = B[order], H[order]
        keep = np.r_[True, np.diff(Bm) != 0]
        Bm, Hm = Bm[keep], Hm[keep]
        return np.interp(B_grid, Bm, Hm, left=np.nan, right=np.nan)

    Hr = interp_H(B_ref, H_ref)
    Hc = interp_H(B_cmp, H_cmp)
    mask = np.isfinite(Hr) & np.isfinite(Hc)
    if mask.sum() < nb * 0.5:
        return np.nan

    err = Hr[mask] - Hc[mask]
    denom = max(np.linalg.norm(Hr[mask]), 1e-12)
    return np.linalg.norm(err) / denom


def loop_rrmse(B_ref, H_ref, B_cmp, H_cmp):
    (B_ra, H_ra), (B_rd, H_rd) = split_branches(B_ref, H_ref)
    (B_ca, H_ca), (B_cd, H_cd) = split_branches(B_cmp, H_cmp)
    e_asc = rrmse_on_common_B(B_ra, H_ra, B_ca, H_ca)
    e_des = rrmse_on_common_B(B_rd, H_rd, B_cd, H_cd)
    if np.isnan(e_asc) and np.isnan(e_des):
        return np.nan
    if np.isnan(e_asc):
        return e_des
    if np.isnan(e_des):
        return e_asc
    return 0.5 * (e_asc + e_des)


def ui_physics(m):
    st.title('MagNet Physics — LLG Parameter Fitting')
    st.markdown("""
    Fitting Landau-Lifshitz-Gilbert (LLG) parameters by sweeping **θ** (easy-axis angle) 
    and **α** (damping) to match a PLECS LLG simulation against the ML Transformer reference B-H loop.
    """)

    N_MODEL = 128 

    st.header("Material & Excitation")
    col1, col2, col3 = st.columns(3)
    with col1:
        material = st.selectbox("Material", MATERIAL_LIST, index=MATERIAL_LIST.index("N49"),
                                key=f"physics_material_{m}")
    with col2:
        freq_khz = st.number_input("Frequency (kHz)", min_value=1.0, max_value=1000.0,
                                   value=70.0, step=10.0, key=f"physics_freq_{m}")
        freq_hz = freq_khz * 1e3
    with col3:
        amp_t = st.number_input("Amplitude (T)", min_value=0.01, max_value=1.0,
                                value=0.25, step=0.05, key=f"physics_amp_{m}")

    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.number_input("Temperature (°C)", min_value=0, max_value=120,
                               value=25, step=5, key=f"physics_temp_{m}")
    with col2:
        waveform = st.selectbox("Waveform", ["sine", "triangle", "trapezoidal"],
                                key=f"physics_wave_{m}")

   
    st.header("Sweep Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("θ (Easy-Axis Angle)")
        theta_min = st.number_input("θ min (°)", value=77.0, step=1.0, key=f"physics_tmin_{m}")
        theta_max = st.number_input("θ max (°)", value=88.0, step=1.0, key=f"physics_tmax_{m}")
        theta_pts = st.number_input("θ points", min_value=3, max_value=100,
                                    value=25, step=5, key=f"physics_tpts_{m}")
    with col2:
        st.subheader("α (LLG Damping)")
        alpha_min = st.number_input("α min", value=15.0, step=1.0, key=f"physics_amin_{m}")
        alpha_max = st.number_input("α max", value=25.0, step=1.0, key=f"physics_amax_{m}")
        alpha_pts = st.number_input("α points", min_value=3, max_value=100,
                                    value=25, step=5, key=f"physics_apts_{m}")

    n_cycles = 15
    cycle_pick = 8

    # ── Generalization
    with st.expander("Generalization Sweep Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gen_freq_span = st.number_input("Freq span ± (kHz)", min_value=1.0, max_value=200.0,
                                            value=30.0, step=5.0, key=f"physics_gfspan_{m}")
        with col2:
            gen_amp_min = st.number_input("Amp min (T)", min_value=0.01, max_value=1.0,
                                          value=0.1, step=0.05, key=f"physics_gamin_{m}")
            gen_amp_max = st.number_input("Amp max (T)", min_value=0.01, max_value=1.0,
                                          value=0.3, step=0.05, key=f"physics_gamax_{m}")
        with col3:
            gen_pts = st.number_input("Grid points per axis", min_value=3, max_value=50,
                                      value=20, step=5, key=f"physics_gpts_{m}")


    run_fitting = st.button("🚀 Run LLG Fitting", key=f"physics_run_{m}")

    if run_fitting:
        _run_fitting_workflow(
            material=material,
            freq_hz=freq_hz,
            amp_t=amp_t,
            temp=temp,
            waveform=waveform,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_pts=int(theta_pts),
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_pts=int(alpha_pts),
            n_cycles=int(n_cycles),
            cycle_pick=int(cycle_pick),
            gen_freq_span_hz=gen_freq_span * 1e3,
            gen_amp_min=gen_amp_min,
            gen_amp_max=gen_amp_max,
            gen_pts=int(gen_pts),
            n_model=N_MODEL,
        )


def _run_fitting_workflow(*, material, freq_hz, amp_t, temp, waveform,
                          theta_min, theta_max, theta_pts,
                          alpha_min, alpha_max, alpha_pts,
                          n_cycles, cycle_pick,
                          gen_freq_span_hz, gen_amp_min, gen_amp_max, gen_pts,
                          n_model):

    st.markdown("---")
    st.subheader(f"Results: {material}  |  {freq_hz/1e3:.0f} kHz  |  {amp_t:.2f} T  |  {waveform}")

    with st.spinner("Phase 1/3 — Computing ML reference B-H loop..."):
        t_ml, B_ml = generate_B_waveform(freq_hz, amp_t, n_model, waveform)
        shift_idx = np.argmax(B_ml)
        B_ml_rolled = np.roll(B_ml, -shift_idx)
        H_ml_rolled = np.asarray(
            BH_Transformer(material, freq_hz, temp, 0.0, B_ml_rolled), dtype=float
        )
        H_ml = np.roll(H_ml_rolled, shift_idx)

        Hc_ml = compute_Hc(B_ml, H_ml)
        if not np.isfinite(Hc_ml):
            st.error("ML loop did not yield a valid coercive field (Hc). Cannot proceed.")
            return

        H_ml_amp = 0.5 * (np.max(H_ml) - np.min(H_ml))

    st.write("**Phase 2/3 — θ/α Sweep** (fitting LLG parameters via PLECS)")
    theta_grid = np.linspace(theta_min, theta_max, theta_pts)
    alpha_grid = np.linspace(alpha_min, alpha_max, alpha_pts)

    Err = np.full((len(theta_grid), len(alpha_grid)), np.nan)
    total = len(theta_grid) * len(alpha_grid)
    progress_bar = st.progress(0)
    status_text = st.empty()
    count = 0

    # Open PLECS model ONCE for the entire sweep
    server = get_plecs_server()
    load_model(server, PLECS_MODEL)
    try:
        for j, theta_deg in enumerate(theta_grid):
            Hani_z, Hani_y = compute_anisotropy(Hc_ml, theta_deg)
            for i, alpha in enumerate(alpha_grid):
                count += 1
                try:
                    H_plecs, B_plecs = run_plecs_sim(
                        server, H_ml_amp, freq_hz, Hani_z, Hani_y, alpha,
                        material, n_cycles=n_cycles, cycle_pick=cycle_pick,
                        waveform=waveform
                    )
                    H_plecs = resample_to_N(H_plecs, n_model)
                    B_plecs = resample_to_N(B_plecs, n_model)
                    Err[j, i] = loop_rrmse(B_ml, H_ml, B_plecs, H_plecs)
                    status_text.text(
                        f"[{count}/{total}] θ={theta_deg:.2f}°, α={alpha:.3f}  →  RRMSE={Err[j,i]:.5f}"
                    )
                except Exception as e:
                    status_text.text(
                        f"[{count}/{total}] θ={theta_deg:.2f}°, α={alpha:.3f}  [skip] {e}"
                    )
                progress_bar.progress(count / total)
    finally:
        close_model(server, PLECS_MODEL)

    if not np.isfinite(Err).any():
        st.error("No valid simulation results. Check that PLECS is running.")
        return

    # Find best/worst
    j_best, i_best = np.unravel_index(np.nanargmin(Err), Err.shape)
    j_worst, i_worst = np.unravel_index(np.nanargmax(Err), Err.shape)

    best_theta, best_alpha = theta_grid[j_best], alpha_grid[i_best]
    worst_theta, worst_alpha = theta_grid[j_worst], alpha_grid[i_worst]
    best_err = Err[j_best, i_best]
    worst_err = Err[j_worst, i_worst]

    best_Hani_z, best_Hani_y = compute_anisotropy(Hc_ml, best_theta)
    worst_Hani_z, worst_Hani_y = compute_anisotropy(Hc_ml, worst_theta)

    st.success("θ/α sweep complete!")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Best θ (°)", f"{best_theta:.3f}")
        st.metric("Best α", f"{best_alpha:.4f}")
        st.metric("Best RRMSE", f"{best_err:.6f}")
    with col2:
        st.metric("Hani_z (A/m)", f"{best_Hani_z:.4f}")
        st.metric("Hani_y (A/m)", f"{best_Hani_y:.4f}")

    with st.spinner("Re-running best & worst simulations for B-H plots..."):
        load_model(server, PLECS_MODEL)
        try:
            H_best, B_best = run_plecs_sim(
                server, H_ml_amp, freq_hz, best_Hani_z, best_Hani_y,
                best_alpha, material, n_cycles=n_cycles, cycle_pick=cycle_pick,
                waveform=waveform
            )
            H_worst, B_worst = run_plecs_sim(
                server, H_ml_amp, freq_hz, worst_Hani_z, worst_Hani_y,
                worst_alpha, material, n_cycles=n_cycles, cycle_pick=cycle_pick,
                waveform=waveform
            )
        finally:
            close_model(server, PLECS_MODEL)
        H_best = resample_to_N(H_best, n_model)
        B_best = resample_to_N(B_best, n_model)
        H_worst = resample_to_N(H_worst, n_model)
        B_worst = resample_to_N(B_worst, n_model)


    B_ml_plot = np.append(B_ml, B_ml[0])
    H_ml_plot = np.append(H_ml, H_ml[0])
    B_worst_plot = np.append(B_worst, B_worst[0])
    H_worst_plot = np.append(H_worst, H_worst[0])
    B_best_plot = np.append(B_best, B_best[0])
    H_best_plot = np.append(H_best, H_best[0])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("RRMSE Heatmap (θ vs α)")
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=Err,
            x=alpha_grid,
            y=theta_grid,
            colorscale="Viridis",
            colorbar=dict(title="RRMSE"),
        ))
        fig_heatmap.add_trace(go.Scatter(
            x=[best_alpha], y=[best_theta], mode="markers",
            marker=dict(symbol="x", size=16, color="red", line=dict(width=2)),
            name="Best",
        ))
        fig_heatmap.add_trace(go.Scatter(
            x=[worst_alpha], y=[worst_theta], mode="markers",
            marker=dict(symbol="circle-open", size=14, color="red", line=dict(width=2)),
            name="Worst",
        ))
        fig_heatmap.update_layout(
            title_text=f"<b>{material}  |  {freq_hz/1e3:.0f} kHz  |  {amp_t:.2f} T</b>",
            title_x=0.5,
            xaxis_title="Damping α",
            yaxis_title="Theta θ (deg)",
            legend=dict(yanchor="bottom", y=0, xanchor="right", x=1),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        st.subheader(f"Worst: θ={worst_theta:.1f}°, α={worst_alpha:.2f}")
        fig_worst = go.Figure()
        fig_worst.add_trace(go.Scatter(
            x=H_ml_plot, y=B_ml_plot * 1e3,
            line=dict(color='mediumslateblue', width=4),
            name="ML (Ref)",
        ))
        fig_worst.add_trace(go.Scatter(
            x=H_worst_plot, y=B_worst_plot * 1e3,
            line=dict(color='firebrick', width=4),
            name="PLECS (LLG)",
        ))
        fig_worst.update_layout(
            title_text=f"<b>RRMSE = {worst_err:.4f}</b>",
            title_x=0.5,
            legend=dict(yanchor="bottom", y=0, xanchor="right", x=0.935),
        )
        fig_worst.update_yaxes(title_text="B - Flux Density [mT]",
                               zeroline=True, zerolinewidth=1.5, zerolinecolor='gray')
        fig_worst.update_xaxes(title_text="H - Field Strength [A/m]",
                               zeroline=True, zerolinewidth=1.5, zerolinecolor='gray')
        st.plotly_chart(fig_worst, use_container_width=True)

    with col3:
        st.subheader(f"Best: θ={best_theta:.1f}°, α={best_alpha:.2f}")
        fig_best = go.Figure()
        fig_best.add_trace(go.Scatter(
            x=H_ml_plot, y=B_ml_plot * 1e3,
            line=dict(color='mediumslateblue', width=4),
            name="ML (Ref)",
        ))
        fig_best.add_trace(go.Scatter(
            x=H_best_plot, y=B_best_plot * 1e3,
            line=dict(color='firebrick', width=4),
            name="PLECS (LLG)",
        ))
        fig_best.update_layout(
            title_text=f"<b>RRMSE = {best_err:.4f}</b>",
            title_x=0.5,
            legend=dict(yanchor="bottom", y=0, xanchor="right", x=0.935),
        )
        fig_best.update_yaxes(title_text="B - Flux Density [mT]",
                              zeroline=True, zerolinewidth=1.5, zerolinecolor='gray')
        fig_best.update_xaxes(title_text="H - Field Strength [A/m]",
                              zeroline=True, zerolinewidth=1.5, zerolinecolor='gray')
        st.plotly_chart(fig_best, use_container_width=True)

    st.write("**Phase 3/3 — Generalization Sweep** (testing best-fit across freq/amp grid)")
    freq_min = max(10e3, freq_hz - gen_freq_span_hz)
    freq_max = freq_hz + gen_freq_span_hz
    amp_min_gen = gen_amp_min
    amp_max_gen = gen_amp_max

    freq_gen = np.linspace(freq_min, freq_max, gen_pts)
    amp_gen = np.linspace(amp_min_gen, amp_max_gen, gen_pts)

    Err_gen = np.full((len(amp_gen), len(freq_gen)), np.nan)
    total_gen = len(amp_gen) * len(freq_gen)
    progress_bar2 = st.progress(0)
    status_text2 = st.empty()
    count_gen = 0

    load_model(server, PLECS_MODEL)
    try:
        for ai, amp in enumerate(amp_gen):
            for fi, ftest in enumerate(freq_gen):
                count_gen += 1
                try:
                    _, B_test = generate_B_waveform(ftest, amp, n_model, waveform)

                    shift_idx_test = np.argmax(B_test)
                    B_test_rolled = np.roll(B_test, -shift_idx_test)
                    H_test_rolled = np.asarray(
                        BH_Transformer(material, ftest, temp, 0.0, B_test_rolled), dtype=float
                    )
                    H_test = np.roll(H_test_rolled, shift_idx_test)

                    H_amp_test = 0.5 * (np.max(H_test) - np.min(H_test))
                    Hc_test = compute_Hc(B_test, H_test)
                    if not np.isfinite(Hc_test):
                        Hc_test = Hc_ml

                    Hani_z_t, Hani_y_t = compute_anisotropy(Hc_test, best_theta)
                    H_t, B_t = run_plecs_sim(
                        server, H_amp_test, ftest, Hani_z_t, Hani_y_t,
                        best_alpha, material, n_cycles=n_cycles, cycle_pick=cycle_pick,
                        waveform=waveform
                    )
                    H_t = resample_to_N(H_t, n_model)
                    B_t = resample_to_N(B_t, n_model)

                    Err_gen[ai, fi] = loop_rrmse(B_test, H_test, B_t, H_t)
                    status_text2.text(
                        f"[{count_gen}/{total_gen}] f={ftest/1e3:.1f} kHz, B={amp:.3f} T  →  RRMSE={Err_gen[ai,fi]:.5f}"
                    )
                except Exception as e:
                    status_text2.text(
                        f"[{count_gen}/{total_gen}] f={ftest/1e3:.1f} kHz, B={amp:.3f} T  [skip] {e}"
                    )
                progress_bar2.progress(count_gen / total_gen)
    finally:
        close_model(server, PLECS_MODEL)

    cg1, cg2, cg3 = st.columns(3)
    with cg1:
        st.subheader("Generalization Sweep")
        fig_gen = go.Figure(data=go.Heatmap(
            z=Err_gen,
            x=np.round(freq_gen / 1e3, 1),
            y=np.round(amp_gen, 3),
            colorscale="Plasma",
            colorbar=dict(title="RRMSE"),
        ))
        fig_gen.add_trace(go.Scatter(
            x=[freq_hz / 1e3], y=[amp_t], mode="markers",
            marker=dict(symbol="star", size=20, color="yellow",
                        line=dict(width=2, color="black")),
            name="Training Point",
        ))
        fig_gen.update_layout(
            title_text=(f"<b>Generalization  |  θ={best_theta:.1f}°, α={best_alpha:.2f}</b>"
                        f"<br>Fitted point: {freq_hz/1e3:.0f} kHz, {amp_t:.2f} T"),
            title_x=0.5,
            xaxis_title="Frequency (kHz)",
            yaxis_title="B Amplitude (T)",
            legend=dict(yanchor="top", y=1, xanchor="right", x=1),
        )
        st.plotly_chart(fig_gen, use_container_width=True)

    st.success("✅ LLG fitting complete!")
    st.markdown("---")

