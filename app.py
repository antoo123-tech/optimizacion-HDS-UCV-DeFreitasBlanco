import streamlit as st
import matplotlib.pyplot as plt

from main import (
    calcular_k0,
    calcular_k,
    calcular_conversion,
    calcular_azufre_salida,
    calcular_t_efluente,
    calcular_t_intercambiador,
    calcular_q_horno,
    resolver_optimizacion_hds,
)

st.set_page_config(page_title="Optimización HDS", layout="wide")
st.title("Optimización Multiobjetivo de una Unidad HDS")
st.caption("Creado por Jose De Freitas y Crisbely Blanco")
st.write("Explora la frontera de Pareto y el *knee point* variando las condiciones del proceso.")

# --- Barra lateral (controles) ---
st.sidebar.header("Parámetros del proceso")

m_dot = st.sidebar.number_input("Flujo másico $\\dot{m}$ ($\\text{kg/h}$)", value=100000.0, step=1000.0)
S_in = st.sidebar.number_input("Azufre entrada $S_{\\text{in}}$ ($\\text{ppm}$)", value=500.0, step=10.0)
Cp = st.sidebar.number_input("Capacidad calorífica $C_p$ ($\\text{kJ/(kg}\\cdot\\text{K)}$)", value=2.5, step=0.1)
Ea = st.sidebar.number_input("Energía de activación $E_a$ ($\\text{J/mol}$)", value=80000.0, step=1000.0)
tau = st.sidebar.number_input("Tiempo espacial $\\tau$ ($\\text{h}$)", value=0.5, step=0.05)
T_alim = st.sidebar.number_input("Temperatura alimentación $T_{\\text{alim}}$ ($\\text{K}$)", value=300.0, step=5.0)

st.sidebar.subheader("Rango y resolución de malla")
T_min = st.sidebar.number_input("$T_{\\mathrm{in}}$ mínima ($\\mathrm{K}$)", value=550.0, step=5.0)
T_max = st.sidebar.number_input("$T_{\\mathrm{in}}$ máxima ($\\mathrm{K}$)", value=650.0, step=5.0)
eps_min = st.sidebar.number_input("$\\epsilon$ mínima", value=0.60, min_value=0.0, max_value=1.0, step=0.05)
eps_max = st.sidebar.number_input("$\\epsilon$ máxima", value=0.90, min_value=0.0, max_value=1.0, step=0.05)
n_grid = st.sidebar.slider("Resolución de malla ($N \\times N$)", min_value=20, max_value=80, value=40, step=5)

params_app = {
    "m_dot": m_dot,
    "S_in": S_in,
    "Cp": Cp,
    "Ea": Ea,
    "R": 8.314,
    "tau": tau,
    "T_alim": T_alim,
}

salida = resolver_optimizacion_hds(T_min, T_max, eps_min, eps_max, n_grid, params_app)
resultados = salida["resultados"]
factibles = salida["factibles"]
pareto = salida["pareto"]
knee = salida["knee"]
f1_n = salida["f1_n"]
f2_n = salida["f2_n"]
a1, a2 = salida["a1"], salida["a2"]
b1, b2 = salida["b1"], salida["b2"]

if len(pareto) >= 2:
    st.subheader("Resultados del Knee Point")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("$T_{\\text{in}}$ óptima ($\\text{K}$)", f"{knee['T_in']:.2f}")
    c2.metric("$S_{\\text{out}}$ ($\\text{ppm}$)", f"{knee['S_out']:.2f}")
    c3.metric("$Q_{\\text{horno}}$ ($\\text{kW}$)", f"{knee['Q_kW']:.2f}")
    c4.metric("Puntos factibles", f"{len(factibles)} / {len(resultados)}")

    st.subheader("Frontera de Pareto y Knee Point")

    f1_fact = [r["f1"] for r in factibles]
    f2_fact = [r["f2"] / 3600.0 for r in factibles]

    f1_par = [p["f1"] for p in pareto]
    f2_par = [p["f2"] / 3600.0 for p in pareto]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(f1_fact, f2_fact, alpha=0.25, label="Factibles ($S_{\\mathrm{out}} \\leq 10\\text{ ppm}$)")
    ax.plot(f1_par, f2_par, marker="o", linewidth=2, label="Frontera de Pareto")
    ax.scatter(
        [knee["f1"]],
        [knee["Q_kW"]],
        s=110,
        zorder=5,
        label="Knee point",
    )

    ax.set_xlabel("$f_1 = 1 - X_S$ (minimizar no-conversión)")
    ax.set_ylabel("$f_2 = Q_{\\mathrm{horno}}$ ($\\mathrm{kW}$)")
    ax.set_title("Compromiso entre no-conversión de azufre y carga del horno")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("Puntos representativos de la frontera de Pareto")
    puntos_clave = [
        {"Rol": "Extremo mayor conversión (menor 1 - X_S)", "p": pareto[0]},
        {"Rol": "Knee point (compromiso)", "p": knee},
        {"Rol": "Extremo menor conversión factible", "p": pareto[-1]},
    ]
    tabla_pareto = [
        {
            "Rol": item["Rol"],
            "T_in (K)": round(item["p"]["T_in"], 2),
            "epsilon": round(item["p"]["epsilon"], 3),
            "X_S": round(item["p"]["X_S"], 4),
            "S_out (ppm)": round(item["p"]["S_out"], 3),
            "Q_horno (kJ/h)": round(item["p"]["Q_kJh"], 1),
            "Q_horno (kW)": round(item["p"]["Q_kW"], 2),
        }
        for item in puntos_clave
    ]
    st.table(tabla_pareto)

    st.subheader("Selección del Knee Point en espacio normalizado")

    fig_norm, ax_n = plt.subplots(figsize=(8, 4))
    ax_n.plot(f1_n, f2_n, marker="o", linewidth=2, label="Frontera normalizada")
    ax_n.plot([a1, b1], [a2, b2], linestyle="--", label="Recta extremos normalizados")
    ax_n.scatter(
        [knee["f1_norm"]],
        [knee["f2_norm"]],
        s=110,
        zorder=5,
        label="Knee point (dist. máxima)",
    )
    ax_n.set_xlabel("$\\tilde{f}_1$ (normalizado)")
    ax_n.set_ylabel("$\\tilde{f}_2$ (normalizado)")
    ax_n.set_title("Distancia perpendicular a la recta de extremos")
    ax_n.grid(True, linestyle="--", alpha=0.4)
    ax_n.legend()
    plt.tight_layout()

    st.pyplot(fig_norm)

    st.subheader("Análisis de sensibilidad ($\\pm 10\\%$) en el Knee Point")

    T_op = knee["T_in"]
    eps_op = knee["epsilon"]
    k0_base = calcular_k0(params_app["Ea"], params_app["tau"], params_app["R"])

    filas_sens = []
    for nombre_param in ["m_dot", "Cp", "Ea", "k0", "tau"]:
        for factor, etiqueta in [(0.90, "-10%"), (1.10, "+10%")]:
            p_copy = params_app.copy()
            k0_usado = k0_base * factor if nombre_param == "k0" else k0_base
            if nombre_param != "k0":
                p_copy[nombre_param] = p_copy[nombre_param] * factor

            k_pert = calcular_k(T_op, k0_usado, p_copy["Ea"], p_copy["R"])
            X_pert = calcular_conversion(k_pert, p_copy["tau"])
            S_pert = calcular_azufre_salida(p_copy["S_in"], X_pert)
            T_efl_pert = calcular_t_efluente(T_op, X_pert)
            T_int_pert = calcular_t_intercambiador(T_efl_pert, eps_op, p_copy["T_alim"])
            Q_kJh_pert, Q_kW_pert = calcular_q_horno(p_copy["m_dot"], p_copy["Cp"], T_op, T_int_pert)

            filas_sens.append({
                "Parámetro": nombre_param,
                "Variación": etiqueta,
                "S_out (ppm)": round(S_pert, 3),
                "Factible (<=10 ppm)": bool(S_pert <= 10.0),
                "Q_horno (kJ/h)": round(Q_kJh_pert, 1),
                "Q_horno (kW)": round(Q_kW_pert, 2),
            })

    st.dataframe(filas_sens, use_container_width=True)

    st.subheader("Exploración de $S_{\\mathrm{in}}$ (450–550 ppm)")
    filas_sin = []
    for s_val in [450.0, 500.0, 550.0]:
        p_s = params_app.copy()
        p_s["S_in"] = s_val
        k0_s = calcular_k0(p_s["Ea"], p_s["tau"], p_s["R"])
        k_s = calcular_k(T_op, k0_s, p_s["Ea"], p_s["R"])
        X_s = calcular_conversion(k_s, p_s["tau"])
        S_s = calcular_azufre_salida(p_s["S_in"], X_s)
        filas_sin.append({
            "S_in (ppm)": s_val,
            "S_out en Knee (ppm)": round(S_s, 3),
            "Factible (<=10 ppm)": bool(S_s <= 10.0),
        })

    st.table(filas_sin)

else:
    st.warning("No hay suficientes puntos factibles para formar una frontera de Pareto con estos parámetros.")
