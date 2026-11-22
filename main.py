import numpy as np


# Calibración de k0
def calcular_k0(Ea, tau, R=8.314, T_ref=600.0, X_ref=0.985):
    num = -np.log(1 - X_ref)
    den = tau * np.exp(-Ea / (R * T_ref))
    k0 = num / den
    return k0

# Cálculo de k(T_in)
def calcular_k(T_in, k0, Ea, R=8.314):
    k = k0 * np.exp(-Ea / (R * T_in))
    return k

# Cálculo de la conversión X_S
def calcular_conversion(k, tau):
    X_S = 1 - np.exp(-k * tau)
    return X_S

# Azufre a la salida S_out
def calcular_azufre_salida(S_in, X_S):
    S_out = S_in * (1 - X_S)
    return S_out

# Temperatura del efluente
def calcular_t_efluente(T_in, X_S, delta_T_max=15.0):
    delta_T = delta_T_max * X_S
    T_efluente = T_in + delta_T
    return T_efluente

# Temperatura del intercambiador
def calcular_t_intercambiador(T_efluente, epsilon, T_alim=300.0):
    T_intercambiador = T_alim + epsilon * (T_efluente - T_alim)
    return T_intercambiador

# Balance de energía del horno (devuelve kJ/h y kW)
def calcular_q_horno(m_dot, Cp, T_in, T_intercambiador):
    Q_kJh = m_dot * Cp * (T_in - T_intercambiador)
    Q_kW = Q_kJh / 3600.0
    return Q_kJh, Q_kW

#SANITY CHECKS:
parametros_base = {
    "m_dot": 100000.0,
    "S_in": 500.0,
    "Cp": 2.5,
    "Ea": 80000.0,
    "R": 8.314,
    "tau": 0.5,
    "T_alim": 300.0,
}

def evaluar_punto(T_in, epsilon, params=parametros_base):
    k0 = calcular_k0(params["Ea"], params["tau"], params["R"])
    k = calcular_k(T_in, k0, params["Ea"], params["R"])
    X_S = calcular_conversion(k, params["tau"])
    S_out = calcular_azufre_salida(params["S_in"], X_S)
    factible = bool(S_out <= 10.0)

    T_efluente = calcular_t_efluente(T_in, X_S)
    T_intercambiador = calcular_t_intercambiador(T_efluente, epsilon, params["T_alim"])
    Q_kJh, Q_kW = calcular_q_horno(params["m_dot"], params["Cp"], T_in, T_intercambiador)

    return {
        "T_in": T_in,
        "epsilon": epsilon,
        "X_S": X_S,
        "S_out": S_out,
        "factible": factible,
        "T_efluente": T_efluente,
        "T_intercambiador": T_intercambiador,
        "Q_kJh": Q_kJh,
        "Q_kW": Q_kW,
    }


def resolver_optimizacion_hds(T_min, T_max, eps_min, eps_max, n_grid, params=parametros_base):
    T_vals = np.linspace(T_min, T_max, n_grid)
    eps_vals = np.linspace(eps_min, eps_max, n_grid)

    resultados = []
    for T_in in T_vals:
        for eps in eps_vals:
            k0 = calcular_k0(params["Ea"], params["tau"], params["R"])
            k = calcular_k(T_in, k0, params["Ea"], params["R"])
            X_S = calcular_conversion(k, params["tau"])
            S_out = calcular_azufre_salida(params["S_in"], X_S)
            factible = bool(S_out <= 10.0)

            T_efl = calcular_t_efluente(T_in, X_S)
            T_int = calcular_t_intercambiador(T_efl, eps, params["T_alim"])
            Q_kJh, Q_kW = calcular_q_horno(params["m_dot"], params["Cp"], T_in, T_int)

            resultados.append({
                "T_in": T_in,
                "epsilon": eps,
                "X_S": X_S,
                "S_out": S_out,
                "factible": factible,
                "Q_kJh": Q_kJh,
                "Q_kW": Q_kW,
                "f1": 1.0 - X_S,
                "f2": Q_kJh,
            })

    factibles = [r for r in resultados if r["factible"]]

    pareto = []
    for A in factibles:
        dominado = any(
            (B["f1"] <= A["f1"] and B["f2"] <= A["f2"]) and (B["f1"] < A["f1"] or B["f2"] < A["f2"])
            for B in factibles
        )
        if not dominado:
            pareto.append(A)

    pareto.sort(key=lambda p: p["f1"])

    knee = None
    f1_n = np.array([])
    f2_n = np.array([])
    a1 = a2 = b1 = b2 = 0.0

    if len(pareto) >= 2:
        f1_p = np.array([p["f1"] for p in pareto])
        f2_p = np.array([p["f2"] for p in pareto])

        r1 = np.max(f1_p) - np.min(f1_p)
        r2 = np.max(f2_p) - np.min(f2_p)

        f1_n = (f1_p - np.min(f1_p)) / r1 if r1 != 0 else np.zeros_like(f1_p)
        f2_n = (f2_p - np.min(f2_p)) / r2 if r2 != 0 else np.zeros_like(f2_p)

        a1, a2 = f1_n[0], f2_n[0]
        b1, b2 = f1_n[-1], f2_n[-1]
        den = np.sqrt((b1 - a1) ** 2 + (b2 - a2) ** 2)

        for i, p in enumerate(pareto):
            num = abs((b1 - a1) * (a2 - f2_n[i]) - (a1 - f1_n[i]) * (b2 - a2))
            p["dist_knee"] = (num / den) if den != 0 else 0.0
            p["f1_norm"] = float(f1_n[i])
            p["f2_norm"] = float(f2_n[i])

        knee = max(pareto, key=lambda p: p["dist_knee"])

    return {
        "resultados": resultados,
        "factibles": factibles,
        "pareto": pareto,
        "knee": knee,
        "f1_n": f1_n,
        "f2_n": f2_n,
        "a1": a1,
        "a2": a2,
        "b1": b1,
        "b2": b2,
    }


if __name__ == "__main__":
    # Sanity check 1: T_in = 650 K, epsilon = 0.90
    sc1 = evaluar_punto(650.0, 0.90)

    # Sanity check 2: T_in = 550 K, epsilon = 0.75
    sc2 = evaluar_punto(550.0, 0.75)

    # Sanity check 3: epsilon -> 1 (prueba por ejemplo con T_in = 600.0 y epsilon = 0.999999)
    sc3 = evaluar_punto(600.0, 0.999999)

    print(
        f"SC1 -> T_in: {sc1['T_in']:.1f} K | eps: {sc1['epsilon']:.2f} | "
        f"X_S: {sc1['X_S']:.4f} | S_out: {sc1['S_out']:.2f} ppm | "
        f"Factible: {sc1['factible']} | Q_horno: {sc1['Q_kJh']:.1f} kJ/h ({sc1['Q_kW']:.2f} kW)"
    )

    print(
        f"SC2 -> T_in: {sc2['T_in']:.1f} K | eps: {sc2['epsilon']:.2f} | "
        f"X_S: {sc2['X_S']:.4f} | S_out: {sc2['S_out']:.2f} ppm | "
        f"Factible: {sc2['factible']} | Q_horno: {sc2['Q_kJh']:.1f} kJ/h ({sc2['Q_kW']:.2f} kW)"
    )

    print(
        f"SC3 -> T_in: {sc3['T_in']:.1f} K | eps: {sc3['epsilon']:.2f} | "
        f"X_S: {sc3['X_S']:.4f} | S_out: {sc3['S_out']:.2f} ppm | "
        f"Factible: {sc3['factible']} | Q_horno: {sc3['Q_kJh']:.1f} kJ/h ({sc3['Q_kW']:.2f} kW)"
    )

    # Crear cuadricula (Grid search)

    # Vectores
    T_vals = np.linspace(550.0, 650.0, 40)
    eps_vals = np.linspace(0.60, 0.90, 40)  # (inicio, final, cantidad)

    resultados_malla = []  # Esto es una lista vacia

    # Recorrer todas las combinaciones con dos bucles
    for T_in in T_vals:
        for eps in eps_vals:
            punto = evaluar_punto(T_in, eps, parametros_base)
            punto["f1_min_azufre"] = 1 - punto["X_S"]
            punto["f2_min_Q"] = punto["Q_kJh"]
            resultados_malla.append(punto)

    # Filtrar los puntos factibles
    factibles = []
    for r in resultados_malla:
        if r["factible"]:
            factibles.append(r)

    # Construir pareto
    pareto = []

    for A in factibles:
        esta_dominado = False
        for B in factibles:
            no_peor_en_nada = (
                B["f1_min_azufre"] <= A["f1_min_azufre"]
                and B["f2_min_Q"] <= A["f2_min_Q"]
            )
            estrictamente_mejor_en_algo = (
                B["f1_min_azufre"] < A["f1_min_azufre"]
                or B["f2_min_Q"] < A["f2_min_Q"]
            )

            if no_peor_en_nada and estrictamente_mejor_en_algo:
                esta_dominado = True
                break

        if not esta_dominado:
            pareto.append(A)

    pareto.sort(key=lambda p: p["f1_min_azufre"])

    fraccion_factible = len(factibles) / len(resultados_malla)
    print(f"Total puntos malla: {len(resultados_malla)}")
    print(f"Puntos factibles: {len(factibles)} ({fraccion_factible:.2%})")
    print(f"Puntos en frontera de Pareto: {len(pareto)}")

    # PASO 3

    # Crear arrays con todos los valores de f1_min_azufre y otro con f2_min_Q
    f1_pareto = np.array([p["f1_min_azufre"] for p in pareto])
    f2_pareto = np.array([p["f2_min_Q"] for p in pareto])

    # Normalizar min-max
    rango_f1 = np.max(f1_pareto) - np.min(f1_pareto)
    rango_f2 = np.max(f2_pareto) - np.min(f2_pareto)

    if rango_f1 != 0:
        f1_norm = (f1_pareto - np.min(f1_pareto)) / rango_f1
    else:
        f1_norm = np.zeros_like(f1_pareto)

    if rango_f2 != 0:
        f2_norm = (f2_pareto - np.min(f2_pareto)) / rango_f2
    else:
        f2_norm = np.zeros_like(f2_pareto)

    # Tomar los dos extremos de la frontera normalizada
    a1, a2 = f1_norm[0], f2_norm[0]
    b1, b2 = f1_norm[-1], f2_norm[-1]

    # Distancia perpendicular de cada punto a la recta

    for i in range(len(pareto)):
        p1 = f1_norm[i]
        p2 = f2_norm[i]

        numerador = abs((b1 - a1) * (a2 - p2) - (a1 - p1) * (b2 - a2))
        denominador = np.sqrt((b1 - a1) ** 2 + (b2 - a2) ** 2)

        if denominador != 0:
            dist = numerador / denominador
        else:
            dist = 0.0

        pareto[i]["f1_norm"] = p1
        pareto[i]["f2_norm"] = p2
        pareto[i]["dist_knee"] = dist

    knee_point = max(pareto, key=lambda p: p["dist_knee"])

    print(
        f"Knee point -> T_in: {knee_point['T_in']:.2f} K | eps: {knee_point['epsilon']:.2f} | "
        f"X_S: {knee_point['X_S']:.4f} | S_out: {knee_point['S_out']:.2f} ppm | "
        f"Q_horno: {knee_point['Q_kJh']:.1f} kJ/h ({knee_point['Q_kW']:.2f} kW)"
    )

    # Paso 4.

    s_out_malla = np.array([r["S_out"] for r in resultados_malla])
    q_kjh_malla = np.array([r["Q_kJh"] for r in resultados_malla])
    q_kw_malla = np.array([r["Q_kW"] for r in resultados_malla])

    print(f"Rango S_out (malla): {np.min(s_out_malla):.2f} a {np.max(s_out_malla):.2f} ppm")
    print(f"Rango Q_horno (malla kJ/h): {np.min(q_kjh_malla):.1f} a {np.max(q_kjh_malla):.1f}")
    print(f"Rango Q_horno (malla kW): {np.min(q_kw_malla):.2f} a {np.max(q_kw_malla):.2f}")

    # Sensibilidad de +10%
    T_op = knee_point["T_in"]
    eps_op = knee_point["epsilon"]
    k0_base = calcular_k0(parametros_base["Ea"], parametros_base["tau"], parametros_base["R"])

    parametros_sens = ["m_dot", "Cp", "Ea", "k0", "tau"]

    for nombre_param in parametros_sens:
        for factor, etiqueta in [(0.90, "-10%"), (1.10, "+10%")]:
            p = parametros_base.copy()  # Copia para no dañar el diccionario original
            k0_usado = k0_base

            if nombre_param == "k0":
                k0_usado = k0_base * factor
            else:
                p[nombre_param] = p[nombre_param] * factor

            k_pert = calcular_k(T_op, k0_usado, p["Ea"], p["R"])
            X_pert = calcular_conversion(k_pert, p["tau"])
            S_pert = calcular_azufre_salida(p["S_in"], X_pert)
            T_efl_pert = calcular_t_efluente(T_op, X_pert)
            T_int_pert = calcular_t_intercambiador(T_efl_pert, eps_op, p["T_alim"])
            Q_kJh_pert, Q_kW_pert = calcular_q_horno(p["m_dot"], p["Cp"], T_op, T_int_pert)

            print(
                f"{nombre_param:6s} ({etiqueta}) -> S_out: {S_pert:.3f} ppm | "
                f"Q_horno: {Q_kJh_pert:.1f} kJ/h ({Q_kW_pert:.2f} kW)"
            )

    # Explorar S_in
    for s_in_prueba in [450.0, 500.0, 550.0]:
        p_s = parametros_base.copy()
        p_s["S_in"] = s_in_prueba
        res_s = evaluar_punto(T_op, eps_op, p_s)
        print(
            f"S_in = {s_in_prueba:.0f} ppm -> S_out en knee: {res_s['S_out']:.3f} ppm | "
            f"Factible: {res_s['factible']}"
        )