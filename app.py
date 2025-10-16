import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Truss Solver Online", layout="wide")

st.title("🔩 Calculadora de Fuerzas Internas en una Armadura 2D")
st.markdown("""
Esta aplicación permite ingresar los **nodos y barras** de una armadura 2D, 
aplicar **cargas nodales**, definir **apoyos** y obtener las **fuerzas axiales** 
en cada barra y las **reacciones** en los apoyos.
""")

# --- Panel lateral ---
st.sidebar.header("Definición del modelo")

n_nodes = st.sidebar.number_input("Número de nodos", min_value=2, value=3, step=1)
n_bars = st.sidebar.number_input("Número de barras", min_value=1, value=3, step=1)

st.sidebar.markdown("---")
st.sidebar.write("📘 Define coordenadas y conectividad abajo ⬇️")

# --- Tablas interactivas ---
st.subheader("📍 Coordenadas de los nodos")
nodes_df = pd.DataFrame({
    "x (m)": [0.0]*n_nodes,
    "y (m)": [0.0]*n_nodes
})
nodes = st.data_editor(nodes_df, use_container_width=True)

st.subheader("🔗 Conectividad de las barras")
bars_df = pd.DataFrame({
    "Nodo i": [1]*n_bars,
    "Nodo j": [2]*n_bars,
    "Área (m²)": [0.001]*n_bars,
    "E (Pa)": [2.1e11]*n_bars
})
bars = st.data_editor(bars_df, use_container_width=True)

st.subheader("🧱 Condiciones de apoyo (1 = fijo, 0 = libre)")
supports_df = pd.DataFrame({
    "Ux fijo": [0]*n_nodes,
    "Uy fijo": [0]*n_nodes
})
supports = st.data_editor(supports_df, use_container_width=True)

st.subheader("⚡ Cargas nodales (N)")
loads_df = pd.DataFrame({
    "Fx": [0.0]*n_nodes,
    "Fy": [0.0]*n_nodes
})
loads = st.data_editor(loads_df, use_container_width=True)

# --- Botón de cálculo ---
if st.button("🔹 Resolver armadura"):
    try:
        nn = len(nodes)
        dof = 2*nn
        K = np.zeros((dof, dof))
        F = np.zeros((dof, 1))

        # Ensamble matriz global
        for idx, bar in bars.iterrows():
            ni = int(bar["Nodo i"]) - 1
            nj = int(bar["Nodo j"]) - 1
            xi, yi = nodes.loc[ni, ["x (m)", "y (m)"]]
            xj, yj = nodes.loc[nj, ["x (m)", "y (m)"]]
            L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
            c = (xj - xi)/L
            s = (yj - yi)/L
            A = bar["Área (m²)"]
            E = bar["E (Pa)"]

            k_local = (A*E/L) * np.array([
                [ c*c,  c*s, -c*c, -c*s],
                [ c*s,  s*s, -c*s, -s*s],
                [-c*c, -c*s,  c*c,  c*s],
                [-c*s, -s*s,  c*s,  s*s]
            ])

            dof_map = [2*ni, 2*ni+1, 2*nj, 2*nj+1]
            for a in range(4):
                for b in range(4):
                    K[dof_map[a], dof_map[b]] += k_local[a, b]

        # Vector de cargas
        for i in range(nn):
            F[2*i] = loads.loc[i, "Fx"]
            F[2*i+1] = loads.loc[i, "Fy"]

        # Restricciones
        fixed_dofs = []
        for i in range(nn):
            if supports.loc[i, "Ux fijo"] == 1:
                fixed_dofs.append(2*i)
            if supports.loc[i, "Uy fijo"] == 1:
                fixed_dofs.append(2*i+1)
        free_dofs = list(set(range(dof)) - set(fixed_dofs))

        # Resolver desplazamientos
        Kff = K[np.ix_(free_dofs, free_dofs)]
        Ff = F[free_dofs]
        Uf = np.linalg.solve(Kff, Ff)
        U = np.zeros((dof, 1))
        for i, dof_idx in enumerate(free_dofs):
            U[dof_idx] = Uf[i]

        # Reacciones
        R = K @ U - F

        # Fuerzas axiales
        axial_forces = []
        for idx, bar in bars.iterrows():
            ni = int(bar["Nodo i"]) - 1
            nj = int(bar["Nodo j"]) - 1
            xi, yi = nodes.loc[ni, ["x (m)", "y (m)"]]
            xj, yj = nodes.loc[nj, ["x (m)", "y (m)"]]
            L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
            c = (xj - xi)/L
            s = (yj - yi)/L
            A = bar["Área (m²)"]
            E = bar["E (Pa)"]

            ui = U[2*ni:2*ni+2].flatten()
            uj = U[2*nj:2*nj+2].flatten()
            u_local = np.array([-c, -s, c, s]) @ np.concatenate([ui, uj])
            N = (A * E / L) * u_local
            axial_forces.append(N)

        bars["Fuerza axial (N)"] = axial_forces
        st.success("✅ Cálculo completado correctamente.")

        # Mostrar resultados
        disp = pd.DataFrame(U.reshape(-1, 2), columns=["Ux (m)", "Uy (m)"])
        reac = pd.DataFrame(R.reshape(-1, 2), columns=["Rx (N)", "Ry (N)"])

        st.subheader("📈 Desplazamientos nodales")
        st.dataframe(disp)

        st.subheader("📉 Reacciones")
        st.dataframe(reac)

        st.subheader("🔩 Fuerzas axiales en barras")
        st.dataframe(bars[["Nodo i", "Nodo j", "Fuerza axial (N)"]])

        # Exportar resultados
        csv = bars.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Descargar resultados (CSV)", csv, "resultados_armadura.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error en el cálculo: {e}")

