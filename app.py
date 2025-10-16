# app.py
import streamlit as st
import numpy as np
import pandas as pd
import math
import io

st.set_page_config(page_title="Truss Solver Online (Interativo)", layout="wide")

# -------------------------
# Utilidades de sesión
# -------------------------
if "nodes" not in st.session_state:
    st.session_state.nodes = []   # list of dicts {'x':float,'y':float}
if "bars" not in st.session_state:
    st.session_state.bars = []    # list of dicts {'n1':int,'n2':int,'A':float,'E':float}
if "loads" not in st.session_state:
    st.session_state.loads = {}   # node_index -> (Px,Py)
if "supports" not in st.session_state:
    st.session_state.supports = {}# node_index -> (ux_locked(bool), uy_locked(bool))
if "last_solution" not in st.session_state:
    st.session_state.last_solution = None

# -------------------------
# Helper functions
# -------------------------
def add_node(x, y):
    st.session_state.nodes.append({'x': float(x), 'y': float(y)})

def add_bar(n1, n2, A=0.001, E=210e9):
    # Basic checks (indices)
    n = len(st.session_state.nodes)
    if n1 < 0 or n1 >= n or n2 < 0 or n2 >= n:
        st.warning("Índices de nodo inválidos al añadir barra.")
        return
    if n1 == n2:
        st.warning("No se puede crear una barra entre el mismo nodo.")
        return
    st.session_state.bars.append({'n1': int(n1), 'n2': int(n2), 'A': float(A), 'E': float(E)})

def remove_node(idx):
    # remove node and any connected bars; adjust connectivity
    if idx < 0 or idx >= len(st.session_state.nodes):
        return
    # remove bars that include this node
    new_bars = []
    for b in st.session_state.bars:
        if b['n1'] == idx or b['n2'] == idx:
            continue
        # fix indices greater than idx
        b2 = {'n1': b['n1'], 'n2': b['n2'], 'A': b['A'], 'E': b['E']}
        if b2['n1'] > idx: b2['n1'] -= 1
        if b2['n2'] > idx: b2['n2'] -= 1
        new_bars.append(b2)
    st.session_state.bars = new_bars
    # remove loads/supports referencing node
    st.session_state.nodes.pop(idx)
    st.session_state.loads = { (k if k<idx else k-1):v for k,v in st.session_state.loads.items() if k!=idx }
    st.session_state.supports = { (k if k<idx else k-1):v for k,v in st.session_state.supports.items() if k!=idx }

def remove_bar(idx):
    if idx < 0 or idx >= len(st.session_state.bars):
        return
    st.session_state.bars.pop(idx)

def set_support(node_idx, ux, uy):
    if ux or uy:
        st.session_state.supports[node_idx] = (bool(ux), bool(uy))
    elif node_idx in st.session_state.supports:
        st.session_state.supports.pop(node_idx)

def set_load(node_idx, px, py):
    if (abs(px) > 0.0) or (abs(py) > 0.0):
        st.session_state.loads[node_idx] = (float(px), float(py))
    elif node_idx in st.session_state.loads:
        st.session_state.loads.pop(node_idx)

# solver
def assemble_and_solve():
    nodes = st.session_state.nodes
    bars = st.session_state.bars
    loads = st.session_state.loads
    supports = st.session_state.supports

    n = len(nodes)
    if n == 0:
        raise ValueError("No hay nodos definidos.")
    if len(bars) == 0:
        raise ValueError("No hay barras definidas.")

    dof = 2*n
    K = np.zeros((dof, dof))
    F = np.zeros(dof)

    # loads vector
    for ni, (Px, Py) in loads.items():
        F[2*ni] = Px
        F[2*ni+1] = Py

    # assemble K
    for b in bars:
        n1 = b['n1']; n2 = b['n2']; A = b['A']; E = b['E']
        x1, y1 = nodes[n1]['x'], nodes[n1]['y']
        x2, y2 = nodes[n2]['x'], nodes[n2]['y']
        L = math.hypot(x2-x1, y2-y1)
        if L == 0:
            raise ValueError(f"Barra con longitud cero entre N{n1} y N{n2}.")
        c = (x2-x1)/L; s = (y2-y1)/L
        k = E*A/L
        k_local = k * np.array([[ c*c, c*s, -c*c, -c*s],
                                [ c*s, s*s, -c*s, -s*s],
                                [-c*c,-c*s,  c*c,  c*s],
                                [-c*s,-s*s,  c*s,  s*s]])
        dof_map = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        for i in range(4):
            for j in range(4):
                K[dof_map[i], dof_map[j]] += k_local[i,j]

    # supports
    fixed = []
    for ni, (ux, uy) in supports.items():
        if ux: fixed.append(2*ni)
        if uy: fixed.append(2*ni+1)
    free = [i for i in range(dof) if i not in fixed]
    if len(free) == 0:
        raise ValueError("No hay grados de libertad libres para resolver.")

    # partition and solve
    Kff = K[np.ix_(free, free)]
    Ff = F[free]
    U = np.zeros(dof)
    Uf = np.linalg.solve(Kff, Ff)
    U[free] = Uf
    reactions = K.dot(U) - F

    # axial forces
    bar_results = []
    for idx, b in enumerate(bars):
        n1 = b['n1']; n2 = b['n2']; A = b['A']; E = b['E']
        x1, y1 = nodes[n1]['x'], nodes[n1]['y']
        x2, y2 = nodes[n2]['x'], nodes[n2]['y']
        L = math.hypot(x2-x1, y2-y1)
        c = (x2-x1)/L; s = (y2-y1)/L
        u_local = np.array([U[2*n1], U[2*n1+1], U[2*n2], U[2*n2+1]])
        delta = (-c*u_local[0] - s*u_local[1] + c*u_local[2] + s*u_local[3])
        N = E*A/L * delta
        bar_results.append({'bar': idx, 'n1': n1, 'n2': n2, 'L_m': L, 'axial_N': float(N)})
    # pack results
    disp = [{'node': i, 'Ux_m': float(U[2*i]), 'Uy_m': float(U[2*i+1])} for i in range(n)]
    reac = [{'node': i, 'Rx_N': float(reactions[2*i]), 'Ry_N': float(reactions[2*i+1])} for i in range(n)]
    return {'U': disp, 'R': reac, 'bars': bar_results}

# -------------------------
# Layout: UI
# -------------------------
st.sidebar.title("Herramientas")
with st.sidebar.expander("Modelo rápido (ejemplo)"):
    if st.button("Cargar ejemplo: Triángulo simple"):
        # clear
        st.session_state.nodes = []
        st.session_state.bars = []
        st.session_state.loads = {}
        st.session_state.supports = {}
        # add nodes (triangle)
        add_node(0.0, 0.0)   # N0
        add_node(3.0, 0.0)   # N1
        add_node(1.5, 2.0)   # N2
        add_bar(0,1, A=0.005)
        add_bar(1,2, A=0.005)
        add_bar(2,0, A=0.005)
        set_support(0, True, True)
        set_support(1, False, True)
        set_load(2, 0.0, -1000.0)
        st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Añadir nodo (manual)")
colx, coly = st.sidebar.columns(2)
with colx:
    new_x = st.number_input("X (m)", value=0.0, key="add_x")
with coly:
    new_y = st.number_input("Y (m)", value=0.0, key="add_y")
if st.sidebar.button("➕ Agregar nodo"):
    add_node(new_x, new_y)
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Añadir barra")
if len(st.session_state.nodes) >= 2:
    n_options = list(range(len(st.session_state.nodes)))
    sel_n1 = st.sidebar.selectbox("Nodo inicio (index)", options=n_options, index=0, key="sel_n1")
    sel_n2 = st.sidebar.selectbox("Nodo fin (index)", options=n_options, index=1, key="sel_n2")
    A_input = st.sidebar.number_input("Área A (m²)", value=0.001, format="%.6f", step=0.0001)
    E_input = st.sidebar.number_input("E (Pa)", value=210e9, format="%.1f", step=1.0)
    if st.sidebar.button("➕ Agregar barra"):
        add_bar(sel_n1, sel_n2, A=A_input, E=E_input)
        st.experimental_rerun()
else:
    st.sidebar.info("Crea al menos 2 nodos para añadir barras")

st.sidebar.markdown("---")
st.sidebar.subheader("Operaciones rápidas")
if st.sidebar.button("Limpiar todo"):
    st.session_state.nodes = []
    st.session_state.bars = []
    st.session_state.loads = {}
    st.session_state.supports = {}
    st.session_state.last_solution = None
    st.experimental_rerun()

# -------------------------
# Main area: model table + plot + editor
# -------------------------
col_model, col_plot = st.columns([1, 2])

with col_model:
    st.subheader("Nodos")
    # show nodes table with remove button per row
    if len(st.session_state.nodes) == 0:
        st.info("No hay nodos (usa la barra lateral para agregar).")
    else:
        nodes_df = pd.DataFrame(st.session_state.nodes).reset_index().rename(columns={'index':'node'})
        st.dataframe(nodes_df.style.format({"x":"{:.3f}", "y":"{:.3f}"}), use_container_width=True)
        remove_idx = st.number_input("Eliminar nodo (index)", min_value=0, max_value=max(0,len(st.session_state.nodes)-1), value=0)
        if st.button("🗑️ Eliminar nodo seleccionado"):
            remove_node(remove_idx)
            st.experimental_rerun()

    st.markdown("---")
    st.subheader("Barras")
    if len(st.session_state.bars) == 0:
        st.info("No hay barras definidas.")
    else:
        bars_df = pd.DataFrame(st.session_state.bars).reset_index().rename(columns={'index':'bar'})
        st.dataframe(bars_df.style.format({"A":"{:.6f}", "E":"{:.3e}"}), use_container_width=True)
        remove_bar_idx = st.number_input("Eliminar barra (index)", min_value=0, max_value=max(0,len(st.session_state.bars)-1), value=0)
        if st.button("🗑️ Eliminar barra seleccionada"):
            remove_bar(remove_bar_idx)
            st.experimental_rerun()

    st.markdown("---")
    st.subheader("Seleccionar nodo / editar cargas y apoyos")
    if len(st.session_state.nodes) > 0:
        n_idx = st.number_input("Nodo index para editar", min_value=0, max_value=len(st.session_state.nodes)-1, value=0, key="edit_node_idx")
        # show coords
        ndata = st.session_state.nodes[n_idx]
        st.write(f"N{n_idx} — x={ndata['x']:.3f} m, y={ndata['y']:.3f} m")
        ux = st.checkbox("Restringir Ux (Ux = 0)", value= st.session_state.supports.get(n_idx, (False,False))[0], key="ux_chk")
        uy = st.checkbox("Restringir Uy (Uy = 0)", value= st.session_state.supports.get(n_idx, (False,False))[1], key="uy_chk")
        if st.button("Aplicar apoyos a nodo"):
            set_support(n_idx, ux, uy)
            st.experimental_rerun()
        px = st.number_input("Px (N)", value=st.session_state.loads.get(n_idx, (0.0,0.0))[0], key="px_in")
        py = st.number_input("Py (N)", value=st.session_state.loads.get(n_idx, (0.0,0.0))[1], key="py_in")
        if st.button("Aplicar carga al nodo"):
            set_load(n_idx, px, py)
            st.experimental_rerun()
    else:
        st.info("Agrega nodos para editar.")

with col_plot:
    st.subheader("Visualización de la armadura")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,6))
    # draw bars
    for i, b in enumerate(st.session_state.bars):
        n1 = st.session_state.nodes[b['n1']]; n2 = st.session_state.nodes[b['n2']]
        xs = [n1['x'], n2['x']]; ys = [n1['y'], n2['y']]
        ax.plot(xs, ys, '-o', color='tab:blue')
        mx, my = (xs[0]+xs[1])/2, (ys[0]+ys[1])/2
        ax.text(mx, my, f"B{i}", color='blue')
    # draw nodes with labels & supports/loads
    for i, n in enumerate(st.session_state.nodes):
        ax.plot(n['x'], n['y'], 'ko')
        ax.text(n['x']+0.02, n['y']+0.02, f"N{i}")
        if i in st.session_state.supports:
            ux, uy = st.session_state.supports[i]
            lbl = ""
            if ux and uy: lbl = "⏹"
            elif ux: lbl = "⟂Ux"
            elif uy: lbl = "⟂Uy"
            ax.text(n['x']-0.1, n['y']-0.05, lbl, color='green')
        if i in st.session_state.loads:
            Px, Py = st.session_state.loads[i]
            ax.arrow(n['x'], n['y'], Px/2000.0, Py/2000.0, head_width=0.03, color='red')
            ax.text(n['x']+0.02, n['y']-0.08, f"{int(Px)},{int(Py)} N", color='red')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.grid(True)
    st.pyplot(fig)

# -------------------------
# Solve area
# -------------------------
st.markdown("---")
st.header("Resolver y resultados")
col1, col2 = st.columns([1,1])

with col1:
    if st.button("▶️ Resolver armadura"):
        try:
            sol = assemble_and_solve()
            st.session_state.last_solution = sol
            st.success("Cálculo completado correctamente ✅")
        except Exception as e:
            st.error(f"Error al resolver: {e}")

with col2:
    if st.button("📥 Descargar último resultado (ZIP CSV)"):
        sol = st.session_state.last_solution
        if sol is None:
            st.warning("Primero debes resolver para generar resultados.")
        else:
            # prepare CSVs in memory
            df_nodes = pd.DataFrame(sol['U']).merge(pd.DataFrame(sol['R']), on='node')
            df_bars = pd.DataFrame(sol['bars'])
            # create zip-like in memory (multiple files)
            csv1 = df_nodes.to_csv(index=False).encode('utf-8')
            csv2 = df_bars.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar nodos (CSV)", csv1, "nodes_results.csv", "text/csv")
            st.download_button("Descargar barras (CSV)", csv2, "bars_results.csv", "text/csv")

# show results if available
if st.session_state.last_solution:
    st.subheader("Resultados recientes")
    sol = st.session_state.last_solution
    df_nodes = pd.DataFrame(sol['U']).merge(pd.DataFrame(sol['R']), on='node')
    df_bars = pd.DataFrame(sol['bars'])
    st.write("Desplazamientos y reacciones por nodo")
    st.dataframe(df_nodes.style.format({"Ux_m":"{:.6e}", "Uy_m":"{:.6e}", "Rx_N":"{:.3f}", "Ry_N":"{:.3f}"}))
    st.write("Fuerzas axiales por barra")
    st.dataframe(df_bars.style.format({"L_m":"{:.4f}", "axial_N":"{:.3f}"}))

# -------------------------
# Instrucciones (expander)
# -------------------------
st.sidebar.markdown("---")
with st.sidebar.expander("🛈 Instrucciones de uso (paso a paso)", expanded=True):
    st.markdown("""
    **Guía rápida para usar la app**  

    1. **Agregar nodos**  
       - Usa el panel lateral *Añadir nodo* para crear nodos con coordenadas (m).  
       - Alternativamente, carga el *ejemplo* para probar rápidamente.

    2. **Agregar barras**  
       - En *Añadir barra* selecciona el índice de nodo inicio y nodo fin (índices comienzan en 0).  
       - Define Área (m²) y E (Pa) si quieres valores distintos y pulsa *Agregar barra*.

    3. **Definir apoyos y cargas**  
       - En *Seleccionar nodo / editar cargas y apoyos* elige un índice de nodo:  
         - Marca `Restringir Ux` o `Restringir Uy` para definir apoyos.  
         - Introduce Px, Py (N) y pulsa *Aplicar carga al nodo*.

    4. **Visualizar**  
       - El panel de la derecha muestra la armadura (nodos, barras, apoyos y vectores de carga).

    5. **Resolver**  
       - Pulsa *Resolver armadura*. Si el sistema está definido y resolvible verás resultados.  
       - Las reacciones se calculan como R = K*U - F.

    6. **Exportar**  
       - Tras resolver puedes descargar los resultados en CSV (nodos / barras).

    **Consejos y advertencias**  
    - Asegúrate de tener suficientes apoyos para evitar mecanismos (matriz singular).  
    - Usa unidades coherentes (m, N, Pa).  
    - Si obtienes errores verifica que no hay barras con longitud 0 ni nodos duplicados en la misma posición.
    """)

# -------------------------
# Footer / créditos
# -------------------------
st.markdown("---")
st.write("Desarrollado con ❤️ — Si quieres mejoras (dibujo con mouse, guardado JSON, exportar PDF o reportes) dime y lo agrego.")
