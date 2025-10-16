# app.py
import streamlit as st
import numpy as np
import pandas as pd
import math
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Truss Solver - Mejorado", layout="wide")

# ---------------- session state inicial ----------------
if "nodes" not in st.session_state:
    st.session_state.nodes = []   # list of dict {'x','y'}
if "bars" not in st.session_state:
    st.session_state.bars = []    # list of dict {'n1','n2','A','E'}
if "loads" not in st.session_state:
    st.session_state.loads = {}   # node -> (Fx,Fy)
if "supports" not in st.session_state:
    st.session_state.supports = {} # node -> type: 'none','rodillo','pinned','fixed'
if "last_solution" not in st.session_state:
    st.session_state.last_solution = None

# ---------------- helpers ----------------
def add_node(x, y):
    st.session_state.nodes.append({'x': float(x), 'y': float(y)})

def add_bar(n1, n2, A=0.001, E=210e9):
    n = len(st.session_state.nodes)
    if n1<0 or n1>=n or n2<0 or n2>=n or n1==n2:
        st.warning("Indices de barra inválidos.")
        return
    st.session_state.bars.append({'n1':int(n1),'n2':int(n2),'A':float(A),'E':float(E)})

def remove_node(idx):
    if idx<0 or idx>=len(st.session_state.nodes): return
    # remove bars connected and reindex
    new_bars=[]
    for b in st.session_state.bars:
        if b['n1']==idx or b['n2']==idx: continue
        b2 = dict(b)
        if b2['n1']>idx: b2['n1']-=1
        if b2['n2']>idx: b2['n2']-=1
        new_bars.append(b2)
    st.session_state.bars = new_bars
    st.session_state.nodes.pop(idx)
    st.session_state.loads = { (k if k<idx else k-1):v for k,v in st.session_state.loads.items() if k!=idx }
    st.session_state.supports = { (k if k<idx else k-1):v for k,v in st.session_state.supports.items() if k!=idx }

def remove_bar(idx):
    if idx<0 or idx>=len(st.session_state.bars): return
    st.session_state.bars.pop(idx)

# ---------------- solver ----------------
def assemble_and_solve():
    nodes = st.session_state.nodes
    bars = st.session_state.bars
    loads = st.session_state.loads
    supports = st.session_state.supports

    n = len(nodes)
    if n==0: raise ValueError("No hay nodos.")
    if len(bars)==0: raise ValueError("No hay barras.")

    dof = 2*n
    K = np.zeros((dof,dof))
    F = np.zeros(dof)

    # loads
    for ni,(Fx,Fy) in loads.items():
        F[2*ni] = Fx
        F[2*ni+1] = Fy

    # assemble
    for b in bars:
        n1=b['n1']; n2=b['n2']; A=b['A']; E=b['E']
        x1,y1 = nodes[n1]['x'], nodes[n1]['y']
        x2,y2 = nodes[n2]['x'], nodes[n2]['y']
        L = math.hypot(x2-x1, y2-y1)
        if L==0:
            raise ValueError(f"Barra con longitud 0 entre N{n1}-N{n2}")
        c=(x2-x1)/L; s=(y2-y1)/L
        k = E*A/L
        k_local = k * np.array([[ c*c, c*s, -c*c, -c*s],
                                [ c*s, s*s, -c*s, -s*s],
                                [-c*c,-c*s,  c*c,  c*s],
                                [-c*s,-s*s,  c*s,  s*s]])
        dof_map = [2*n1,2*n1+1,2*n2,2*n2+1]
        for i in range(4):
            for j in range(4):
                K[dof_map[i],dof_map[j]] += k_local[i,j]

    # supports -> interpret types:
    fixed_dofs=[]
    for ni, typ in supports.items():
        if typ == 'rodillo':
            # roller vertical (only Uy fixed) -> typical roller: fix Uy
            fixed_dofs.append(2*ni+1)
        elif typ == 'pinned':
            fixed_dofs.append(2*ni); fixed_dofs.append(2*ni+1)
        elif typ == 'fixed':
            # for truss, we cannot compute moment; fixed acts like pinned for translation
            fixed_dofs.append(2*ni); fixed_dofs.append(2*ni+1)
        # 'none' -> nothing

    free = [i for i in range(2*n) if i not in fixed_dofs]
    if len(free)==0: raise ValueError("No hay grados libres para resolver (todo fijo).")
    # check that Kff is not singular
    Kff = K[np.ix_(free,free)]
    Ff = F[free]
    # try solve
    U = np.zeros(2*n)
    try:
        Uf = np.linalg.solve(Kff, Ff)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError("Singular matrix: la estructura está inestable o mal apoyada.")
    U[free] = Uf
    reactions = K.dot(U) - F

    # axial forces
    bar_results=[]
    for idx,b in enumerate(bars):
        n1=b['n1']; n2=b['n2']; A=b['A']; E=b['E']
        x1,y1 = nodes[n1]['x'], nodes[n1]['y']
        x2,y2 = nodes[n2]['x'], nodes[n2]['y']
        L = math.hypot(x2-x1, y2-y1); c=(x2-x1)/L; s=(y2-y1)/L
        u_local = np.array([U[2*n1], U[2*n1+1], U[2*n2], U[2*n2+1]])
        delta = (-c*u_local[0] - s*u_local[1] + c*u_local[2] + s*u_local[3])
        N = E*A/L * delta
        bar_results.append({'bar':idx,'n1':n1,'n2':n2,'L_m':L,'axial_N':float(N)})
    disp = [{'node':i,'Ux_m':float(U[2*i]),'Uy_m':float(U[2*i+1])} for i in range(n)]
    reac = [{'node':i,'Rx_N':float(reactions[2*i]),'Ry_N':float(reactions[2*i+1])} for i in range(n)]
    return {'U':disp,'R':reac,'bars':bar_results}

# ---------------- plotting utilities ----------------
def auto_limits(nodes, margin=0.1):
    xs = [n['x'] for n in nodes]; ys=[n['y'] for n in nodes]
    if not xs:
        return (-1,1,-1,1)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = xmax - xmin; dy = ymax - ymin
    if dx == 0: dx = 1.0
    if dy == 0: dy = 1.0
    # add margin proportionally
    xmin -= max(dx*margin, 0.1)
    xmax += max(dx*margin, 0.1)
    ymin -= max(dy*margin, 0.1)
    ymax += max(dy*margin, 0.1)
    # ensure square aspect by expanding the smaller side
    box_w = xmax - xmin; box_h = ymax - ymin
    if box_w > box_h:
        extra = (box_w - box_h)/2
        ymin -= extra; ymax += extra
    else:
        extra = (box_h - box_w)/2
        xmin -= extra; xmax += extra
    return xmin, xmax, ymin, ymax

def draw_support_symbol(ax, x, y, typ, size=0.15):
    # typ: 'rodillo' (roller vertical), 'pinned', 'fixed'
    if typ == 'rodillo':
        # draw small triangle base and circle roller
        ax.add_patch(plt.Polygon([[x-0.08*size, y-0.02*size],[x+0.08*size,y-0.02*size],[x,y-0.15*size]], color='green'))
        ax.add_patch(plt.Circle((x-0.04*size,y-0.02*size - 0.02*size), 0.02*size, color='green'))
        ax.add_patch(plt.Circle((x+0.04*size,y-0.02*size - 0.02*size), 0.02*size, color='green'))
    elif typ == 'pinned':
        ax.add_patch(plt.Polygon([[x-0.1*size,y-0.02*size],[x+0.1*size,y-0.02*size],[x,y-0.15*size]], color='green'))
        ax.plot([x-0.06*size,x+0.06*size],[y-0.18*size,y-0.18*size], color='black', linewidth=0.8)
    elif typ == 'fixed':
        # triangle + solid block to indicate fixed (and small arc for moment)
        ax.add_patch(plt.Rectangle((x-0.12*size, y-0.15*size-0.02*size), 0.24*size, 0.03*size, color='green'))
        ax.add_patch(plt.Polygon([[x-0.06*size,y-0.02*size],[x+0.06*size,y-0.02*size],[x,y-0.12*size]], color='green'))
        # draw a small curved arc to indicate moment (visual only)
        theta = np.linspace(-math.pi/2, 0, 20)
        arc_x = x + 0.08*size*np.cos(theta)
        arc_y = y - 0.05*size + 0.08*size*np.sin(theta)
        ax.plot(arc_x, arc_y, color='orange')

def plot_model(nodes, bars, loads, supports, show_labels=True):
    fig, ax = plt.subplots(figsize=(6,6))
    if not nodes:
        ax.text(0.5,0.5,"No hay nodos. Usa el panel lateral para añadir nodos.", ha='center')
        return fig
    xmin,xmax,ymin,ymax = auto_limits(nodes, margin=0.12)
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    # draw bars
    for i,b in enumerate(bars):
        n1 = nodes[b['n1']]; n2 = nodes[b['n2']]
        ax.plot([n1['x'], n2['x']], [n1['y'], n2['y']], '-o', color='tab:blue', linewidth=2)
        mx, my = (n1['x']+n2['x'])/2, (n1['y']+n2['y'])/2
        if show_labels:
            ax.text(mx, my, f"B{i}", color='blue', fontsize=9)
    # draw nodes
    xs = [n['x'] for n in nodes]; ys=[n['y'] for n in nodes]
    ax.scatter(xs, ys, c='k', zorder=5)
    for i,n in enumerate(nodes):
        ax.text(n['x']+ (xmax-xmin)*0.01, n['y']+ (ymax-ymin)*0.01, f"N{i}")
    # draw supports
    for i,typ in supports.items():
        n = nodes[i]
        draw_support_symbol(ax, n['x'], n['y'], typ, size=(xmax-xmin)*0.03)
    # draw loads: scale arrows so they are visible
    if loads:
        # compute representative scale
        maxF = max(math.hypot(Fx,Fy) for Fx,Fy in loads.values())
        if maxF==0: maxF = 1.0
        # desired arrow length (in data units) for the max force
        desired_px = 0.15 * (xmax - xmin)  # arrow length for maxF
        scale = maxF / desired_px
        min_vis = 0.05 * (xmax - xmin)  # minimum visible length
        for i,(Fx,Fy) in loads.items():
            n = nodes[i]
            # compute length using scale but ensure min_vis
            L = math.hypot(Fx,Fy)
            if L==0:
                # draw tiny marker
                ax.plot(n['x'], n['y'], marker=(3,0,0), color='red')
                continue
            ax_len = max(L/scale, min_vis)
            ux = Fx / L; uy = Fy / L
            ax.annotate('', xy=(n['x']+ux*ax_len, n['y']+uy*ax_len), xytext=(n['x'], n['y']),
                        arrowprops=dict(arrowstyle="->", color='red', lw=2),
                        )
            ax.text(n['x']+ux*ax_len*1.05, n['y']+uy*ax_len*1.05, f"{int(Fx)},{int(Fy)} N", color='red')
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    return fig

# ---------------- UI ----------------
st.title("🔩 Truss Solver - visual mejorada")
st.markdown("Interfaz para definir nodos/barras, apoyos y cargas. Flechas y escala automáticas.")

# sidebar controls
with st.sidebar:
    st.header("Modelo / Edición")
    if st.button("Ejemplo: triángulo estable"):
        st.session_state.nodes=[]; st.session_state.bars=[]; st.session_state.loads={}; st.session_state.supports={}
        add_node(0.0,0.0); add_node(4.0,0.0); add_node(2.0,2.5)
        add_bar(0,1,A=0.005); add_bar(1,2,A=0.005); add_bar(2,0,A=0.005)
        st.session_state.supports[0]='pinned'; st.session_state.supports[1]='rodillo'
        st.session_state.loads[2]=(0.0,-1000.0)
        st.success("Ejemplo cargado. Pulsa 'Resolver armadura' en la sección principal.")
    st.markdown("---")
    st.subheader("Agregar nodo (manual)")
    nx = st.number_input("X (m)", value=0.0, step=0.1, key="nx")
    ny = st.number_input("Y (m)", value=0.0, step=0.1, key="ny")
    if st.button("➕ Agregar nodo"):
        add_node(nx, ny)
        st.success("Nodo agregado.")
    st.markdown("---")
    st.subheader("Agregar barra")
    if len(st.session_state.nodes)>=2:
        options = list(range(len(st.session_state.nodes)))
        n1_sel = st.selectbox("Nodo i", options=options, index=0, key="seln1")
        n2_sel = st.selectbox("Nodo j", options=options, index=1, key="seln2")
        Aval = st.number_input("Área A (m²)", value=0.001, format="%.6f")
        Eval = st.number_input("E (Pa)", value=210e9, format="%.1f")
        if st.button("➕ Agregar barra (i-j)"):
            add_bar(n1_sel, n2_sel, A=Aval, E=Eval)
            st.success("Barra agregada.")
    else:
        st.info("Agrega al menos 2 nodos para crear una barra.")
    st.markdown("---")
    st.subheader("Editar / Borrar")
    if st.session_state.nodes:
        rem_n = st.number_input("Eliminar nodo index", min_value=0, max_value=max(0,len(st.session_state.nodes)-1), value=0, key="remn")
        if st.button("🗑️ Eliminar nodo"):
            remove_node(rem_n); st.success("Nodo eliminado.")
    if st.session_state.bars:
        rem_b = st.number_input("Eliminar barra index", min_value=0, max_value=max(0,len(st.session_state.bars)-1), value=0, key="remb")
        if st.button("🗑️ Eliminar barra"):
            remove_bar(rem_b); st.success("Barra eliminada.")
    st.markdown("---")
    st.subheader("Apoyos & Cargas (por nodo)")
    if st.session_state.nodes:
        sel_node = st.selectbox("Seleccionar nodo index", options=list(range(len(st.session_state.nodes))), key="selnode_support")
        type_opt = st.selectbox("Tipo de apoyo", options=['none','rodillo','pinned','fixed'], index=0, help="rodillo=solo Uy; pinned=Ux+Uy; fixed=Ux+Uy (momento no calculado para truss)")
        if st.button("Aplicar apoyo"):
            if type_opt=='none':
                if sel_node in st.session_state.supports: st.session_state.supports.pop(sel_node)
            else:
                st.session_state.supports[sel_node]=type_opt
            st.success("Apoyo actualizado.")
        Fx = st.number_input("Px (N) para nodo seleccionado", value=st.session_state.loads.get(sel_node,(0.0,0.0))[0], key="Fx")
        Fy = st.number_input("Py (N) para nodo seleccionado", value=st.session_state.loads.get(sel_node,(0.0,0.0))[1], key="Fy")
        if st.button("Aplicar carga"):
            if abs(Fx)>0 or abs(Fy)>0:
                st.session_state.loads[sel_node]=(float(Fx),float(Fy))
            else:
                if sel_node in st.session_state.loads: st.session_state.loads.pop(sel_node)
            st.success("Carga aplicada.")

# main area
col1, col2 = st.columns([1,1.2])

with col1:
    st.subheader("Lista de nodos y barras")
    if st.session_state.nodes:
        st.table(pd.DataFrame(st.session_state.nodes).reset_index().rename(columns={'index':'node'}))
    else:
        st.info("No hay nodos definidos.")
    st.markdown("---")
    if st.session_state.bars:
        st.table(pd.DataFrame(st.session_state.bars).reset_index().rename(columns={'index':'bar'}))
    else:
        st.info("No hay barras definidas.")

with col2:
    st.subheader("Visualización (escala automática)")
    fig = plot_model(st.session_state.nodes, st.session_state.bars, st.session_state.loads, st.session_state.supports)
    st.pyplot(fig)

st.markdown("---")
st.subheader("Resolver & Exportar")
colA, colB = st.columns(2)
with colA:
    if st.button("▶️ Resolver armadura"):
        try:
            sol = assemble_and_solve()
            st.session_state.last_solution = sol
            st.success("Cálculo completado correctamente ✅")
        except Exception as e:
            st.error(f"Error al resolver: {e}")

with colB:
    if st.button("📥 Descargar resultados (CSV)"):
        sol = st.session_state.last_solution
        if sol is None:
            st.warning("Primero resuelve la estructura.")
        else:
            df_nodes = pd.DataFrame(sol['U']).merge(pd.DataFrame(sol['R']), on='node')
            df_bars = pd.DataFrame(sol['bars'])
            # prepare CSVs
            csv1 = df_nodes.to_csv(index=False).encode('utf-8')
            csv2 = df_bars.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar nodos (CSV)", csv1, "nodes_results.csv", "text/csv")
            st.download_button("Descargar barras (CSV)", csv2, "bars_results.csv", "text/csv")

# show results
if st.session_state.last_solution:
    st.markdown("---")
    st.subheader("Resultados recientes")
    sol = st.session_state.last_solution
    df_nodes = pd.DataFrame(sol['U']).merge(pd.DataFrame(sol['R']), on='node')
    df_bars = pd.DataFrame(sol['bars'])
    st.write("Desplazamientos y reacciones por nodo")
    st.dataframe(df_nodes.style.format({"Ux_m":"{:.6e}","Uy_m":"{:.6e}","Rx_N":"{:.3f}","Ry_N":"{:.3f}"}))
    st.write("Fuerzas axiales por barra")
    st.dataframe(df_bars.style.format({"L_m":"{:.4f}","axial_N":"{:.3f}"}))

# instructions
with st.expander("🛈 Instrucciones (breve)"):
    st.markdown("""
    **Cómo usar**
    1. Añade nodos (coordenadas en m).
    2. Añade barras entre índices de nodos.
    3. Selecciona un nodo y aplica apoyo (rodillo/pinned/fixed) y/o cargas Px,Py.
    4. Pulsa *Resolver armadura*.
    5. Descarga resultados CSV.

    **Notas**
    - El empotramiento se muestra gráficamente pero el solver es para *armaduras* (solo fuerzas axiales). Si quieres un pórtico con momentos dime y lo implemento.
    - Flechas y escala se ajustan automáticamente para que todo sea visible.
    """)

st.markdown("---")
st.caption("Si quieres que el empotramiento incluya momentos (pórtico 2D), puedo ampliarlo (con elementos de viga).")
