# plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
import scipy
from matplotlib.patches import Circle
from windrose import WindroseAxes

def plot_plan_view(C1, x, y, title, wind_dir=None, wind_speed=None, puff_list=None, stability_class=1, n_show=10):
    fig, ax_main = plt.subplots(figsize=(8, 6))

    # Integra la concentrazione nel tempo lungo l'asse 2 (T)
    data = np.trapz(C1, axis=2) * 1e6  # µg/m³ #type:ignore
    vmin = np.percentile(data, 5)
    vmax = np.percentile(data, 95)

    # Plot della concentrazione integrata
    pcm = ax_main.pcolor(x, y, data, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax_main, label=r'$\mu g \cdot m^{-3}$')
    ax_main.set_xlabel('x (m)')
    ax_main.set_ylabel('y (m)')
    ax_main.set_title(title)
    ax_main.axis('equal')

    if wind_dir is not None and wind_speed is not None:
        inset_pos = [0.65, 0.65, 0.3, 0.3]  # left, bottom, width, height in figure coords
        ax_inset = WindroseAxes(fig, inset_pos)
        fig.add_axes(ax_inset)

        # Plot rosa dei venti con direzioni e velocità
        wind_dir = np.array(wind_dir) % 360
        wind_speed = np.full_like(wind_dir, fill_value=wind_speed, dtype=float)
        ax_inset.bar(wind_dir, wind_speed, normed=True, opening=0.8, edgecolor='white')
        ax_inset.set_legend(loc='lower right', title='Wind speed (m/s)')
        ax_inset.set_title("Rosa dei venti")

    # Plot puff sopra la plan view
    if puff_list is not None and len(puff_list) > 0:
        # Parametri σ_y empirici per classi A-F (Pasquill-Gifford)
        a_vals = [0.22, 0.16, 0.11, 0.08, 0.06, 0.04]
        b_vals = [0.90, 0.88, 0.86, 0.83, 0.80, 0.78]
        a = a_vals[stability_class - 1]
        b = b_vals[stability_class - 1]

        for i, puff in enumerate(puff_list):
            # if i % n_show != 0:
            #     continue  # salta puff intermedi

            distance = np.sqrt(puff.x**2 + puff.y**2)
            sigma_y = a * (distance + 1)**b  # evita 0^b

            circle = Circle((puff.x, puff.y), 2 * sigma_y, color='white', fill=False, lw=1.5)
            ax_main.add_patch(circle)
            ax_main.plot(puff.x, puff.y, 'wo', markersize=3)

        ax_main.legend(["Puff center (2σ)"], loc='lower right')

    plt.tight_layout()
    plt.show()

def plot_surface_time(C1, times, x_idx, y_idx, stability, stab_label, wind_label):

    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 6))
    signal = 1e6 * np.squeeze(C1[y_idx, x_idx, :])
    ax1.plot(times, signal, label="Hourly mean")
    ax1.plot(times, smooth(signal, 24), 'r', label="Daily mean")
    ax1.set_ylabel('Mass loading ($m$ g m$^{-3}$)')
    ax1.set_title(stab_label + '\n' + wind_label)
    ax1.legend()

    ax2.plot(times, stability)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Stability')

    plt.tight_layout()
    plt.show()

def plot_height_slice(C1, y, z, stab_label, wind_label):
    plt.figure(figsize=(8, 6))
    data = np.mean(C1, axis=2) * 1e6
    plt.pcolor(y,z, data, cmap='jet')      
    plt.clim(0,1e2)
    plt.xlabel('y (metres)')
    plt.ylabel('z (metres)')
    plt.title(stab_label + '\n' + wind_label)
    cb1=plt.colorbar()
    cb1.set_label(r'$\mu$ g m$^{-3}$')
    plt.show()

def plot_surface_view_3d(C, x, y, z=None, times=None, 
                                   source=None, sensors=None, 
                                   t_index=None, z_index=None, 
                                   binary_map= None,
                                   title="Distribuzione di concentrazione (3D)"):

    """
    C       : ndarray (nx, ny, nt) oppure (nx, ny, nz, nt) se disponibile
    x, y    : coordinate spaziali (1D)
    z       : (opzionale) array z se 4D
    times   : (opzionale) array tempi
    source  : (x, y) posizione sorgente
    sensors : lista [(x1, y1), (x2, y2), ...]
    t_index : tempo da visualizzare (se None -> media su tempo)
    z_index : piano verticale da visualizzare (se None -> primo piano)
    """

    is_4d = C.ndim == 4

    if is_4d:
        if z_index is None:
            z_index = 0
        C_plane = C[:, :, z_index, :]  # estrai piano z
    else:
        C_plane = C

    if t_index is None:
        data = np.mean(C_plane, axis=2)
        time_str = "media temporale"
    else:
        data = C_plane[:, :, t_index]
        time_str = f"t={times[t_index]:.2f}" if times is not None else f"t_index={t_index}"

    # µg/m³ conversion
    data = data * 1e6

    # Meshgrid
    X, Y = np.meshgrid(x, y)
    Z = data.T  # attenzione: trasposto per combaciare con meshgrid

    # Colormap contrastata
    vmin = np.percentile(Z, 5)
    vmax = np.percentile(Z, 95)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='jet', vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)

    # Overlay mappa edifici
    H, W = binary_map.shape
    y_map, x_map = np.meshgrid(np.arange(W), np.arange(H))
    buildings = np.where(binary_map == 0)
    ax.bar3d(buildings[1], buildings[0], 0, 1, 1, np.max(Z)*0.1, color='gray', alpha=0.5, shade=True, label="Edifici")

    # Overlay sorgente
    if source is not None:
        ax.scatter(source[0], source[1], np.max(Z)*1.1, color='cyan', marker='*', s=200, label='Sorgente')

    # Overlay sensori
    if sensors is not None:
        for sx, sy in sensors:
            ax.scatter(sx, sy, np.max(Z)*1.05, color='lime', marker='o', s=80)
        ax.scatter([], [], [], color='lime', marker='o', label='Sensori')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel(r'Concentrazione [$\mu g \cdot m^{-3}$]')

    ax.set_title(f"{title}\n({time_str}, z={z[z_index]:.2f} m)" if is_4d and z is not None else f"{title}\n({time_str})")

    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label=r'$\mu g \cdot m^{-3}$')
    ax.legend()
    plt.tight_layout()
    plt.show()

def animate_plan_view(C1, x, y, binary_map=None, sensor_locs=None, interval=200, save_path=None):
    """
    Anima la dispersione temporale su mappa planare.
    
    Parametri:
    - C1: array (Y, X, T)
    - x, y: meshgrid (2D) delle coordinate in metri
    - binary_map: (Y, X) - 1 = suolo libero, 0 = edificio
    - sensor_locs: lista di tuple (x, y) in metri
    - interval: tempo tra i frame in ms
    - save_path: se fornito, salva la gif (es. 'dispersion.gif')
    """
    assert C1.ndim == 3, "C1 deve avere shape (Y, X, T)"
    Y, X, T = C1.shape

    # Conversione a microgrammi/m^3
    C1_micro = C1 * 1e6

    vmin = np.percentile(C1_micro, 5)
    vmax = np.percentile(C1_micro, 95)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('jet')

    # Primo frame
    img = ax.pcolormesh(x, y, C1_micro[:, :, 0], cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    cb = fig.colorbar(img, ax=ax)
    cb.set_label(r'$\mu g \cdot m^{-3}$')

    # Overlay edifici
    if binary_map is not None:
        buildings_overlay = ax.imshow((binary_map == 0), extent=(x.min(), x.max(), y.min(), y.max()),
                                      origin='lower', cmap='Greys', alpha=0.3)

    # Overlay sensori
    if sensor_locs is not None:
        sensor_scatter = ax.scatter(*zip(*sensor_locs), marker='^', c='black', s=80, label='Sensori')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    title = ax.set_title("Dispersione al tempo t = 0")

    def update(t):
        img.set_array(C1_micro[:, :, t].ravel())
        title.set_text(f"Dispersione al tempo t = {t}")
        return img, title

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=False)

    plt.tight_layout()

    if save_path:
        ani.save(save_path, writer='pillow', fps=1000//interval)
        print(f"✅ Animazione salvata in: {save_path}")
    else:
        plt.show()

def plot_puff_on_map(C1, x_grid, y_grid, center_lat, center_lon, timestep=-1, threshold=0.00, cutoff_norm=0.10, zoom_start=13, sensor_locs=None):
    deg_per_m = 1 / 111320

    lat_grid = center_lat + y_grid * deg_per_m
    lon_grid = center_lon + x_grid * deg_per_m / np.cos(np.deg2rad(center_lat))

    C = C1[:, :, timestep] if timestep >= 0 else np.mean(C1, axis=2)
    C_max = np.max(C)
    print(f"Max concentrazione: {C_max}")
    if C_max == 0:
        raise ValueError("Tutte le concentrazioni sono nulle")

    C_norm = C / C_max

    print(f"Valori normalizzati: min {np.min(C_norm)}, max {np.max(C_norm)}")

    points = []
    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):
            if C[i, j] > threshold and C_norm[i, j] > cutoff_norm:
                points.append([lat_grid[i, j], lon_grid[i, j], C_norm[i, j]])
    print(f"Punti selezionati: {len(points)}")

    if not points:
        raise ValueError("Nessuna concentrazione supera la soglia impostata")

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="cartodbpositron")
    HeatMap(points, radius=10, blur=5, max_zoom=1).add_to(m)

    folium.CircleMarker(
        location=[center_lat, center_lon],
        radius=7,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.9,
        popup='Punto di origine'
    ).add_to(m)

    legend_html = '''
     <div style="
     position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 90px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white;
     opacity: 0.8;
     padding: 10px;
     ">
     <b>Legenda concentrazione</b><br>
     <i style="background: linear-gradient(to right, blue, red); 
         display:inline-block; width: 100px; height: 10px;"></i><br>
     <small>Blu: basso</small><br>
     <small>Rosso: alto</small>
     </div>
     '''
    m.get_root().add_child(folium.Element(legend_html))
    if sensor_locs:
        m = plot_sensors_on_map(sensor_locs, m)
    return m

def meters_to_latlon(x, y, origin_lat, origin_lon):
    R = 6378137  # raggio terrestre medio (m)
    dLat = y / R
    dLon = x / (R * np.cos(np.pi * origin_lat / 180))
    lat = origin_lat + dLat * (180 / np.pi)
    lon = origin_lon + dLon * (180 / np.pi)
    return lat, lon

def folium_map_plot(sensor_coords_geo, source_geo, map_center=None, zoom_start=14):
    if map_center is None:
        map_center = np.mean(sensor_coords_geo, axis=0)

    m = folium.Map(location=map_center, zoom_start=zoom_start)

    # Sensori
    for i, (lat, lon) in enumerate(sensor_coords_geo):
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="red", icon="info-sign"),
            popup=f"Sensore {i+1}"
        ).add_to(m)

    # Sorgente stimata
    folium.Marker(
        location=source_geo,
        icon=folium.Icon(color="green", icon="star"),
        popup="Sorgente stimata"
    ).add_to(m)

    return m

def plot_sensors_on_map(sensor_positions, mappa):
    import folium
    for pos in sensor_positions:
        folium.Marker(location=pos, popup="Sensore", icon=folium.Icon(color='blue')).add_to(mappa)
    for i, pos in enumerate(sensor_positions):
        folium.Marker(location=pos, popup=f"Sensore {i+1}", icon=folium.Icon(color='blue')).add_to(mappa)
    return mappa

def plot_concentration_with_sensors(C, x, y, sensors, source, times, time_index=0, title=""):
    """
    C: array (X, Y, T)
    x, y: assi griglia
    sensors: lista di tuple (x, y)
    source: tuple (x_src, y_src)
    """
    conc_slice = C[:, :, time_index]

    extent = (x.min(), x.max(), y.min(), y.max())
    fig, ax = plt.subplots(figsize=(8, 6))
    im=ax.imshow(conc_slice.T, origin='lower', extent=extent, cmap='viridis')

    plt.colorbar(im, ax=ax, label="Concentrazione (a.u.)")

    # Sensori
    sensors_x, sensors_y = zip(*sensors)
    ax.scatter(sensors_x, sensors_y, c="red", marker="o", s=50, label="Sensori")

    # Sorgente
    ax.scatter(source[0], source[1], c="black", marker="*", s=100, label="Sorgente")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title or f"Concentrazione al tempo t={round(times[time_index],2)}")
    ax.legend()
    plt.tight_layout()
    plt.show()

