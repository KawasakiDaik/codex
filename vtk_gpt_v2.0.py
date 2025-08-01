#!/usr/bin/env python3
"""
vtk_visualizer_no_mayavi.py

Mayavi に頼らず、純粋な VTK (+PySide6) だけで
点群／等値面／ボリュームレンダリングを切り替えられるスタンドアロン可視化ツール。

------------------------------
使い方例
------------------------------
1) ボリュームレンダリングを GUI 表示
   python vtk_visualizer_no_mayavi.py /path/to/data_dir --plot-type volume --gui

2) 等値面 (iso-surface) を PNG に保存（オフスクリーン）
   python vtk_visualizer_no_mayavi.py /path/to/data_dir \
          --plot-type iso --iso-value 0.1 --output iso.png

3) 点群表示（GUI 自動起動）
   python vtk_visualizer_no_mayavi.py /path/to/data_dir --plot-type points

依存:
  • numpy
  • vtk (>=9.2 推奨)
  • PySide6  ※--gui オプション使用時のみ
"""

import argparse
import re
from pathlib import Path

import numpy as np
import vtk
from matplotlib import cm
from vtk.util import numpy_support as vtknp


# ----------------------------------------------------------------------
# データ読み込み
# ----------------------------------------------------------------------
def discover_files(folder: Path):
    """フォルダ内の連番ファイル (1,2,3,...) を昇順で返す"""
    patt = re.compile(r"^\d+$")
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and patt.match(p.name)],
        key=lambda p: int(p.name),
    )


def parse_slice_file(txt_path: Path):
    """
    Parse one FDTD slice file (N.txt) that stores the electric‑field
    distribution in the xy‑plane at a fixed z position.

    Returns
    -------
    field : np.ndarray, shape (ny, nx)
        Real‑part electric‑field values.  The first index corresponds to y,
        the second to x.
    x : np.ndarray, shape (nx,)
        x‑coordinate vector [m].
    y : np.ndarray, shape (ny,)
        y‑coordinate vector [m].
    z : float
        z‑coordinate value [m] extracted from the header.
    """
    with open(txt_path, "r") as f:
        lines = [ln.rstrip() for ln in f]

    # --- z position ----------------------------------------------------
    import re

    z_match = re.search(r"z=([+-]?[0-9.eE+-]+)\(m\)", lines[0])
    if not z_match:
        raise ValueError(f"Cannot parse z value in header of {txt_path}")
    z_pos = float(z_match.group(1))

    # --- locate sections ----------------------------------------------
    x_hdr = next(i for i, ln in enumerate(lines) if ln.lower().startswith("x("))
    # collect x values until blank line
    x_vals = []
    for ln in lines[x_hdr + 1 :]:
        if ln.strip() == "":
            break
        x_vals.append(float(ln))
    nx = len(x_vals)

    y_start = x_hdr + 1 + nx + 1  # skip blank line
    assert lines[y_start].lower().startswith("y(")
    y_vals = []
    for ln in lines[y_start + 1 :]:
        if ln.strip() == "":
            break
        y_vals.append(float(ln))
    ny = len(y_vals)

    # after y list, one blank, then a header line, then the data matrix
    # find first blank following y section
    idx = y_start + 1 + ny
    while idx < len(lines) and lines[idx].strip() != "":
        idx += 1
    # skip blank and repeated header
    data_start = idx + 2

    matrix_lines = lines[data_start : data_start + nx]
    if len(matrix_lines) != nx:
        raise ValueError(
            f"{txt_path}: expected {nx} matrix rows, found {len(matrix_lines)}"
        )

    matrix = np.array([[float(v) for v in ln.split()] for ln in matrix_lines])
    if matrix.shape != (nx, ny):
        raise ValueError(
            f"{txt_path}: expected matrix shape ({nx},{ny}), got {matrix.shape}"
        )

    # transpose so axis0 = y, axis1 = x  →  shape (ny, nx)
    field = matrix.T
    return field, np.asarray(x_vals), np.asarray(y_vals), z_pos


def build_volume(dir_path: Path):
    """
    Assemble all slice files in *dir_path* into a 3‑D numpy volume.

    Returns
    -------
    volume : np.ndarray, shape (nz, ny, nx)
    spacing : tuple(float, float, float)  (dx, dy, dz) in metres
    """
    files = discover_files(dir_path)
    if not files:
        raise RuntimeError(f"No numeric slice files in {dir_path}")

    slices = []
    z_positions = []
    for f in files:
        field, x, y, z = parse_slice_file(f)
        slices.append(field)
        z_positions.append(z)

    # sort by z (ascending)
    order = np.argsort(z_positions)
    volume = np.stack([slices[i] for i in order], axis=0)  # (nz, ny, nx)
    z_sorted = np.asarray(z_positions)[order]

    # approximate spacing
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))
    dz = np.mean(np.diff(z_sorted)) if len(z_sorted) > 1 else 1.0

    spacing = (dx, dy, dz)
    return volume, spacing


# ----------------------------------------------------------------------
# NumPy → VTK 変換
# ----------------------------------------------------------------------
def numpy_to_vtk_image(volume, spacing):
    nz, ny, nx = volume.shape
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(*spacing)

    flat = volume.ravel(order="C")
    arr = vtk.vtkFloatArray()
    arr.SetNumberOfValues(flat.size)
    for i, v in enumerate(flat):
        arr.SetValue(i, float(v))
    img.GetPointData().SetScalars(arr)
    return img


# ----------------------------------------------------------------------
# アクター生成
# ----------------------------------------------------------------------
def make_point_cloud(img, point_size=3, threshold=0.0, max_points=20000,
                     use_absolute=False, colormap='jet'):
    """
    Build a vtkActor for point‑cloud rendering with optional thresholding,
    down‑sampling and colormap selection.
    """
    # Convert vtkImageData to numpy for filtering
    dims = img.GetDimensions()
    scalars = vtknp.vtk_to_numpy(img.GetPointData().GetScalars())
    if use_absolute:
        scalars = np.abs(scalars)
    x_idx, y_idx, z_idx = np.indices(dims).reshape(3, -1)
    # spacing / origin
    spacing = img.GetSpacing()
    coords = np.vstack((x_idx * spacing[0],
                        y_idx * spacing[1],
                        z_idx * spacing[2])).T
    # Thresholding
    vmin, vmax = scalars.min(), scalars.max()
    cutoff = vmin + threshold * (vmax - vmin)
    sel = scalars >= cutoff
    if sel.sum() == 0:
        sel = scalars >= vmin  # fallback
    coords = coords[sel]
    values = scalars[sel]
    # Down‑sample
    if coords.shape[0] > max_points:
        idx = np.random.choice(coords.shape[0], max_points, replace=False)
        coords, values = coords[idx], values[idx]

    # Build PolyData
    pts = vtk.vtkPoints()
    vtk_arr = vtknp.numpy_to_vtk(coords.astype(np.float32))
    vtk_arr.SetNumberOfComponents(3)
    pts.SetData(vtk_arr)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.GetPointData().SetScalars(vtknp.numpy_to_vtk(values))

    lut = vtk.vtkLookupTable()
    cm_data = (np.array(cm.get_cmap(colormap)(np.linspace(0, 1, 256))) * 255).astype(np.uint8)
    lut.SetNumberOfTableValues(256)
    for i, rgba in enumerate(cm_data):
        lut.SetTableValue(i, *(rgba / 255))
    lut.Build()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    mapper.SetLookupTable(lut)
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarRange(vmin, vmax)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(point_size)
    # store config for GUI updates
    actor._pc_cache = dict(img=img, point_size=point_size, threshold=threshold,
                           max_points=max_points, use_absolute=use_absolute,
                           colormap=colormap)
    return actor


def _rebuild_point_actor(old_actor, **updates):
    # Make shallow copy of cached parameters and apply any updates
    cfg = old_actor._pc_cache.copy()
    cfg.update(updates)

    # Pull out the reference image so we never pass it twice
    img = cfg.pop('img', None)
    if img is None:                       # safety‑check
        raise RuntimeError("Cached actor is missing its 'img' reference")

    # Recreate the point‑cloud actor without duplicating the 'img' kwarg
    new_actor = make_point_cloud(img, **cfg)
    return new_actor


def make_iso_surface(img, iso_val):
    contour = vtk.vtkContourFilter()
    contour.SetInputData(img)
    contour.SetValue(0, iso_val)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # Attach contour filter for interactive GUI updates
    actor.contour_filter = contour  # keep a handle for interactive updates
    return actor


def make_volume_render(img):
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(img)

    vol_prop = vtk.vtkVolumeProperty()
    ctf = vtk.vtkColorTransferFunction()
    ctf.AddRGBPoint(0, 0.0, 0.0, 1.0)
    ctf.AddRGBPoint(1, 1.0, 0.0, 0.0)
    otf = vtk.vtkPiecewiseFunction()
    otf.AddPoint(0, 0.0)
    otf.AddPoint(1, 1.0)
    vol_prop.SetColor(ctf)
    vol_prop.SetScalarOpacity(otf)

    vol = vtk.vtkVolume()
    vol.SetMapper(mapper)
    vol.SetProperty(vol_prop)
    # Attach volume property and opacity transfer function for GUI
    vol.vol_prop = vol_prop         # for later GUI tweaks
    vol.opacity_tf = otf            # scalar‑opacity TF
    return vol


# ----------------------------------------------------------------------
# レンダラ & 出力
# ----------------------------------------------------------------------
def build_renderer(actor):
    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(0.1, 0.1, 0.2)
    ren.ResetCamera()               # ensure the scene is framed
    return ren


def render_offscreen(ren, outfile):
    win = vtk.vtkRenderWindow()
    win.AddRenderer(ren)
    win.SetOffScreenRendering(1)
    win.SetSize(800, 600)
    win.Render()

    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(win)
    w2if.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(outfile))
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()
    print(f"Saved PNG -> {outfile}")


def _render_gui_basic(ren):
    from PySide6 import QtWidgets
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    app = QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()

    frame = QtWidgets.QFrame()
    layout = QtWidgets.QVBoxLayout(frame)
    vtk_widget = QVTKRenderWindowInteractor(frame)
    layout.addWidget(vtk_widget)

    win.setCentralWidget(frame)
    win.resize(1024, 768)

    vtk_widget.GetRenderWindow().AddRenderer(ren)
    iren = vtk_widget.GetRenderWindow().GetInteractor()
    iren.Initialize()
    win.show()
    app.exec()


# New interactive GUI with sliders for iso/volume/points plus extended controls
def render_gui(ren, actor, plot_type, img):
    """
    Launch a Qt window with an embedded VTK render‑window *and*
    a simple control panel that exposes:
        • iso‑value slider  (plot_type == 'iso')
        • point‑size slider (plot_type == 'points')
        • opacity slider    (plot_type == 'volume')
        plus extended controls from legacy v10.0_E.py:
        • threshold slider
        • max points slider (points)
        • colormap combo box
        • use absolute checkbox
        • plot type radio buttons
    """
    from PySide6 import QtWidgets, QtCore
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    # ---- Qt boilerplate ------------------------------------------------
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    central = QtWidgets.QSplitter()
    win.setCentralWidget(central)

    # left: VTK widget
    vtk_widget = QVTKRenderWindowInteractor()
    vtk_widget.GetRenderWindow().AddRenderer(ren)
    central.addWidget(vtk_widget)

    # right: control panel
    panel = QtWidgets.QWidget()
    form = QtWidgets.QFormLayout(panel)
    central.addWidget(panel)
    central.setStretchFactor(0, 4)
    central.setStretchFactor(1, 1)

    # ---- slider helpers -----------------------------------------------
    def _add_slider(name, rng, init, cb):
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(*rng)
        slider.setValue(init)
        slider.valueChanged.connect(cb)
        form.addRow(QtWidgets.QLabel(name), slider)

    def _add_combo(name, choices, init, cb):
        combo = QtWidgets.QComboBox()
        combo.addItems(choices)
        combo.setCurrentText(init)
        combo.currentTextChanged.connect(cb)
        form.addRow(QtWidgets.QLabel(name), combo)

    def _add_check(name, init, cb):
        chk = QtWidgets.QCheckBox()
        chk.setChecked(init)
        chk.stateChanged.connect(lambda state: cb(bool(state)))
        form.addRow(QtWidgets.QLabel(name), chk)

    # ---- state storage ------------------------------------------------
    gui_state = {
        'point_size': 3,
        'threshold': 0.0,
        'max_points': 20000,
        'use_absolute': False,
        'colormap': 'jet',
        'iso_value': 0.0,
        'opacity': 100,
        'plot_type': plot_type
    }

    # ---- plot type switching -------------------------------------------
    def _switch_plot_type(new_type):
        nonlocal actor
        if new_type == gui_state['plot_type']:
            return
        gui_state['plot_type'] = new_type
        # Remove old actor
        ren.RemoveAllViewProps()
        if new_type == "points":
            actor_new = make_point_cloud(img,
                                         point_size=gui_state['point_size'],
                                         threshold=gui_state['threshold'],
                                         max_points=gui_state['max_points'],
                                         use_absolute=gui_state['use_absolute'],
                                         colormap=gui_state['colormap'])
        elif new_type == "iso":
            actor_new = make_iso_surface(img, gui_state['iso_value'])
        else:
            actor_new = make_volume_render(img)
            # If use_absolute affects volume opacity TF, update opacity here
            if gui_state['use_absolute']:
                # Adjust opacity TF to be stronger if use_absolute is True
                otf = actor_new.opacity_tf
                otf.RemoveAllPoints()
                otf.AddPoint(0, 0.0)
                otf.AddPoint(1, gui_state['opacity'] / 100)
        actor = actor_new
        ren.AddActor(actor)
        ren.ResetCamera()
        vtk_widget.GetRenderWindow().Render()

    # Plot type radio buttons
    plot_types = ['points', 'iso', 'volume']
    radio_group = QtWidgets.QButtonGroup()
    radio_layout = QtWidgets.QHBoxLayout()
    for pt in plot_types:
        rb = QtWidgets.QRadioButton(pt.capitalize())
        if pt == plot_type:
            rb.setChecked(True)
        radio_group.addButton(rb)
        radio_layout.addWidget(rb)
    form.addRow(QtWidgets.QLabel("Plot type"), radio_layout)

    def _on_plot_type_changed():
        selected = None
        for btn in radio_group.buttons():
            if btn.isChecked():
                selected = btn.text().lower()
                break
        if selected is not None:
            _switch_plot_type(selected)

    radio_group.buttonClicked.connect(_on_plot_type_changed)

    # ---- plot‑specific controls ---------------------------------------

    # Threshold slider (0-100)
    def _threshold_cb(val):
        nonlocal actor
        gui_state['threshold'] = val / 100
        if gui_state['plot_type'] == "points":
            ren.RemoveActor(actor)
            actor = _rebuild_point_actor(actor, threshold=gui_state['threshold'])
            ren.AddActor(actor)
        elif gui_state['plot_type'] == "iso" and hasattr(actor, "contour_filter"):
            cmin, cmax = actor.contour_filter.GetInput().GetScalarRange()
            iso_val = cmin + gui_state['threshold'] * (cmax - cmin)
            actor.contour_filter.SetValue(0, iso_val)
        vtk_widget.GetRenderWindow().Render()

    _add_slider("Threshold", (0, 100), int(gui_state['threshold']*100), _threshold_cb)

    # Max points slider (1000 - 100000)
    def _max_points_cb(val):
        nonlocal actor
        gui_state['max_points'] = val
        if gui_state['plot_type'] == "points":
            ren.RemoveActor(actor)
            actor = _rebuild_point_actor(actor, max_points=gui_state['max_points'])
            ren.AddActor(actor)
            vtk_widget.GetRenderWindow().Render()

    _add_slider("Max points", (1000, 100000), gui_state['max_points'], _max_points_cb)

    # Colormap combo box
    cmaps = ['jet','viridis','plasma','inferno','magma','Blues','Greens','Reds','Oranges','Purples','RdBu','coolwarm','bone','copper']
    def _colormap_cb(val):
        nonlocal actor
        gui_state['colormap'] = val
        if gui_state['plot_type'] == "points":
            ren.RemoveActor(actor)
            actor = _rebuild_point_actor(actor, colormap=gui_state['colormap'])
            ren.AddActor(actor)
            vtk_widget.GetRenderWindow().Render()

    _add_combo("Colormap", cmaps, gui_state['colormap'], _colormap_cb)

    # Use absolute checkbox
    def _use_abs_cb(flag):
        nonlocal actor
        gui_state['use_absolute'] = flag
        if gui_state['plot_type'] == "points":
            ren.RemoveActor(actor)
            actor = _rebuild_point_actor(actor, use_absolute=flag)
            ren.AddActor(actor)
        elif gui_state['plot_type'] == "volume" and hasattr(actor, "opacity_tf"):
            alpha = gui_state['opacity'] / 100
            tf = actor.opacity_tf
            tf.RemoveAllPoints()
            tf.AddPoint(0, 0.0)
            tf.AddPoint(1, alpha)
            # Could add more complex logic here if needed
        vtk_widget.GetRenderWindow().Render()

    _add_check("Use Absolute", gui_state['use_absolute'], _use_abs_cb)

    if plot_type == "iso" and hasattr(actor, "contour_filter"):
        # iso‑value slider (0‑1000 → real range)
        cmin, cmax = actor.contour_filter.GetInput().GetScalarRange()
        def _iso_cb(val):
            iso = cmin + val / 1000 * (cmax - cmin)
            gui_state['iso_value'] = iso
            actor.contour_filter.SetValue(0, iso)
            vtk_widget.GetRenderWindow().Render()
        init = int((actor.contour_filter.GetValue(0) - cmin) / (cmax - cmin) * 1000) if (cmax - cmin) != 0 else 0
        _add_slider("Iso‑threshold", (0, 1000), init, _iso_cb)

        # opacity slider for iso surface
        def _iso_opacity_cb(val):
            actor.GetProperty().SetOpacity(val / 100)
            vtk_widget.GetRenderWindow().Render()
        _add_slider("Opacity", (0, 100), 100, _iso_opacity_cb)

    elif plot_type == "points":
        # point size slider
        def _size_cb(val):
            nonlocal actor
            gui_state['point_size'] = max(val, 1)
            if gui_state['plot_type'] != "points":
                return
            prop = actor.GetProperty() if hasattr(actor, "GetProperty") else None
            if prop and hasattr(prop, "SetPointSize"):
                prop.SetPointSize(gui_state['point_size'])
                vtk_widget.GetRenderWindow().Render()
        _add_slider("Point size", (1, 20), int(actor.GetProperty().GetPointSize() or 3), _size_cb)

        # opacity slider
        def _pt_opacity_cb(val):
            prop = actor.GetProperty()
            if hasattr(prop, "SetOpacity"):
                prop.SetOpacity(val / 100)
                vtk_widget.GetRenderWindow().Render()
        _add_slider("Opacity", (0, 100), 100, _pt_opacity_cb)

    elif plot_type == "volume" and hasattr(actor, "opacity_tf"):
        # opacity gain slider
        def _vol_opacity_cb(val):
            gui_state['opacity'] = val
            if not hasattr(actor, "opacity_tf"):   # guard against non‑volume actor
                return
            alpha = val / 100
            tf = actor.opacity_tf
            tf.RemoveAllPoints()
            tf.AddPoint(0, 0.0)
            tf.AddPoint(1, alpha)
            vtk_widget.GetRenderWindow().Render()
        _add_slider("Opacity", (1, 100), 100, _vol_opacity_cb)

    # ---- finalise ------------------------------------------------------
    vtk_widget.Initialize()
    win.resize(1200, 800)
    win.show()
    app.exec()


# ----------------------------------------------------------------------
# エントリポイント
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="VTK-only 3D visualizer")
    ap.add_argument("data_dir", help="連番スライスが入ったディレクトリ")
    ap.add_argument("--plot-type", choices=["points", "iso", "volume"],
                    default="points")
    ap.add_argument("--iso-value", type=float, default=0.0,
                    help="等値面抽出しきい値 (--plot-type iso)")
    ap.add_argument("--gui", action="store_true",
                    help="Qt GUI を起動（省略時は --output がなければ GUI 起動）")
    ap.add_argument("--output", help="PNG 書き出しファイル名（指定時オフスクリーン）")
    args = ap.parse_args()

    gui_state = {
        'threshold': 0.0,
        'max_points': 20000,
        'use_absolute': False,
        'colormap': 'jet'
    }

    volume, spacing = build_volume(Path(args.data_dir))
    img = numpy_to_vtk_image(volume, spacing)

    if args.plot_type == "points":
        actor = make_point_cloud(img,
                                 point_size=3,
                                 threshold=gui_state['threshold'],
                                 max_points=gui_state['max_points'],
                                 use_absolute=gui_state['use_absolute'],
                                 colormap=gui_state['colormap'])
    elif args.plot_type == "iso":
        actor = make_iso_surface(img, args.iso_value)
    else:
        actor = make_volume_render(img)

    ren = build_renderer(actor)

    if args.output:
        render_offscreen(ren, args.output)
    if args.gui or not args.output:
        render_gui(ren, actor, args.plot_type, img)


if __name__ == "__main__":
    main()