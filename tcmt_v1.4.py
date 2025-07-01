"""
2モード光共振器反射スペクトル計算プログラム (パラメータスイープ機能付き)

使用方法:
1. 基本パラメータを base_params で設定
2. パラメータスイープを行う場合:
   - do_sweep = True に設定
   - sweep_param でスイープするパラメータ名を指定
   - sweep_configs でスイープ範囲を定義（start, end, step）
3. 通常の単一計算を行う場合:
   - do_sweep = False に設定

出力:
- スペクトルグラフ（反射率・位相）
- 数値表（コンソール出力）
- Excelファイル（output/xlsx_data または output/sweep_data）

パラメータスイープ可能な項目:
- gamma_abs_a/b: 吸収損失
- gamma_1_a/b: ポート1結合損失
- eta: ポート2結合比率
- g_r/g_i: モード間結合
- r: 反射係数
- lambda_a/b: 共振波長
- n: 屈折率
- l: 位相整合層厚さ
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import warnings

# openpyxlのインポート確認
try:
    import openpyxl
except ImportError:
    warnings.warn("openpyxlがインストールされていません。Excel保存機能が使用できません。")

# 物理定数
hbar = 1.054571817e-34  # J·s
eV_to_J = 1.602176634e-19  # J/eV
meV_to_J = 1.602176634e-22  # J/meV
meV_to_rad_s = meV_to_J / hbar  # meVをrad/sに変換

def calculate_eigenmode_properties(params, wavelength_eval=None):
    """
    有効ハミルトニアンから固有モードの特性を計算
    
    Parameters:
    -----------
    params : dict
        システムパラメータ
    wavelength_eval : float
        評価する波長 (nm)。Noneの場合は共振波長の平均値を使用
    
    Returns:
    --------
    dict : モード±の特性（波長、線幅、Q値など）
    """
    # パラメータの展開
    gamma_abs_a = params['gamma_abs_a']
    gamma_abs_b = params['gamma_abs_b']
    gamma_1_a = params['gamma_1_a']
    gamma_1_b = params['gamma_1_b']
    lambda_a = params['lambda_a']
    lambda_b = params['lambda_b']
    eta = params['eta']
    g_r = params['g_r']
    # g_i はポート損失から計算
    gamma_2_a = eta * gamma_1_a
    gamma_2_b = eta * gamma_1_b
    g_i = np.sqrt(gamma_1_a * gamma_1_b) + np.sqrt(gamma_2_a * gamma_2_b)
    r = params['r']
    t = params['t']
    n = params['n']
    l = params['l']
    l_g = params['l_g']
    phi_m = params['phi_m']
    
    # エネルギーの計算
    energy_a = 1240000.0 / lambda_a  # meV
    energy_b = 1240000.0 / lambda_b  # meV
    
    # 複素結合係数
    g = g_r - 1j * g_i
    g_star = g_r + 1j * g_i
    
    # 全減衰率
    gamma_a = (gamma_abs_a + gamma_1_a + gamma_2_a)
    gamma_b = (gamma_abs_b + gamma_1_b + gamma_2_b)

    # 放射率
    gamma_rad_a = gamma_1_a + gamma_2_a
    gamma_rad_b = gamma_1_b + gamma_2_b

    # ポート結合係数
    d_2a = np.sqrt(2 * gamma_2_a)
    d_2b = np.sqrt(2 * gamma_2_b)
    
    # 評価波長（デフォルトは共振波長の平均）
    if wavelength_eval is None:
        wavelength_eval = (lambda_a + lambda_b) / 2
    
    # 評価波長での位相計算
    wavelength_m = wavelength_eval * 1e-9  # m
    k = 2 * np.pi / wavelength_m
    phi = 2 * ( n * l + l_g ) * k + phi_m
    
    # ミラーの反射係数
    r_phi = np.exp(1j * phi) / (1 - r * np.exp(1j * phi))
    
    # 有効減衰率
    gamma_a_eff = gamma_a - d_2a ** 2 * r_phi 
    gamma_b_eff = gamma_b - d_2b ** 2 * r_phi 
    
    # 有効結合係数
    g_eff = g + r_phi * d_2a * d_2b
    
    # PDFの式に基づく計算
    # Δω = ω_a - ω_b, Δγ = γ_a^eff - γ_b^eff
    delta_omega = energy_a - energy_b
    delta_gamma = gamma_a_eff - gamma_b_eff
    
    # √[(Δω - iΔγ)² + 4|g^eff|²] の計算
    sqrt_term = np.sqrt((delta_omega - 1j * delta_gamma)**2 + 4 * np.abs(g_eff)**2)
    
    # 固有周波数 ω_±
    omega_plus = (energy_a + energy_b) / 2 + np.real(sqrt_term) / 2
    omega_minus = (energy_a + energy_b) / 2 - np.real(sqrt_term) / 2
    
    # 線幅 Γ_±
    gamma_plus = (np.real(gamma_a_eff) + np.real(gamma_b_eff)) / 2 - np.imag(sqrt_term) / 2
    gamma_minus = (np.real(gamma_a_eff) + np.real(gamma_b_eff)) / 2 + np.imag(sqrt_term) / 2
    
    # 実部を取る（線幅は正の値）
    gamma_plus = np.real(gamma_plus)
    gamma_minus = np.real(gamma_minus)
    
    # 波長の計算
    lambda_plus = 1240000.0 / omega_plus if omega_plus > 0 else np.inf
    lambda_minus = 1240000.0 / omega_minus if omega_minus > 0 else np.inf
    
    # Q値の計算
    Q_plus = omega_plus / gamma_plus if gamma_plus > 0 else 0
    Q_minus = omega_minus / gamma_minus if gamma_minus > 0 else 0
    
    return {
        'mode_plus': {
            'wavelength': lambda_plus,
            'energy': omega_plus,
            'linewidth': gamma_plus,
            'Q': Q_plus
        },
        'mode_minus': {
            'wavelength': lambda_minus,
            'energy': omega_minus,
            'linewidth': gamma_minus,
            'Q': Q_minus
        },
        'evaluation_wavelength': wavelength_eval
    }

def calculate_single_mode_reflection(params, mode='a'):
    """
    1モード光共振器の反射スペクトルを計算
    
    Parameters:
    -----------
    params : dict
        システムパラメータ
    mode : str
        'a' または 'b' でモードを指定
    
    Returns:
    --------
    S_r : array
        反射係数の配列
    mode_props : dict
        モードの特性
    """
    # モードに応じたパラメータ選択
    if mode == 'a':
        gamma_abs = params['gamma_abs_a']
        gamma_1 = params['gamma_1_a']
        lambda_mode = params['lambda_a']
    else:
        gamma_abs = params['gamma_abs_b']
        gamma_1 = params['gamma_1_b'] 
        lambda_mode = params['lambda_b']
    
    eta = params['eta']
    r = params['r']
    t = params['t']
    n = params['n']
    l = params['l']
    l_g = params['l_g']
    phi_m = params['phi_m']
    wavelength_range = params['wavelength_range']
    
    # gamma_2の計算
    gamma_2 = eta * gamma_1
    
    # エネルギーの計算
    energy_mode = 1240000.0 / lambda_mode  # meV
    
    # 全減衰率
    gamma_total = (gamma_abs + gamma_1 + gamma_2)
    
    # ポート結合係数
    d_1 = np.sqrt(2 * gamma_1)
    d_2 = np.sqrt(2 * gamma_2)
    
    # 反射スペクトルの計算
    S_r = []
    
    for wavelength_nm in wavelength_range:
        # 波長からエネルギーを計算
        energy = 1240000.0 / wavelength_nm  # meV
        
        # 波長をメートルに変換
        wavelength = wavelength_nm * 1e-9  # m
        k = 2 * np.pi / wavelength
        
        # 位相
        phi = 2 * ( n * l + l_g ) * k + phi_m
        
        # ミラーの反射係数
        r_phi = np.exp(1j * phi) / (1 - r * np.exp(1j * phi))
        
        # 有効減衰率
        gamma_eff = gamma_total - d_2 ** 2 * r_phi 
        
        # D係数
        D = d_1 + r_phi * t * d_2
        D_star = np.conj(D)
        
        # 離調
        delta_energy = energy - energy_mode
        
        # S_r の計算（PDFの式に従う）
        numerator = D * D_star
        denominator = 1j * delta_energy + gamma_eff
        S_r_value = -r + r_phi * t**2 + numerator / denominator
        
        S_r.append(S_r_value)
    
    S_r = np.array(S_r)

    # モード特性の計算
    Q = energy_mode / gamma_total
    idx_res = np.argmin(np.abs(wavelength_range - lambda_mode))
    reflectance = np.abs(S_r[idx_res])**2 if len(S_r) > 0 else 0.0

    mode_props = {
        'wavelength': lambda_mode,
        'energy': energy_mode,
        'linewidth': gamma_total,
        'Q': Q,
        'reflection_intensity': reflectance,
        'gamma_eff_at_resonance': np.real(gamma_eff)  # 共振での有効減衰率（近似値）
    }

    return S_r, mode_props

def calculate_reflection_spectrum(params):
    """
    2モード光共振器の反射スペクトルを計算
    """
    # パラメータの展開
    gamma_abs_a = params['gamma_abs_a']
    gamma_abs_b = params['gamma_abs_b']
    gamma_1_a = params['gamma_1_a']
    gamma_1_b = params['gamma_1_b']
    eta = params['eta']
    lambda_a = params['lambda_a']
    lambda_b = params['lambda_b']
    
    # gamma_2の計算
    gamma_2_a = eta * gamma_1_a
    gamma_2_b = eta * gamma_1_b
    
    # 波長からエネルギーを計算
    energy_a = 1240000.0 / lambda_a  # meV
    energy_b = 1240000.0 / lambda_b  # meV
    g_r = params['g_r']
    # g_i はポート損失から計算
    g_i = np.sqrt(gamma_1_a * gamma_1_b) + np.sqrt(gamma_2_a * gamma_2_b)
    r = params['r']
    t = params['t']
    n = params['n']
    l = params['l']
    wavelength_range = params['wavelength_range']
    
    # 複素結合係数
    g = g_r - 1j * g_i
    g_star = g_r + 1j * g_i
    
    # 全減衰率
    gamma_a = (gamma_abs_a + gamma_1_a + gamma_2_a)
    gamma_b = (gamma_abs_b + gamma_1_b + gamma_2_b)
    
    # ポート結合係数
    d_1a = np.sqrt(2 * gamma_1_a)
    d_1b = np.sqrt(2 * gamma_1_b)
    d_2a = np.sqrt(2 * gamma_2_a)
    d_2b = np.sqrt(2 * gamma_2_b)
    
    # 光速
    c = 3e8  # m/s
    h = 6.62607015e-34  # J·s
    
    # 反射スペクトルの計算
    S_r = []
    
    for wavelength_nm in wavelength_range:
        # 波長からエネルギーを計算
        energy = 1240000.0 / wavelength_nm  # meV
        
        # 波長をメートルに変換
        wavelength = wavelength_nm * 1e-9  # m
        k = 2 * np.pi / wavelength
        
        # 位相
        phi = 2 * n * l * k
        
        # ミラーの反射係数
        r_phi = np.exp(1j * phi) / (1 - r * np.exp(1j * phi))
        
        # 有効減衰率
        gamma_a_eff = gamma_a - d_2a ** 2 * r_phi 
        gamma_b_eff = gamma_b - d_2b ** 2 * r_phi 
        
        # 有効結合係数
        g_eff = g + r_phi * d_2a * d_2b
        g_star_eff = g_star + r_phi * d_2a * d_2b
        
        # D係数
        D_a = d_1a + r_phi * t * d_2a
        D_b = d_1b + r_phi * t * d_2b
        D_a_star = np.conj(D_a)
        D_b_star = np.conj(D_b)
        
        # 離調
        delta_energy_a = energy - energy_a
        delta_energy_b = energy - energy_b
        
        # M行列の要素
        M11 = 1j * delta_energy_a + gamma_a_eff
        M12 = g_eff
        M21 = g_star_eff
        M22 = 1j * delta_energy_b + gamma_b_eff
        
        # M行列の行列式
        det_M = M11 * M22 - M12 * M21
        
        # S_m の計算
        term1 = D_a_star * (1j * delta_energy_a + gamma_a_eff) + D_b_star * (1j * delta_energy_b + gamma_b_eff)
        term2 = D_a_star * D_b * g_eff + D_a * D_b_star * g_star_eff
        S_m = (term1 + term2) / det_M
        
        # 全反射係数
        S_r_value = (t**2 * r_phi - r) + S_m
        S_r.append(S_r_value)
    
    # 固有モードの特性を計算
    eigenmode_props = calculate_eigenmode_properties(params)
    
    # 固有モードでの反射強度を計算
    for mode_name in ['mode_plus', 'mode_minus']:
        mode_lambda = eigenmode_props[mode_name]['wavelength']
        if mode_lambda != np.inf and 600 <= mode_lambda <= 1200:
            # 最も近い波長インデックスを見つける
            idx = np.argmin(np.abs(wavelength_range - mode_lambda))
            eigenmode_props[mode_name]['reflection_intensity'] = np.abs(S_r[idx])**2
        else:
            eigenmode_props[mode_name]['reflection_intensity'] = 0.0
    
    return np.array(S_r), eigenmode_props

def parameter_sweep(base_params, sweep_param_name, sweep_values, mode='coupled'):
    """
    パラメータスイープを実行し、複数のスペクトルを計算
    
    Parameters:
    -----------
    base_params : dict
        基本パラメータ
    sweep_param_name : str
        スイープするパラメータ名
    sweep_values : array-like
        スイープする値のリスト
    mode : str
        'coupled' (2モード結合), 'single_a', 'single_b'
    
    Returns:
    --------
    results : list of dict
        各スイープ値での結果
    """
    results = []
    
    for value in sweep_values:
        # パラメータをコピーして更新
        params = base_params.copy()
        params[sweep_param_name] = value
        
        # tの再計算（rが変更された場合）
        if sweep_param_name == 'r':
            params['t'] = np.sqrt(1 - value**2)
        
        # gamma_2の再計算（etaまたはgamma_1が変更された場合）
        if sweep_param_name in ['eta', 'gamma_1_a', 'gamma_1_b']:
            params['gamma_2_a'] = params['eta'] * params['gamma_1_a']
            params['gamma_2_b'] = params['eta'] * params['gamma_1_b']
        
        # スペクトル計算
        if mode == 'coupled':
            S_r, eigenmode_props = calculate_reflection_spectrum(params)
        elif mode == 'single_a':
            S_r, eigenmode_props = calculate_single_mode_reflection(params, mode='a')
        elif mode == 'single_b':
            S_r, eigenmode_props = calculate_single_mode_reflection(params, mode='b')
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        results.append({
            'param_value': value,
            'S_r': S_r,
            'eigenmode_props': eigenmode_props,
            'reflectance': np.abs(S_r)**2,
            'phase': np.angle(S_r)
        })
    
    return results

def plot_parameter_sweep(params, results, sweep_param_name, sweep_values, 
                        plot_type='reflectance', title_suffix=''):
    """
    パラメータスイープ結果をプロット
    
    Parameters:
    -----------
    params : dict
        基本パラメータ（波長範囲を含む）
    results : list of dict
        スイープ結果
    sweep_param_name : str
        スイープしたパラメータ名
    sweep_values : array-like
        スイープ値
    plot_type : str
        'reflectance' または 'phase'
    title_suffix : str
        タイトルの追加文字列
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # カラーマップの設定
    colors = plt.cm.viridis(np.linspace(0, 1, len(sweep_values)))
    
    for i, (result, value) in enumerate(zip(results, sweep_values)):
        if plot_type == 'reflectance':
            ax.plot(params['wavelength_range'], result['reflectance'], 
                   color=colors[i], label=f'{sweep_param_name}={value:.2f}',
                   linewidth=1.5, alpha=0.8)
            ylabel = 'Reflectance |S_r|²'
            ylim = (0, 1.1)
        else:  # phase
            ax.plot(params['wavelength_range'], result['phase'], 
                   color=colors[i], label=f'{sweep_param_name}={value:.2f}',
                   linewidth=1.5, alpha=0.8)
            ylabel = 'Phase (rad)'
            ylim = (-np.pi, np.pi)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(params['wavelength_range'][0], params['wavelength_range'][-1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Parameter Sweep: {sweep_param_name}{title_suffix}', fontsize=14)
    
    # 凡例の設定
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    return fig, ax

def create_sweep_values(start, end, step):
    """
    スイープ値の配列を作成
    
    Parameters:
    -----------
    start : float
        開始値
    end : float
        終了値
    step : float
        ステップ
    
    Returns:
    --------
    array : スイープ値の配列
    """
    return np.arange(start, end + step/2, step)

def save_sweep_to_excel(base_params, sweep_param_name, sweep_values, results, 
                        file_path=None, output_dir=None):
    """
    パラメータスイープ結果をExcelファイルに保存
    
    Parameters:
    -----------
    base_params : dict
        基本パラメータ
    sweep_param_name : str
        スイープしたパラメータ名
    sweep_values : array-like
        スイープ値
    results : list of dict
        スイープ結果
    file_path : str, optional
        保存先のファイルパス
    output_dir : str, optional
        出力ディレクトリ
    """
    # デフォルトのファイルパス設定
    if file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"sweep_{sweep_param_name}_{timestamp}.xlsx"
        
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "output", "sweep_data")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: ディレクトリ作成に失敗しました: {e}")
            output_dir = os.getcwd()
        
        file_path = os.path.join(output_dir, file_name)
    
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # 1. スイープ設定シート
            sweep_info = {
                'Parameter': ['Sweep Parameter', 'Start', 'End', 'Step', 'Number of Points'],
                'Value': [sweep_param_name, sweep_values[0], sweep_values[-1], 
                         sweep_values[1] - sweep_values[0] if len(sweep_values) > 1 else 0,
                         len(sweep_values)]
            }
            df_sweep_info = pd.DataFrame(sweep_info)
            df_sweep_info.to_excel(writer, sheet_name='Sweep Info', index=False)
            
            # 2. スイープ結果サマリー
            summary_data = {
                f'{sweep_param_name}': sweep_values,
            }
            
            # 固有モード情報を追加（結合系の場合）
            if 'mode_plus' in results[0]['eigenmode_props']:
                summary_data.update({
                    'Mode+ λ (nm)': [r['eigenmode_props']['mode_plus']['wavelength'] for r in results],
                    'Mode+ Q': [r['eigenmode_props']['mode_plus']['Q'] for r in results],
                    'Mode- λ (nm)': [r['eigenmode_props']['mode_minus']['wavelength'] for r in results],
                    'Mode- Q': [r['eigenmode_props']['mode_minus']['Q'] for r in results],
                })
            else:
                # 単一モードの場合
                summary_data.update({
                    'λ (nm)': [r['eigenmode_props']['wavelength'] for r in results],
                    'Q': [r['eigenmode_props']['Q'] for r in results],
                })
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Sweep Summary', index=False)
            
            # 3. 各スイープ値でのスペクトルデータ（最初の5つまで）
            for i, (value, result) in enumerate(zip(sweep_values[:5], results[:5])):
                sheet_name = f'Spectrum_{sweep_param_name}={value:.2f}'
                spectrum_data = {
                    'Wavelength (nm)': base_params['wavelength_range'],
                    'Reflectance': result['reflectance'],
                    'Phase (rad)': result['phase'],
                }
                df_spectrum = pd.DataFrame(spectrum_data)
                df_spectrum.to_excel(writer, sheet_name=sheet_name, index=False)
            
        print(f"\nスイープデータをExcelファイルに保存しました: {file_path}")
        return file_path
        
    except Exception as e:
        print(f"\nエラー: Excelファイルの保存に失敗しました: {e}")
        return None

def save_to_excel(params, S_r, eigenmode_props, 
                  S_r_single_a=None, mode_props_a=None,
                  S_r_single_b=None, mode_props_b=None,
                  file_path=None, output_dir=None):
    """
    計算結果をExcelファイルに保存
    
    Parameters:
    -----------
    params : dict
        入力パラメータ
    S_r : array
        反射スペクトル（結合系）
    eigenmode_props : dict
        固有モード特性
    S_r_single_a, S_r_single_b : array, optional
        1モード反射スペクトル
    mode_props_a, mode_props_b : dict, optional
        1モード特性
    file_path : str, optional
        保存先のファイルパス（フルパス指定）
    output_dir : str, optional
        出力ディレクトリ（file_pathがNoneの場合に使用）
    """
    # デフォルトのファイルパス設定
    if file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"optical_resonator_{timestamp}.xlsx"
        
        # 出力ディレクトリの設定
        if output_dir is None:
            # カレントディレクトリにoutputフォルダを作成
            output_dir = os.path.join(os.getcwd(), "output", "xlsx_data")
        
        # ディレクトリが存在しない場合は作成
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: ディレクトリ作成に失敗しました: {e}")
            # フォールバック：カレントディレクトリを使用
            output_dir = os.getcwd()
        
        file_path = os.path.join(output_dir, file_name)
    
    # gamma_2の計算
    gamma_2_a = params['eta'] * params['gamma_1_a']
    gamma_2_b = params['eta'] * params['gamma_1_b']
    
    # Excelライターの作成
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            
            # 1. 入力パラメータシート
            input_params_data = {
                'Parameter': [],
                'Value': [],
                'Unit': []
            }
            
            # 減衰率パラメータ
            param_info = [
                ('gamma_abs_a', 'Absorption loss mode a', 'meV'),
                ('gamma_abs_b', 'Absorption loss mode b', 'meV'),
                ('gamma_1_a', 'Port 1 coupling loss mode a', 'meV'),
                ('gamma_1_b', 'Port 1 coupling loss mode b', 'meV'),
                ('eta', 'Port 2 coupling ratio', ''),
                ('gamma_2_a (calculated)', f'{gamma_2_a:.3f}', 'meV'),
                ('gamma_2_b (calculated)', f'{gamma_2_b:.3f}', 'meV'),
                ('lambda_a', 'Resonance wavelength mode a', 'nm'),
                ('lambda_b', 'Resonance wavelength mode b', 'nm'),
                ('g_r', 'Mode coupling (real part)', 'meV'),
                ('g_i (calculated)', f'{(np.sqrt(params["gamma_1_a"] * params["gamma_1_b"]) + np.sqrt(gamma_2_a * gamma_2_b)):.3f}', 'meV'),
                ('r', 'Port coupling reflection coefficient', ''),
                ('t', 'Port coupling transmission coefficient', ''),
                ('n', 'Refractive index', ''),
                ('l', 'Phase matching layer thickness', 'm'),
            ]
            
            for item in param_info:
                if len(item) == 3:
                    key, desc, unit = item
                    if key in params:
                        input_params_data['Parameter'].append(desc)
                        input_params_data['Value'].append(params[key])
                        input_params_data['Unit'].append(unit)
                    else:
                        # 計算値の場合
                        input_params_data['Parameter'].append(item[0])
                        input_params_data['Value'].append(item[1])
                        input_params_data['Unit'].append(item[2])
            
            df_input = pd.DataFrame(input_params_data)
            df_input.to_excel(writer, sheet_name='Input Parameters', index=False)
            
            # 2. 1モード解析結果（新規追加）
            if mode_props_a is not None and mode_props_b is not None:
                single_mode_data = {
                    'Mode': ['Mode a', 'Mode b'],
                    'Wavelength (nm)': [
                        mode_props_a['wavelength'],
                        mode_props_b['wavelength']
                    ],
                    'Energy (meV)': [
                        mode_props_a['energy'],
                        mode_props_b['energy']
                    ],
                    'Linewidth (meV)': [
                        mode_props_a['linewidth'],
                        mode_props_b['linewidth']
                    ],
                    'Reflectance': [
                        mode_props_a.get('reflection_intensity', 0.0),
                        mode_props_b.get('reflection_intensity', 0.0)
                    ],
                    'Q value': [
                        mode_props_a['Q'],
                        mode_props_b['Q']
                    ]
                }
                df_single = pd.DataFrame(single_mode_data)
                df_single.to_excel(writer, sheet_name='Single Mode Analysis', index=False)
            
            # 3. 結合モード解析結果
            output_data = {
                'Mode': ['Mode +', 'Mode -'],
                'Wavelength (nm)': [
                    eigenmode_props['mode_plus']['wavelength'],
                    eigenmode_props['mode_minus']['wavelength']
                ],
                'Energy (meV)': [
                    eigenmode_props['mode_plus']['energy'],
                    eigenmode_props['mode_minus']['energy']
                ],
                'Linewidth (μeV)': [
                    eigenmode_props['mode_plus']['linewidth'] * 1000,
                    eigenmode_props['mode_minus']['linewidth'] * 1000
                ],
                'Q value': [
                    eigenmode_props['mode_plus']['Q'],
                    eigenmode_props['mode_minus']['Q']
                ],
                'Reflectance': [
                    eigenmode_props['mode_plus']['reflection_intensity'],
                    eigenmode_props['mode_minus']['reflection_intensity']
                ]
            }
            
            # 追加の計算パラメータ
            gamma_a = 2 * (params['gamma_abs_a'] + params['gamma_1_a'] + gamma_2_a)
            gamma_b = 2 * (params['gamma_abs_b'] + params['gamma_1_b'] + gamma_2_b)
            # 放射率
            gamma_rad_a = params['gamma_1_a'] + gamma_2_a
            gamma_rad_b = params['gamma_1_b'] + gamma_2_b
            energy_a = 1240000.0 / params['lambda_a']
            energy_b = 1240000.0 / params['lambda_b']
            Q_a = energy_a / gamma_a
            Q_b = energy_b / gamma_b
            
            # 元のモードの情報も追加
            additional_info = {
                'Parameter': [
                    'Original Mode a Energy',
                    'Original Mode b Energy',
                    'Total damping rate γ_a',
                    'Total damping rate γ_b',
                    'Radiative damping rate γ_rad_a',
                    'Radiative damping rate γ_rad_b',
                    'Original Q_a',
                    'Original Q_b',
                    'Evaluation wavelength'
                ],
                'Value': [
                    energy_a,
                    energy_b,
                    gamma_a,
                    gamma_b,
                    gamma_rad_a,
                    gamma_rad_b,
                    Q_a,
                    Q_b,
                    eigenmode_props['evaluation_wavelength']
                ],
                'Unit': [
                    'meV',
                    'meV',
                    'meV',
                    'meV',
                    'meV',
                    'meV',
                    '',
                    '',
                    'nm'
                ]
            }
            
            df_output = pd.DataFrame(output_data)
            df_output.to_excel(writer, sheet_name='Coupled Mode Analysis', index=False)
            
            # 追加情報を同じシートの下に追加
            df_additional = pd.DataFrame(additional_info)
            df_additional.to_excel(writer, sheet_name='Coupled Mode Analysis', 
                                  startrow=len(df_output) + 3, index=False)
            
            # 4. 結合系スペクトルデータ
            spectrum_data = {
                'Wavelength (nm)': params['wavelength_range'],
                'Reflectance': np.abs(S_r)**2,
                'Phase (rad)': np.angle(S_r),
                'Re[S_r]': np.real(S_r),
                'Im[S_r]': np.imag(S_r)
            }
            
            df_spectrum = pd.DataFrame(spectrum_data)
            df_spectrum.to_excel(writer, sheet_name='Coupled Spectrum Data', index=False)
            
            # 5. 1モードスペクトルデータ（新規追加）
            if S_r_single_a is not None and S_r_single_b is not None:
                single_spectrum_data = {
                    'Wavelength (nm)': params['wavelength_range'],
                    'Reflectance_a': np.abs(S_r_single_a)**2,
                    'Phase_a (rad)': np.angle(S_r_single_a),
                    'Reflectance_b': np.abs(S_r_single_b)**2,
                    'Phase_b (rad)': np.angle(S_r_single_b),
                }
                
                df_single_spectrum = pd.DataFrame(single_spectrum_data)
                df_single_spectrum.to_excel(writer, sheet_name='Single Mode Spectrum Data', index=False)
        
            print(f"\nデータをExcelファイルに保存しました: {file_path}")
        
    except PermissionError:
        print(f"\nエラー: ファイル '{file_path}' に書き込めません。")
        print("ファイルが開かれている可能性があります。")
        return None
    except Exception as e:
        print(f"\nエラー: Excelファイルの保存に失敗しました: {e}")
        return None
    
    return file_path

# 使用例
if __name__ == "__main__":
    # 基本パラメータ設定
    base_params = {
        # 減衰率 [meV]
        'gamma_abs_a': 10,    
        'gamma_abs_b': 2,
        'gamma_1_a': 10,
        'gamma_1_b': 0.1,     
        'eta': 0.001,   
        
        # 共振波長 [nm]
        'lambda_a': 800.0,       
        'lambda_b': 900.0,       
        
        # モード間結合 [meV]
        'g_r': 3,                 
        
        # ポート結合の反射・透過係数（より現実的な値）
        'r': 1,                         # 部分反射
        't': np.sqrt(1 - 1**2),         # ≈ 0.436
        
        # 位相整合層
        'n': 1.37,
        'l': 1.5e-7,            
        'l_g': 0.1e-7,
        'phi_m': -170,
        
        # 波長範囲 [nm]
        'wavelength_range': np.linspace(600, 1200, 1200)
    }
    
    
    # スイープ設定の例
    sweep_configs = {
        'gamma_abs_a': {'start': 10, 'end': 50, 'step': 10},
        'gamma_1_a': {'start': 0, 'end': 40, 'step': 5},
        'g_r': {'start': 0, 'end': 10, 'step': 2},
        'eta': {'start': 0.2, 'end': 1.0, 'step': 0.2},
        'r': {'start': 0.7, 'end': 0.95, 'step': 0.05},
        'lambda_a': {'start': 780, 'end': 820, 'step': 10},
        'n': {'start': 1.3, 'end': 1.5, 'step': 0.05},
        'l': {'start': 1.0e-7, 'end': 2.0e-7, 'step': 2.0e-8},
    }
    
    # 実行するスイープを選択（例：gamma_abs_aをスイープ）
    sweep_param = 'gamma_1_a'  # 変更可能: 'gamma_abs_a', 'gamma_1_a', 'g_r', 'eta', 'r' など
    
    # スイープ実行のON/OFF
    do_sweep = True  # False にすると通常の単一計算を実行
    
    if do_sweep and sweep_param in sweep_configs:
        config = sweep_configs[sweep_param]
        sweep_values = create_sweep_values(config['start'], config['end'], config['step'])
        
        print(f"\n=== Parameter Sweep: {sweep_param} ===")
        print(f"Range: {config['start']} to {config['end']}, Step: {config['step']}")
        print(f"Values: {sweep_values}")
        
        # スイープ実行
        print("\nCalculating spectra...")
        results_coupled = parameter_sweep(base_params, sweep_param, sweep_values, mode='coupled')
        results_single_a = parameter_sweep(base_params, sweep_param, sweep_values, mode='single_a')
        
        # 結果のサマリー表示
        print(f"\n{'='*80}")
        print(f"Sweep Results Summary: {sweep_param}")
        print(f"{'='*80}")
        print(f"{'Value':>8} | {'Mode+λ(nm)':>10} | {'Mode-λ(nm)':>10} | {'Mode+Q':>8} | {'Mode-Q':>8}")
        print(f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
        
        for i, value in enumerate(sweep_values):
            if 'mode_plus' in results_coupled[i]['eigenmode_props']:
                mode_plus = results_coupled[i]['eigenmode_props']['mode_plus']
                mode_minus = results_coupled[i]['eigenmode_props']['mode_minus']
                print(f"{value:>8.2f} | {mode_plus['wavelength']:>10.2f} | "
                      f"{mode_minus['wavelength']:>10.2f} | "
                      f"{mode_plus['Q']:>8.0f} | {mode_minus['Q']:>8.0f}")
        
        # プロット作成
        # 1. 結合系の反射率スペクトル
        fig1, ax1 = plot_parameter_sweep(base_params, results_coupled, sweep_param, 
                                        sweep_values, plot_type='reflectance',
                                        title_suffix=' - Coupled System')
        
        # 2. 結合系の位相スペクトル
        fig2, ax2 = plot_parameter_sweep(base_params, results_coupled, sweep_param, 
                                        sweep_values, plot_type='phase',
                                        title_suffix=' - Coupled System')
        
        # 3. 単一モードAの反射率スペクトル
        fig3, ax3 = plot_parameter_sweep(base_params, results_single_a, sweep_param, 
                                        sweep_values, plot_type='reflectance',
                                        title_suffix=' - Single Mode A')
        
        # 4. 2Dカラーマップ（coupled_mode）
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        
        # 反射率データを2D配列に整理
        reflectance_2d = np.array([result['reflectance'] for result in results_coupled])
        
        # カラーマップ表示
        im = ax4.imshow(reflectance_2d, aspect='auto', origin='lower',
                       extent=[base_params['wavelength_range'][0], 
                              base_params['wavelength_range'][-1],
                              sweep_values[0], sweep_values[-1]],
                       cmap='hot', vmin=0, vmax=1)
        
        ax4.set_xlabel('Wavelength (nm)', fontsize=12)
        ax4.set_ylabel(f'{sweep_param}', fontsize=12)
        ax4.set_title(f'Reflectance Map - Parameter Sweep: {sweep_param}', fontsize=14)
        
        # カラーバー追加
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Reflectance |S_r|²', fontsize=12)

        # 5. 2Dカラーマップ（single_mode）
        fig5, ax5 = plt.subplots(figsize=(10, 6))

        # 反射率データを2D配列に整理
        reflectance_2d = np.array([result['reflectance'] for result in results_single_a])
        
        # カラーマップ表示(single_mode)
        im = ax5.imshow(reflectance_2d, aspect='auto', origin='lower',
                       extent=[base_params['wavelength_range'][0], 
                              base_params['wavelength_range'][-1],
                              sweep_values[0], sweep_values[-1]],
                       cmap='hot', vmin=0, vmax=1)
        
        ax5.set_xlabel('Wavelength (nm)', fontsize=12)
        ax5.set_ylabel(f'{sweep_param}', fontsize=12)
        ax5.set_title(f'Reflectance Map - Parameter Sweep: {sweep_param}', fontsize=14)
        
        # カラーバー追加
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('Reflectance |S_r|²', fontsize=12)
        
        # スイープ結果をExcelに保存
        sweep_excel_path = save_sweep_to_excel(base_params, sweep_param, sweep_values, 
                                              results_coupled)
        
        plt.show()
        
    else:
        # 通常の単一計算実行（元のコードと同じ）
        params = base_params.copy()
        
        # gamma_2の計算値を追加
        gamma_2_a = params['eta'] * params['gamma_1_a']
        gamma_2_b = params['eta'] * params['gamma_1_b']
        
        # 1モード（mode a のみ）の反射スペクトル計算
        S_r_single_a, mode_props_a = calculate_single_mode_reflection(params, mode='a')
        
        # 1モード（mode b のみ）の反射スペクトル計算
        S_r_single_b, mode_props_b = calculate_single_mode_reflection(params, mode='b')
        
        # 2モード結合系の反射スペクトル計算
        S_r_coupled, eigenmode_props = calculate_reflection_spectrum(params)
        
        # 数値表を先に出力（グラフ表示前に）
        print("\n" + "="*80)
        print("1モード解析（独立）")
        print("="*80)
        
        print("| Mode  | Wavelength (nm) | Reflectance | Linewidth (meV) | Q value |")
        print("|-------|----------------|------------|-----------------|---------|")
        print(f"| Mode a | {mode_props_a['wavelength']:>14.2f} | {mode_props_a['reflection_intensity']:>11.6f} | "
              f"{mode_props_a['linewidth']:>15.3f} | {mode_props_a['Q']:>7.0f} |")
        print(f"| Mode b | {mode_props_b['wavelength']:>14.2f} | {mode_props_b['reflection_intensity']:>11.6f} | "
              f"{mode_props_b['linewidth']:>15.3f} | {mode_props_b['Q']:>7.0f} |")
        
        print("\n" + "="*80)
        print("Coupled Mode Dip Analysis")
        print("="*80)
        print(f"評価波長: {eigenmode_props['evaluation_wavelength']:.1f} nm\n")

        print("| Mode  | Wavelength (nm) | Reflectance | Linewidth (μeV) | Q value |")
        print("|-------|----------------|------------|-----------------|---------|")
        
        for mode_name, mode_label in [('mode_plus', 'Mode +'), ('mode_minus', 'Mode -')]:
            mode = eigenmode_props[mode_name]
            print(f"| {mode_label:<5} | {mode['wavelength']:>14.2f} | {mode['reflection_intensity']:>11.6f} | "
                  f"{mode['linewidth']*1000:>15.2f} | {mode['Q']:>7.0f} |")
        
        # パラメータサマリー
        gamma_a = 2 * (params['gamma_abs_a'] + params['gamma_1_a'] + gamma_2_a)
        gamma_b = 2 * (params['gamma_abs_b'] + params['gamma_1_b'] + gamma_2_b)
        gamma_rad_a = params['gamma_1_a'] + gamma_2_a
        gamma_rad_b = params['gamma_1_b'] + gamma_2_b
        energy_a = 1240000.0 / params['lambda_a']
        energy_b = 1240000.0 / params['lambda_b']
        
        print("\n" + "="*80)
        print("パラメータサマリー")
        print("="*80)
        print(f"モードa: {energy_a:.2f} meV ({params['lambda_a']:.1f} nm)")
        print(f"モードb: {energy_b:.2f} meV ({params['lambda_b']:.1f} nm)")
        gi_auto = np.sqrt(params['gamma_1_a'] * params['gamma_1_b']) + np.sqrt(gamma_2_a * gamma_2_b)
        print(f"モード間結合: g = ({params['g_r']:.3f} - i{gi_auto:.3f}) meV")
        print(f"全減衰率γ_a: {gamma_a:.3f} meV ({gamma_a*1000:.1f} μeV)")
        print(f"全減衰率γ_b: {gamma_b:.3f} meV ({gamma_b*1000:.1f} μeV)")
        
        # 元のモードのQ値
        Q_a = energy_a / gamma_a
        Q_b = energy_b / gamma_b
        print(f"\n元のモードのQ値: Q_a = {Q_a:.0f}, Q_b = {Q_b:.0f}")
        print("="*80 + "\n")
        
        # Excelファイルに保存（1モードの結果も含む）
        # カスタム出力ディレクトリを指定する場合：
        # output_dir = "/path/to/your/directory"
        # excel_file_path = save_to_excel(params, S_r_coupled, eigenmode_props,
        #                                 S_r_single_a, mode_props_a,
        #                                 S_r_single_b, mode_props_b,
        #                                 output_dir=output_dir)
        
        excel_file_path = save_to_excel(params, S_r_coupled, eigenmode_props,
                                        S_r_single_a, mode_props_a,
                                        S_r_single_b, mode_props_b)
        
        # プロット（2x2グリッド）
        title_fontsize = 12      # グラフタイトル
        label_fontsize = 10      # 軸ラベル
        tick_fontsize  = 9       # 目盛り数字
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.95, hspace=0.3, wspace=0.25)

        # 1. 1モード反射スペクトル（左上）
        ax1.plot(params['wavelength_range'], np.abs(S_r_single_a)**2, 'r-', 
                 linewidth=1.5, label='Mode a only')
        ax1.plot(params['wavelength_range'], np.abs(S_r_single_b)**2, 'b-', 
                 linewidth=1.5, label='Mode b only')
        ax1.set_ylabel('Reflectance |S_r|²', fontsize=label_fontsize)
        ax1.set_xlim(600, 1200)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Single Mode Reflection Spectra', fontsize=title_fontsize)
        ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax1.legend(loc='upper right', fontsize=9)
        
        # 共振波長を縦線で表示
        ax1.axvline(params['lambda_a'], color='red', linestyle='--', alpha=0.3)
        ax1.axvline(params['lambda_b'], color='blue', linestyle='--', alpha=0.3)
        
        # 2. 1モード位相スペクトル（右上）
        ax2.plot(params['wavelength_range'], np.angle(S_r_single_a), 'r-', 
                 linewidth=1.5, label='Mode a only')
        ax2.plot(params['wavelength_range'], np.angle(S_r_single_b), 'b-', 
                 linewidth=1.5, label='Mode b only')
        ax2.set_ylabel('Phase (rad)', fontsize=label_fontsize)
        ax2.set_xlim(600, 1200)
        ax2.set_ylim(-np.pi, np.pi)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Single Mode Phase Spectra', fontsize=title_fontsize)
        ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax2.legend(loc='upper right', fontsize=9)
        
        # 共振波長を縦線で表示
        ax2.axvline(params['lambda_a'], color='red', linestyle='--', alpha=0.3)
        ax2.axvline(params['lambda_b'], color='blue', linestyle='--', alpha=0.3)
        
        # 3. 2モード反射スペクトル（左下）
        ax3.plot(params['wavelength_range'], np.abs(S_r_coupled)**2, 'k-', linewidth=1.5)
        ax3.set_xlabel('Wavelength (nm)', fontsize=label_fontsize)
        ax3.set_ylabel('Reflectance |S_r|²', fontsize=label_fontsize)
        ax3.set_xlim(600, 1200)
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Coupled Two-Mode Reflection Spectrum', fontsize=title_fontsize)
        ax3.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # 元の共振波長を縦線で表示
        ax3.axvline(params['lambda_a'], color='red', linestyle='--', alpha=0.3, 
                    label=f'Mode a ({params["lambda_a"]:.1f} nm)')
        ax3.axvline(params['lambda_b'], color='blue', linestyle='--', alpha=0.3, 
                    label=f'Mode b ({params["lambda_b"]:.1f} nm)')
        
        # 固有モードの波長を縦線で表示（範囲内の場合のみ）
        if 600 <= eigenmode_props['mode_plus']['wavelength'] <= 1200:
            ax3.axvline(eigenmode_props['mode_plus']['wavelength'], color='green', 
                        linestyle=':', linewidth=2, 
                        label=f'Mode + ({eigenmode_props["mode_plus"]["wavelength"]:.1f} nm)')
        if 600 <= eigenmode_props['mode_minus']['wavelength'] <= 1200:
            ax3.axvline(eigenmode_props['mode_minus']['wavelength'], color='orange', 
                        linestyle=':', linewidth=2,
                        label=f'Mode - ({eigenmode_props["mode_minus"]["wavelength"]:.1f} nm)')
        ax3.legend(loc='upper right', fontsize=9)
        
        # 4. 2モード位相スペクトル（右下）
        ax4.plot(params['wavelength_range'], np.angle(S_r_coupled), 'g-', linewidth=1.5)
        ax4.set_xlabel('Wavelength (nm)', fontsize=label_fontsize)
        ax4.set_ylabel('Phase (rad)', fontsize=label_fontsize)
        ax4.set_xlim(600, 1200)
        ax4.set_ylim(-np.pi, np.pi)
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Coupled Two-Mode Phase Spectrum', fontsize=title_fontsize)
        ax4.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # 固有モードの波長を縦線で表示
        ax4.axvline(params['lambda_a'], color='red', linestyle='--', alpha=0.3)
        ax4.axvline(params['lambda_b'], color='blue', linestyle='--', alpha=0.3)
        if 600 <= eigenmode_props['mode_plus']['wavelength'] <= 1200:
            ax4.axvline(eigenmode_props['mode_plus']['wavelength'], color='green', 
                        linestyle=':', linewidth=2, alpha=0.5)
        if 600 <= eigenmode_props['mode_minus']['wavelength'] <= 1200:
            ax4.axvline(eigenmode_props['mode_minus']['wavelength'], color='orange', 
                        linestyle=':', linewidth=2, alpha=0.5)

        # パラメータテーブル
        param_table = [
            ('γ_abs_a', f'{params["gamma_abs_a"]:.1f}', 'meV'),
            ('γ_rad_a', f'{gamma_rad_a:.1f}', 'meV'),
            ('γ_abs_b', f'{params["gamma_abs_b"]:.1f}', 'meV'),
            ('γ_rad_b', f'{gamma_rad_b:.1f}', 'meV'),
            ('λ_a', f'{params["lambda_a"]:.1f}', 'nm'),
            ('λ_b', f'{params["lambda_b"]:.1f}', 'nm'),
            ('Q_a', f'{Q_a:.0f}', ''),
            ('Q_b', f'{Q_b:.0f}', ''),
            ('g_r', f'{params["g_r"]:.3f}', 'meV'),
            ('g_i', f'{gi_auto:.3f}', 'meV'),
            ('η', f'{params["eta"]:.1f}', ''),
            ('n', f'{params["n"]:.2f}', ''),
            ('l', f'{params["l"]*1e9:.1f}', 'nm'),
        ]
        cell_text = [[k, v, u] for (k, v, u) in param_table]

        # 表を下部に追加
        ax_table = fig.add_axes([0.15, 0.02, 0.7, 0.18])  # [left, bottom, width, height]
        ax_table.axis('off')
        table = ax_table.table(
            cellText=cell_text,
            colLabels=['Parameter', 'Value', 'Unit'],
            loc='center',
            cellLoc='center',
            colColours=['#e6e6e6']*3
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        plt.show()