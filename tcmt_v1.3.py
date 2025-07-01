"""
Optical resonator reflection spectrum calculation with parameter sweep.
Modified according to instructions:
- Imaginary coupling gi is now computed as sum_j sqrt(gamma_j_a * gamma_j_b) for j=1,2.
- Removed print of coupled mode eigenvalues.
- Print and export tables for single mode A, coupled mode original modes, and
  reflectance dip peaks.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import warnings

try:
    import openpyxl  # noqa: F401
except ImportError:
    warnings.warn("openpyxlがインストールされていません。Excel保存機能が使用できません。")

hbar = 1.054571817e-34
meV_to_J = 1.602176634e-22
meV_to_rad_s = meV_to_J / hbar


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def compute_gi(gamma_1_a, gamma_1_b, gamma_2_a, gamma_2_b):
    """Calculate imaginary coupling gi from radiative rates."""
    return np.sqrt(gamma_1_a * gamma_1_b) + np.sqrt(gamma_2_a * gamma_2_b)


def find_absorption_peaks(wavelengths, reflectance):
    """Find dips (absorption peaks) in reflectance spectrum.

    Parameters
    ----------
    wavelengths : array
        Wavelength array [nm].
    reflectance : array
        Reflectance |S_r|^2.

    Returns
    -------
    list of dict
        Each dict has keys: wavelength, absorption, linewidth, Q.
    """
    absorption = 1 - reflectance
    peaks = []
    for i in range(1, len(absorption) - 1):
        if absorption[i] > absorption[i - 1] and absorption[i] >= absorption[i + 1]:
            peaks.append(i)
    results = []
    for idx in peaks:
        lam_peak = wavelengths[idx]
        peak_val = absorption[idx]
        half = peak_val / 2
        # search left
        left = idx
        while left > 0 and absorption[left] > half:
            left -= 1
        # interpolate
        if left == 0:
            lam_left = wavelengths[0]
        else:
            lam_left = np.interp(
                half,
                [absorption[left], absorption[left + 1]],
                [wavelengths[left], wavelengths[left + 1]],
            )
        # search right
        right = idx
        while right < len(absorption) - 1 and absorption[right] > half:
            right += 1
        if right == len(absorption) - 1:
            lam_right = wavelengths[-1]
        else:
            lam_right = np.interp(
                half,
                [absorption[right - 1], absorption[right]],
                [wavelengths[right - 1], wavelengths[right]],
            )
        linewidth = lam_right - lam_left
        Q = lam_peak / linewidth if linewidth > 0 else np.nan
        results.append(
            {
                "wavelength": lam_peak,
                "absorption": peak_val,
                "linewidth": linewidth,
                "Q": Q,
            }
        )
    return results


# ----------------------------------------------------------------------
# Calculation functions
# ----------------------------------------------------------------------

def calculate_eigenmode_properties(params, wavelength_eval=None):
    """Calculate eigenmode properties of coupled system."""
    gamma_abs_a = params["gamma_abs_a"]
    gamma_abs_b = params["gamma_abs_b"]
    gamma_1_a = params["gamma_1_a"]
    gamma_1_b = params["gamma_1_b"]
    lambda_a = params["lambda_a"]
    lambda_b = params["lambda_b"]
    eta = params["eta"]
    g_r = params["g_r"]
    r = params["r"]
    t = params["t"]
    n = params["n"]
    l = params["l"]
    l_g = params["l_g"]
    phi_m = params["phi_m"]

    gamma_2_a = eta * gamma_1_a
    gamma_2_b = eta * gamma_1_b
    g_i = compute_gi(gamma_1_a, gamma_1_b, gamma_2_a, gamma_2_b)

    energy_a = 1240000.0 / lambda_a
    energy_b = 1240000.0 / lambda_b

    g = g_r - 1j * g_i

    gamma_a =  (gamma_abs_a + gamma_1_a + gamma_2_a)
    gamma_b =  (gamma_abs_b + gamma_1_b + gamma_2_b)

    d_2a = np.sqrt(2 * gamma_2_a)
    d_2b = np.sqrt(2 * gamma_2_b)

    if wavelength_eval is None:
        wavelength_eval = (lambda_a + lambda_b) / 2

    wavelength_m = wavelength_eval * 1e-9
    k = 2 * np.pi / wavelength_m
    phi = 2 * n * l * k

    r_phi = np.exp(1j * phi) / (1 - r * np.exp(1j * phi))

    gamma_a_eff = gamma_a - d_2a ** 2 * r_phi 
    gamma_b_eff = gamma_b - d_2b ** 2 * r_phi 

    g_eff = g + r_phi * d_2a * d_2b

    delta_omega = energy_a - energy_b
    delta_gamma = gamma_a_eff - gamma_b_eff

    sqrt_term = np.sqrt((delta_omega - 1j * delta_gamma) ** 2 + 4 * np.abs(g_eff) ** 2)

    omega_plus = (energy_a + energy_b) / 2 + np.real(sqrt_term) / 2
    omega_minus = (energy_a + energy_b) / 2 - np.real(sqrt_term) / 2

    gamma_plus = (np.real(gamma_a_eff) + np.real(gamma_b_eff)) / 2 - np.imag(sqrt_term) / 2
    gamma_minus = (np.real(gamma_a_eff) + np.real(gamma_b_eff)) / 2 + np.imag(sqrt_term) / 2

    gamma_plus = np.real(gamma_plus)
    gamma_minus = np.real(gamma_minus)

    lambda_plus = 1240000.0 / omega_plus if omega_plus > 0 else np.inf
    lambda_minus = 1240000.0 / omega_minus if omega_minus > 0 else np.inf

    Q_plus = omega_plus / gamma_plus if gamma_plus > 0 else 0
    Q_minus = omega_minus / gamma_minus if gamma_minus > 0 else 0

    return {
        "mode_plus": {
            "wavelength": lambda_plus,
            "energy": omega_plus,
            "linewidth": gamma_plus,
            "Q": Q_plus,
        },
        "mode_minus": {
            "wavelength": lambda_minus,
            "energy": omega_minus,
            "linewidth": gamma_minus,
            "Q": Q_minus,
        },
        "evaluation_wavelength": wavelength_eval,
    }


def calculate_single_mode_reflection(params, mode="a"):
    """Calculate single mode reflection spectrum."""
    if mode == "a":
        gamma_abs = params["gamma_abs_a"]
        gamma_1 = params["gamma_1_a"]
        lambda_mode = params["lambda_a"]
    else:
        gamma_abs = params["gamma_abs_b"]
        gamma_1 = params["gamma_1_b"]
        lambda_mode = params["lambda_b"]

    eta = params["eta"]
    r = params["r"]
    t = params["t"]
    n = params["n"]
    l = params["l"]
    l_g = params["l_g"]
    phi_m = params["phi_m"]
    wavelength_range = params["wavelength_range"]

    gamma_2 = eta * gamma_1

    energy_mode = 1240000.0 / lambda_mode

    gamma_total =  (gamma_abs + gamma_1 + gamma_2)

    d_1 = np.sqrt(2 * gamma_1)
    d_2 = np.sqrt(2 * gamma_2)

    S_r = []
    for wavelength_nm in wavelength_range:
        energy = 1240000.0 / wavelength_nm
        wavelength = wavelength_nm * 1e-9
        k = 2 * np.pi / wavelength
        phi = 2 * (n * l + l_g) * k + phi_m
        r_phi = -np.exp(1j * phi) / (1 + r * np.exp(1j * phi))
        gamma_eff = gamma_total - d_2 ** 2 * r_phi 
        D = d_1 + r_phi * t * d_2
        D_star = np.conj(D)
        delta_energy = energy - energy_mode
        numerator = D * D_star
        denominator = 1j * delta_energy + gamma_eff
        S_r_value = -r + r_phi * t ** 2 + numerator / denominator
        S_r.append(S_r_value)

    reflectance = np.abs(S_r) ** 2
    idx = np.argmin(np.abs(wavelength_range - lambda_mode))
    refl_at_res = reflectance[idx]

    Q = energy_mode / gamma_total

    mode_props = {
        "wavelength": lambda_mode,
        "energy": energy_mode,
        "linewidth": gamma_total,
        "Q": Q,
        "reflection": refl_at_res,
        "gamma_eff_at_resonance": gamma_eff,
    }

    return np.array(S_r), mode_props


def calculate_reflection_spectrum(params):
    """Calculate reflection spectrum for coupled system."""
    gamma_abs_a = params["gamma_abs_a"]
    gamma_abs_b = params["gamma_abs_b"]
    gamma_1_a = params["gamma_1_a"]
    gamma_1_b = params["gamma_1_b"]
    eta = params["eta"]
    lambda_a = params["lambda_a"]
    lambda_b = params["lambda_b"]

    gamma_2_a = eta * gamma_1_a
    gamma_2_b = eta * gamma_1_b
    g_i = compute_gi(gamma_1_a, gamma_1_b, gamma_2_a, gamma_2_b)

    energy_a = 1240000.0 / lambda_a
    energy_b = 1240000.0 / lambda_b
    g_r = params["g_r"]
    r = params["r"]
    t = params["t"]
    n = params["n"]
    l = params["l"]
    wavelength_range = params["wavelength_range"]

    g = g_r - 1j * g_i
    g_star = g_r + 1j * g_i

    gamma_a =  (gamma_abs_a + gamma_1_a + gamma_2_a)
    gamma_b =  (gamma_abs_b + gamma_1_b + gamma_2_b)

    d_1a = np.sqrt(2 * gamma_1_a)
    d_1b = np.sqrt(2 * gamma_1_b)
    d_2a = np.sqrt(2 * gamma_2_a)
    d_2b = np.sqrt(2 * gamma_2_b)

    S_r = []
    for wavelength_nm in wavelength_range:
        energy = 1240000.0 / wavelength_nm
        wavelength = wavelength_nm * 1e-9
        k = 2 * np.pi / wavelength
        phi = 2 * n * l * k
        r_phi = -np.exp(1j * phi) / (1 + r * np.exp(1j * phi))
        gamma_a_eff = gamma_a - d_2a ** 2 * r_phi 
        gamma_b_eff = gamma_b - d_2b ** 2 * r_phi 
        g_eff = g + r_phi * d_2a * d_2b
        g_star_eff = g_star + r_phi * d_2a * d_2b
        D_a = d_1a + r_phi * t * d_2a
        D_b = d_1b + r_phi * t * d_2b
        D_a_star = np.conj(D_a)
        D_b_star = np.conj(D_b)
        delta_energy_a = energy - energy_a
        delta_energy_b = energy - energy_b
        M11 = 1j * delta_energy_a + gamma_a_eff
        M12 = g_eff
        M21 = g_star_eff
        M22 = 1j * delta_energy_b + gamma_b_eff
        det_M = M11 * M22 - M12 * M21
        term1 = D_a_star * (1j * delta_energy_a + gamma_a_eff) + D_b_star * (
            1j * delta_energy_b + gamma_b_eff
        )
        term2 = D_a_star * D_b * g_eff + D_a * D_b_star * g_star_eff
        S_m = (term1 + term2) / det_M
        S_r_value = t ** 2 * r_phi - r + S_m
        S_r.append(S_r_value)

    S_r = np.array(S_r)
    eigenmode_props = calculate_eigenmode_properties(params)

    reflectance = np.abs(S_r) ** 2
    for mode_name in ["mode_plus", "mode_minus"]:
        mode_lambda = eigenmode_props[mode_name]["wavelength"]
        if mode_lambda != np.inf and 600 <= mode_lambda <= 1200:
            idx = np.argmin(np.abs(wavelength_range - mode_lambda))
            eigenmode_props[mode_name]["reflection_intensity"] = reflectance[idx]
        else:
            eigenmode_props[mode_name]["reflection_intensity"] = 0.0

    mode_info = {
        "mode_a": {
            "wavelength": lambda_a,
            "energy": energy_a,
            "linewidth": gamma_a,
            "Q": energy_a / gamma_a,
        },
        "mode_b": {
            "wavelength": lambda_b,
            "energy": energy_b,
            "linewidth": gamma_b,
            "Q": energy_b / gamma_b,
        },
    }
    for name, info in mode_info.items():
        idx = np.argmin(np.abs(wavelength_range - info["wavelength"]))
        info["reflection"] = reflectance[idx]

    dip_peaks = find_absorption_peaks(wavelength_range, reflectance)

    return S_r, eigenmode_props, mode_info, dip_peaks


# ----------------------------------------------------------------------
# Parameter sweep utilities
# ----------------------------------------------------------------------

def parameter_sweep(base_params, sweep_param_name, sweep_values, mode="coupled"):
    results = []
    for value in sweep_values:
        params = base_params.copy()
        params[sweep_param_name] = value
        if sweep_param_name == "r":
            params["t"] = np.sqrt(1 - value ** 2)
        if sweep_param_name in ["eta", "gamma_1_a", "gamma_1_b"]:
            params["gamma_2_a"] = params["eta"] * params["gamma_1_a"]
            params["gamma_2_b"] = params["eta"] * params["gamma_1_b"]
        if mode == "coupled":
            S_r, eigenmode_props, mode_info, dip_peaks = calculate_reflection_spectrum(params)
        elif mode == "single_a":
            S_r, mode_props = calculate_single_mode_reflection(params, mode="a")
            eigenmode_props = {"mode_a": mode_props}
            mode_info = None
            dip_peaks = None
        elif mode == "single_b":
            S_r, mode_props = calculate_single_mode_reflection(params, mode="b")
            eigenmode_props = {"mode_b": mode_props}
            mode_info = None
            dip_peaks = None
        else:
            raise ValueError(f"Unknown mode: {mode}")
        results.append(
            {
                "param_value": value,
                "S_r": S_r,
                "eigenmode_props": eigenmode_props,
                "mode_info": mode_info,
                "dip_peaks": dip_peaks,
                "reflectance": np.abs(S_r) ** 2,
                "phase": np.angle(S_r),
            }
        )
    return results


# ----------------------------------------------------------------------
# Excel output
# ----------------------------------------------------------------------

def save_to_excel(
    params,
    S_r,
    eigenmode_props,
    mode_info,
    dip_peaks,
    S_r_single_a=None,
    mode_props_a=None,
    S_r_single_b=None,
    mode_props_b=None,
    file_path=None,
    output_dir=None,
):
    if file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"optical_resonator_{timestamp}.xlsx"
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "output", "xlsx_data")
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            output_dir = os.getcwd()
        file_path = os.path.join(output_dir, file_name)

    gamma_2_a = params["eta"] * params["gamma_1_a"]
    gamma_2_b = params["eta"] * params["gamma_1_b"]

    try:
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Single mode sheet
            if mode_props_a is not None and mode_props_b is not None:
                df_single = pd.DataFrame(
                    {
                        "Mode": ["Mode a", "Mode b"],
                        "Wavelength (nm)": [
                            mode_props_a["wavelength"],
                            mode_props_b["wavelength"],
                        ],
                        "Linewidth (meV)": [
                            mode_props_a["linewidth"],
                            mode_props_b["linewidth"],
                        ],
                        "Q value": [mode_props_a["Q"], mode_props_b["Q"]],
                        "Reflectance": [
                            mode_props_a["reflection"],
                            mode_props_b["reflection"],
                        ],
                    }
                )
                df_single.to_excel(writer, sheet_name="Single Mode Analysis", index=False)

            # Coupled mode (original modes)
            df_modes = pd.DataFrame(
                {
                    "Mode": ["Mode a", "Mode b"],
                    "Wavelength (nm)": [
                        mode_info["mode_a"]["wavelength"],
                        mode_info["mode_b"]["wavelength"],
                    ],
                    "Linewidth (meV)": [
                        mode_info["mode_a"]["linewidth"],
                        mode_info["mode_b"]["linewidth"],
                    ],
                    "Q value": [
                        mode_info["mode_a"]["Q"],
                        mode_info["mode_b"]["Q"],
                    ],
                    "Reflectance": [
                        mode_info["mode_a"]["reflection"],
                        mode_info["mode_b"]["reflection"],
                    ],
                }
            )
            df_modes.to_excel(writer, sheet_name="Coupled Modes", index=False)

            # Eigenmodes
            df_eigen = pd.DataFrame(
                {
                    "Mode": ["Mode +", "Mode -"],
                    "Wavelength (nm)": [
                        eigenmode_props["mode_plus"]["wavelength"],
                        eigenmode_props["mode_minus"]["wavelength"],
                    ],
                    "Linewidth (meV)": [
                        eigenmode_props["mode_plus"]["linewidth"],
                        eigenmode_props["mode_minus"]["linewidth"],
                    ],
                    "Q value": [
                        eigenmode_props["mode_plus"]["Q"],
                        eigenmode_props["mode_minus"]["Q"],
                    ],
                    "Reflectance": [
                        eigenmode_props["mode_plus"]["reflection_intensity"],
                        eigenmode_props["mode_minus"]["reflection_intensity"],
                    ],
                }
            )
            df_eigen.to_excel(writer, sheet_name="Coupled Eigenmodes", index=False)

            # Dip peaks
            if dip_peaks:
                df_peaks = pd.DataFrame(dip_peaks)
                df_peaks.to_excel(writer, sheet_name="Dip Peaks", index=False)

            # Spectrum
            df_spec = pd.DataFrame(
                {
                    "Wavelength (nm)": params["wavelength_range"],
                    "Reflectance": np.abs(S_r) ** 2,
                    "Phase (rad)": np.angle(S_r),
                }
            )
            df_spec.to_excel(writer, sheet_name="Spectrum", index=False)
        print(f"\nデータをExcelファイルに保存しました: {file_path}")
    except Exception as e:
        print(f"\nエラー: Excelファイルの保存に失敗しました: {e}")
        return None
    return file_path


# ----------------------------------------------------------------------
# Example main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    base_params = {
        "gamma_abs_a": 10,
        "gamma_abs_b": 2,
        "gamma_1_a": 10,
        "gamma_1_b": 1,
        "eta": 0.0001,
        "lambda_a": 800.0,
        "lambda_b": 900.0,
        "g_r": 3,
        "r": 1,
        "t": np.sqrt(1 - 1 ** 2),
        "n": 1.37,
        "l": 1.5e-7,
        "l_g": 0.1e-7,
        "phi_m": -160,
        "wavelength_range": np.linspace(600, 1200, 1200),
    }

    params = base_params.copy()

    S_r_single_a, mode_props_a = calculate_single_mode_reflection(params, mode="a")
    S_r_single_b, mode_props_b = calculate_single_mode_reflection(params, mode="b")
    S_r_coupled, eigenmode_props, mode_info, dip_peaks = calculate_reflection_spectrum(params)

    print("\n" + "=" * 80)
    print("Single Mode A")
    print("=" * 80)
    print(
        f"λ = {mode_props_a['wavelength']:.2f} nm, "
        f"Reflectance = {mode_props_a['reflection']:.4f}, "
        f"Linewidth = {mode_props_a['linewidth']:.3f} meV, Q = {mode_props_a['Q']:.0f}"
    )

    print("\n" + "=" * 80)
    print("Coupled Modes (original)")
    print("=" * 80)
    for name in ["mode_a", "mode_b"]:
        info = mode_info[name]
        print(
            f"{name} : λ = {info['wavelength']:.2f} nm, "
            f"Reflectance = {info['reflection']:.4f}, "
            f"Linewidth = {info['linewidth']:.3f} meV, Q = {info['Q']:.0f}"
        )

    if dip_peaks:
        print("\n" + "=" * 80)
        print("Dip Peaks")
        print("=" * 80)
        for peak in dip_peaks:
            print(
                f"λ = {peak['wavelength']:.2f} nm, "
                f"Absorption = {peak['absorption']:.4f}, "
                f"Δλ = {peak['linewidth']:.2f} nm, Q = {peak['Q']:.0f}"
            )

    save_to_excel(
        params,
        S_r_coupled,
        eigenmode_props,
        mode_info,
        dip_peaks,
        S_r_single_a,
        mode_props_a,
        S_r_single_b,
        mode_props_b,
    )

    plt.figure()
    plt.plot(params["wavelength_range"], np.abs(S_r_coupled) ** 2, label="coupled")
    plt.plot(params["wavelength_range"], np.abs(S_r_single_a) ** 2, label="mode a")
    plt.plot(params["wavelength_range"], np.abs(S_r_single_b) ** 2, label="mode b")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance |S_r|^2")
    plt.legend()
    plt.xlim(600, 1200)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.show()
