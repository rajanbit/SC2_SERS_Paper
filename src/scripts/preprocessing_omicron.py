import glob
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from numpy.polynomial.polynomial import Polynomial

class preprocess:

	def __init__(self, filepaths):
		self.spectra_files = glob.glob(f'{filepaths}/**/*.txt', recursive=True)

	def crop_spectrum(self, wavelengths, intensities, region=(351, 1850)):
		mask = (wavelengths >= region[0]) & (wavelengths <= region[1])
		return wavelengths[mask], intensities[mask]

	def whitaker_hayes_despike(self, intensities, threshold=3):
		mean_intensity = np.mean(intensities)
		std_intensity = np.std(intensities)
		spikes = np.abs(intensities - mean_intensity) > threshold * std_intensity
		corrected_intensities = intensities.copy()
		corrected_intensities[spikes] = mean_intensity
		return corrected_intensities

	def savgol_denoise(self, intensities, window_length=9, polyorder=3):
		return savgol_filter(intensities, window_length, polyorder)

	def aspls_baseline(self, intensities, lam=1e5, p=0.01):
		L = len(intensities)
		D = csc_matrix(np.diff(np.eye(L), 2))
		I = csc_matrix(np.eye(L))
		w = np.ones(L)
		DTD = D.T @ D
		for i in range(10):
			W = spsolve(I + lam * DTD, w[:L-2] * intensities[:L-2])
			w = p * (intensities[:L-2] > W) + (1-p) * (intensities[:L-2] <= W)
		W = W.flatten()
		last_two_intensities = intensities[-2:].reshape(-1, 1).flatten()
		baseline_corrected_intensities = np.concatenate([W, last_two_intensities])
		return baseline_corrected_intensities

	def polynomial_baseline_correction(self, wavelengths, intensities, degree=3):
		p = Polynomial.fit(wavelengths, intensities, deg=degree)
		baseline = p(wavelengths)
		corrected = intensities - baseline
		return corrected

	def min_max_normalise(self, intensities):
		scaler = MinMaxScaler()
		intensities_2d = intensities.reshape(-1, 1)
		normalized_intensities = scaler.fit_transform(intensities_2d).flatten()
		return normalized_intensities

	def preprocess_pipeline(self, wavelengths, intensities):
		wavelengths, intensities = self.crop_spectrum(wavelengths, intensities)
		intensities = self.whitaker_hayes_despike(intensities)
		intensities = self.savgol_denoise(intensities)
		intensities = self.polynomial_baseline_correction(wavelengths, intensities)
		intensities = self.min_max_normalise(intensities)
		return pd.DataFrame({'wavelength': wavelengths, 'intensity': intensities})

	def denoise(self, wavelengths, intensities):
		processed_df = self.preprocess_pipeline(wavelengths, intensities)
		processed_df = processed_df.loc[(processed_df["wavelength"] >= 401) & (processed_df["wavelength"] <= 1800)]
		return processed_df

	def spectra2df(self):
		spectra_array = []
		class_array = []
		sample_array = []
		spectra_name = []

		for i in range(len(self.spectra_files)):
			if self.spectra_files[i].split("/")[2] == "Omicron":
				df = pd.read_csv(self.spectra_files[i], delimiter=',', header=None, names=["wavelength", "intensity"], skiprows=2)
				df = self.denoise(df["wavelength"].values, df["intensity"].values)
				spectra_array.append(df["intensity"].to_numpy())
				class_array.append(self.spectra_files[i].split("/")[4])
				sample_array.append(f'{self.spectra_files[i].split("/")[-2]}/{self.spectra_files[i].split("/")[-1]}')
				if i == 0:
					spectra_name = df["wavelength"].to_numpy()
		spectra_df = pd.DataFrame(spectra_array, columns=spectra_name)
		spectra_df["Class"] = class_array
		spectra_df["Sample"] = sample_array
		return spectra_df
