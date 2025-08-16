import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import block_diag
from abc import ABC, abstractmethod
import tensorflow as tf
from itertools import combinations_with_replacement
import random
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.backend import clear_session
import gc
from scipy.stats import qmc
from skopt import gp_minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from pathlib import Path
import sys
from time import time


def plot_design(design, basis_list, runs, t_detail=100, style='seaborn-v0_8', sub_x=2, sub_y=2, colour="#f0de89", name=None, figsize=(15, 10)):
    def split_array_by_columns(arr, col_splits):
        arrays = []
        start_col = 0
        for num_cols in col_splits:
            end_col = start_col + num_cols
            sub_array = arr[:, start_col:end_col]
            arrays.append(sub_array)
            start_col = end_col
        return arrays

    plt.style.use(style)
    t_values = np.linspace(0, 1, t_detail)
    coefficients_split = split_array_by_columns(design, [basis_list[i].num_basis() for i in range(len(basis_list))])
    y_values = [basis_list[j].evaluate_combination(coefficients_split[j][i], t_values)
                for i in range(runs)
                for j in range(len(coefficients_split))]
    y_values = np.array(y_values).reshape(runs, len(basis_list), t_detail)

    for i in range(y_values.shape[1]):  # i bases
        # set figure size to be big

        fig, axes = plt.subplots(sub_x, sub_y, figsize=figsize)
        axes = axes.flatten()

        # Calculate breakpoints for the current basis
        num_basis = basis_list[i].num_basis()
        degree_basis = basis_list[i].degree
        break_points = np.linspace(0, 1, (num_basis + degree_basis + 1) - 2 * (degree_basis + 1) + 2) # TODO work this out correctly now I think it's wrong.

        # Initialize a list to store y-values at breakpoints for all runs
        y_break_points = []

        # Evaluate the combination of basis functions at breakpoints for each run
        for j in range(runs):
            coeffs = coefficients_split[i][j]
            y_bp = basis_list[i].evaluate_combination(coeffs, break_points)
            y_break_points.append(y_bp)

        y_break_points = np.array(y_break_points)  # Shape: (runs, num_basis)

        for j in range(y_values.shape[0]):  # j runs
            axes[j].plot(t_values, y_values[j][i])
            axes[j].plot(break_points, y_break_points[j], 'o', color=colour, markersize=4)
            axes[j].set_title(f'Run {j + 1}')
            axes[j].set_xlabel('t')
            axes[j].grid(False)
            axes[j].set_ylim(-1.3, 1.3)
        fig.suptitle(f'Functional Factor {i + 1} ', fontsize=16)
        plt.tight_layout()
        if name is not None:
            plt.savefig(f"{name}.png")
        plt.show()
        plt.close()

class Basis:
    def evaluate_basis_function(self, i, t):
        raise NotImplementedError("The method evaluate_basis_function must be implemented by the subclass.")

    def get_basis_support(self, i):
        raise NotImplementedError("The method get_basis_support must be implemented by the subclass.")

    def num_basis(self):
        raise NotImplementedError("The method num_basis must be implemented by the subclass.")

    def evaluate_combination(self, coefficients, t_values):
        num_basis = self.num_basis()
        if len(coefficients) != num_basis:
            raise ValueError("Length of coefficients must match number of basis functions.")
        result = np.zeros_like(t_values, dtype=float)
        for i in range(num_basis):
            basis_values = np.array([self.evaluate_basis_function(i, t) for t in t_values])
            result += coefficients[i] * basis_values
        return result

    def plot_basis_functions(self, t_values):
        num_basis = self.num_basis()
        plt.style.use('fivethirtyeight')
        for i in range(num_basis):
            basis_values = np.array([self.evaluate_basis_function(i, t) for t in t_values])
            plt.plot(t_values, basis_values)
        plt.title(f'{self.__class__.__name__} Functions')
        plt.xlabel('t')
        plt.ylabel('basis function value')
        plt.grid(False)
        plt.ylim(-1.2, 1.2)
        plt.show()

    def plot_experimental_run(self, t_values, coefficients):
        s_values_fourier = self.evaluate_combination(coefficients, t_values)

        # Plot the combined Fourier function
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(8, 6))
        plt.plot(t_values, s_values_fourier)
        plt.title(f'Experimental run of {self.__class__.__name__}')
        plt.xlabel('t')
        plt.ylabel('Combined Function Value')
        plt.grid(False)
        plt.ylim(-1.2, 1.2)
        plt.show()

class BSplineBasis(Basis):
    def __init__(self, degree, total_knots_num):
        self.degree = degree
        self.order = degree + 1
        self.internal_knots_num = total_knots_num

        self.internal_knots = np.linspace(0, 1, self.internal_knots_num)[1:-1]
        self.lower_bound_knots = np.zeros(self.order)
        self.upper_bound_knots = np.ones(self.order)
        self.augmented_knots = np.concatenate((self.lower_bound_knots, self.internal_knots, self.upper_bound_knots))

        self.num_basis_functions = len(self.augmented_knots) - self.order

    def evaluate_basis_function(self, i, t):
        return self._evaluate_bspline_basis_function(i, self.degree, t)

    def _evaluate_bspline_basis_function(self, i, k, t):
        knots = self.augmented_knots
        if k == 0:
            if knots[i] <= t < knots[i + 1]:
                return 1.0
            elif t == knots[-1] and t == knots[i + 1]:
                return 1.0  # Handle special case at the end of the knot vector
            else:
                return 0.0
        else:
            denom1 = knots[i + k] - knots[i]
            term1 = 0.0
            if denom1 != 0:
                term1 = ((t - knots[i]) / denom1) * self._evaluate_bspline_basis_function(i, k - 1, t)

            denom2 = knots[i + k + 1] - knots[i + 1]
            term2 = 0.0
            if denom2 != 0:
                term2 = ((knots[i + k + 1] - t) / denom2) * self._evaluate_bspline_basis_function(i + 1, k - 1, t)

            return term1 + term2

    def get_basis_support(self, i):
        start = self.augmented_knots[i]
        end = self.augmented_knots[i + self.order]
        return start, end

    def num_basis(self):
        return self.num_basis_functions

class FourierBasis(Basis):
    def __init__(self, num_basis_functions):
        self.num_basis_functions = num_basis_functions

    def evaluate_basis_function(self, i, t):
        if i == 0:
            return 1.0  # Constant term
        elif i % 2 == 1:
            n = (i + 1) // 2
            return np.sqrt(2) * np.sin(2 * np.pi * n * t)
        else:
            n = i // 2
            return np.sqrt(2) * np.cos(2 * np.pi * n * t)

    def get_basis_support(self, i):
        return 0.0, 1.0

    def num_basis(self):
        return self.num_basis_functions

class PolynomialBasis(Basis):
    def __init__(self, degree):
        self.degrees = list(range(degree))
        self.num_basis_functions = len(self.degrees)

    def evaluate_basis_function(self, i, t):
        return t ** self.degrees[i]

    def get_basis_support(self, i):
        return 0.0, 1.0  # Polynomials are defined over [0,1]

    def num_basis(self):
        return self.num_basis_functions

class JMatrix:
    def __init__(self, basis_pairs):
        self.basis_pairs = basis_pairs

    @staticmethod
    def _compute_element(basis1, basis2):
        num_basis1 = basis1.num_basis()
        num_basis2 = basis2.num_basis()
        J = np.zeros((num_basis1, num_basis2))

        for i in range(num_basis1):
            for j in range(num_basis2):
                # Determine overlapping support
                a1, b1 = basis1.get_basis_support(i)
                a2, b2 = basis2.get_basis_support(j)
                a = max(a1, a2)
                b = min(b1, b2)
                if a >= b:
                    J[i, j] = 0.0
                    continue

                def integrand(t):
                    return basis1.evaluate_basis_function(i, t) * basis2.evaluate_basis_function(j, t)

                integral_value, _ = quad(integrand, a, b, limit=1000)
                J[i, j] = integral_value

        return J

    def compute(self):
        J_blocks = []
        for basis1, basis2 in self.basis_pairs:
            J_element = self._compute_element(basis1, basis2)
            J_blocks.append(J_element)

        J = block_diag(*J_blocks)
        return J

class BaseModel(ABC):
    @abstractmethod
    def compute_objective(self, *args, **kwargs):
        """Compute the objective function for the model."""
        pass

class ScalarOnFunctionModel(BaseModel):
    def __init__(self, bases_pairs=None):

        self.basis_pairs = bases_pairs
        self.J = self.compute_J()
        self.Kx = self.J.shape[0]
        self.Kb = self.J.shape[1]

    def __str__(self):
        return (f"ScalarOnFunctionModel(Kx={self.Kx}, Kb={self.Kb}, "
                f"Kx_family='{self.Kx_family}', Kb_family='{self.Kb_family}', "
                f"k_degree={self.k_degree}, knots_num={self.knots_num})")

    def __repr__(self):
        return (f"ScalarOnFunctionModel(Kx={self.Kx}, Kb={self.Kb}, "
                f"Kx_family='{self.Kx_family}', Kb_family='{self.Kb_family}', "
                f"k_degree={self.k_degree}, knots_num={self.knots_num})")

    def Covar(self, X, m, library='numpy'):
        if library == 'numpy':
            ones = np.ones((X.shape[0], 1))
            Zetta = np.concatenate((ones, X @ self.J), axis=1)
            Covar = Zetta.T @ Zetta
            return Covar
        elif library == 'tensorflow':
            batch_size = tf.shape(X)[0]
            ones = tf.ones((batch_size, m, 1))
            X = tf.reshape(X, (-1, m, self.Kx))
            Z = tf.concat([ones, tf.matmul(X, self.J)], axis=2)
            Covar = tf.matmul(Z, Z, transpose_a=True)
            return Covar

    def compute_objective(self, Gamma):
        Covar = self.Covar(Gamma, library='numpy')
        try:
            P_inv = np.linalg.inv(Covar)
        except np.linalg.LinAlgError:
            return np.nan

        value = np.trace(P_inv)

        return value

    def compute_objective_input(self, x, i, j, Gamma):
        Gamma[i, j] = x
        return self.compute_objective(Gamma)

    def compute_objective_tf(self, X, m, n):
        Covar = self.Covar(X, m, library='tensorflow')
        batch_size = tf.shape(X)[0]
        ones = tf.ones((batch_size, m, 1))
        X = tf.reshape(X, (-1, m, n))
        Z = tf.concat([ones, tf.matmul(X, self.J)], axis=2)
        Z_t_Z = tf.matmul(Z, Z, transpose_a=True)

        det = tf.linalg.det(Z_t_Z)
        epsilon = 1e-6  # TODO: affects results significantly!! need to find out how to set it dynamically.
        condition = tf.abs(det)[:, None, None] < epsilon

        diagonal = tf.linalg.diag_part(Z_t_Z) + epsilon
        Z_t_Z_epsilon = Z_t_Z + tf.linalg.diag(diagonal - tf.linalg.diag_part(Z_t_Z))
        Z_t_Z_regularized = tf.where(condition, Z_t_Z_epsilon, Z_t_Z)

        M = tf.linalg.inv(Z_t_Z_regularized)
        value = tf.linalg.trace(M)
        return tf.where(value < 0, tf.constant(1e10), value)

    def compute_objective_bo(self, X, m, n):
        ones = np.ones((m, 1)).reshape(-1, 1)
        X = np.array(X).reshape(m, n)
        Z = np.hstack((ones, X @ self.J))
        try:
            M = np.linalg.inv(Z.T @ Z)
        except np.linalg.LinAlgError:
            return 1e10
        result = np.trace(M)
        return 1e10 if result < 0 else result

    def compute_J(self):
        return JMatrix(self.basis_pairs).compute()

class BaseOptimizer(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def optimize(self, *args, **kwargs):
        """Optimize the model's design matrix to meet specific criteria."""
        pass

class NBDO:
    def __init__(self, model, latent_dim,
                 base=2, max_layers=None, alpha=0.0,
                 latent_space_activation='tanh', output_layer_activation='tanh'):
        self.model = model
        self.runs = None
        # self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.base = base
        self.max_layers = max_layers
        self.alpha = alpha
        self.latent_space_activation = latent_space_activation
        self.output_space_activation = output_layer_activation

        self.input_dim = None
        self.encoder = None
        self.latent = None
        self.decoder = None
        self.autoencoder = None
        self.num_layers = None

        self.input_layer = None
        self.output_layer = None

        self.train_set = None
        self.val_set = None
        self.history = None

        self.optimal_latent_var = None
        self.optimal_cr = None
        self.optimal_des = None
        self.search_history = None
        self.eval_history = None

    def __repr__(self):
        return f"NBDO(\n" \
               f"  model: {self.model.__class__.__name__},\n" \
               f"  max_layers: {self.max_layers},\n" \
               f"  latent_dim: {self.latent_dim},\n" \
               f"  input_dim: {self.input_dim},\n" \
               f"  num_layers: {self.num_layers},\n" \
               f"  train_set: {self.train_set.shape if self.train_set is not None else None},\n" \
               f"  val_set: {self.val_set.shape if self.val_set is not None else None}\n" \
               f"  base: {self.base},\n" \
               f"  alpha: {self.alpha},\n" \
               f"  latent_space_activation: {self.latent_space_activation},\n" \
               f"  output_space_activation: {self.output_space_activation},\n" \
               f")"

    def __str__(self):
        return f"NBDO Model Summary:\n" \
               f"  Model Type: {self.model.__class__.__name__}\n" \
               f"  --------------------------------------------\n" \
               f"  Max Dimension: {self.max_layers}\n" \
               f"  Input Dimension: {self.input_dim}\n" \
               f"  Latent Dimension: {self.latent_dim}\n" \
               f"  Number of Layers: {self.num_layers}\n" \
               f"  --------------------------------------------\n" \
               f"  Training Set Size: {self.train_set.shape[0] if self.train_set is not None else None}\n" \
               f"  Validation Set Size: {self.val_set.shape[0] if self.val_set is not None else None}\n" \
               f"  --------------------------------------------\n" \
               f"  Base: {self.base}\n" \
               f"  Alpha: {self.alpha}\n" \
               f"  Latent Space Activation: {self.latent_space_activation}\n" \
               f"  Output Space Activation: {self.output_space_activation}"

    def _build_encoder(self):

        self.num_layers = int(np.log(self.input_dim / self.latent_dim) / np.log(self.base))
        self.num_layers = min(self.num_layers, self.max_layers) if self.max_layers is not None else self.num_layers

        self.input_layer = Input(shape=(self.input_dim,))
        encoder = self.input_layer
        for layer in range(self.num_layers):
            n_neurons = int(self.input_dim / (self.base ** (layer + 1)))
            encoder = Dense(n_neurons, activation=LeakyReLU(alpha=self.alpha))(encoder)

        latent = Dense(self.latent_dim, activation=self.latent_space_activation, name='latent')(encoder)
        self.encoder = Model(self.input_layer, latent, name='encoder')

    def _build_decoder(self):

        latent_inputs = Input(shape=(self.latent_dim,))
        decoder = latent_inputs
        for layer in range(self.num_layers, 0, -1):
            n_neurons = int(self.input_dim / self.base ** layer)
            decoder = Dense(n_neurons, activation=LeakyReLU(alpha=self.alpha))(decoder)
        self.output_layer = Dense(self.input_dim, activation=self.output_space_activation)(decoder)
        self.decoder = Model(latent_inputs, self.output_layer, name='decoder')

    def _build_autoencoder(self):
        self._build_encoder()
        self._build_decoder()

        autoencoder_input = self.input_layer
        latent_representation = self.encoder(autoencoder_input)
        autoencoder_output = self.decoder(latent_representation)

        self.autoencoder = Model(autoencoder_input, autoencoder_output, name='autoencoder')

    def _get_custom_loss(self):
        if isinstance(self.model, ScalarOnFunctionModel):
            def custom_loss(y_true, y_pred):
                reconstruction_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
                m = self.runs
                n = self.model.Kx
                objective_value = self.model.compute_objective_tf(y_pred, m, n)
                return objective_value

            return custom_loss
        elif isinstance(self.model, FunctionOnFunctionModel):
            def custom_loss(y_true, y_pred):
                m = self.runs
                n = self.model.Kx
                objective_value = self.model.compute_objective_tf(y_pred, m, n)
                return objective_value
            return custom_loss
        elif isinstance(self.model, ScalarOnScalarModel):
            def custom_loss(y_true, y_pred):
                reconstruction_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
                m = self.runs
                n = self.model.Kx[0]
                objective_value = self.model.compute_objective_tf(y_pred, m, n)
                return objective_value
            return custom_loss

    def compute_train_set(self, num_designs, runs, epsilon=1e-10, type='random'):
        def model_matrix_columns(n, p):
            from math import comb
            return comb(n + p, p)

        self.runs = runs
        design_matrix = []
        valid_count = 0

        # Determine number of raw input features (NOT the number of model matrix terms)
        if isinstance(self.model, ScalarOnScalarModel):
            n_features = self.model.Kx[0]
        elif isinstance(self.model, ScalarOnFunctionModel):
            n_features = self.model.Kx
        elif isinstance(self.model, FunctionOnFunctionModel):
            n_features = self.model.Kx
        else:
            raise ValueError("Unsupported model type.")

        max_attempts = int(1e6)

        for attempt in range(max_attempts):
            if valid_count == num_designs:
                break

            # Generate candidate design of shape (runs, n_features)
            if type == 'random':
                candidate_matrix = np.random.uniform(-1, 1, size=(runs, n_features))
            elif type == 'LHC':
                sampler = qmc.LatinHypercube(d=n_features)
                lhs_sample = sampler.random(n=runs)
                candidate_matrix = 2 * lhs_sample - 1  # scale to [-1, 1]
            else:
                raise ValueError("type must be 'random' or 'LHC'")

            # Compute information matrix for validation
            try:
                if isinstance(self.model, ScalarOnScalarModel):
                    ZtZ = self.model.calc_covar_matrix(candidate_matrix)
                elif isinstance(self.model, ScalarOnFunctionModel):
                    Z = np.hstack((np.ones((runs, 1)), candidate_matrix @ self.model.J))
                    ZtZ = Z.T @ Z
                elif isinstance(self.model, FunctionOnFunctionModel):
                    Gamma = np.hstack((np.ones((runs, 1)), candidate_matrix))
                    Z = np.matmul(Gamma, self.model.J)
                    ZtZ = Z.T @ Z
                else:
                    continue

                # Accept design if well-conditioned or if explicitly ScalarOnScalar (always accepted)
                if np.linalg.det(ZtZ) > epsilon or isinstance(self.model, ScalarOnScalarModel) or isinstance(self.model, ScalarOnFunctionModel):
                    design_matrix.append(candidate_matrix)
                    valid_count += 1

            except np.linalg.LinAlgError:
                continue

        if valid_count < num_designs:
            raise RuntimeError(f"Only found {valid_count} valid designs after {max_attempts} attempts.")

        # Stack and flatten: (num_designs, runs * n_features)
        reshaped_design_matrix = np.stack(design_matrix).reshape(num_designs, -1)
        self.train_set, self.val_set = train_test_split(reshaped_design_matrix,
                                                        test_size=0.2,
                                                        random_state=42)
        self.input_dim = self.train_set.shape[1]


    def fit(self, epochs, batch_size=32,
            patience=50, optimizer=tf.keras.optimizers.legacy.RMSprop()):
        self._build_autoencoder()
        custom_loss = self._get_custom_loss()
        self.autoencoder.compile(optimizer=optimizer, loss=custom_loss)
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.autoencoder.build(input_shape=(None, self.input_dim))
        self.history = self.autoencoder.fit(self.train_set, self.train_set,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            validation_data=(self.val_set, self.val_set),
                                            callbacks=[early_stopping])

        return self.history

    def optimize(self, n_calls=10, acq_func='EI', acq_optimizer='sampling',
                 n_random_starts=5, verbose=True):

        def objective(latent_var):
            latent_var = np.array(latent_var).reshape(1, -1)
            decoded = self.decoder.predict(latent_var)
            if self.model.__class__.__name__ == 'ScalarOnFunctionModel':
                optimality = self.model.compute_objective_bo(X=decoded, m=self.runs, n=self.model.Kx)
                return optimality
            elif self.model.__class__.__name__ == 'FunctionOnFunctionModel':
                optimality = self.model.compute_objective_bo(decoded, self.runs, self.model.Kx)
                return optimality
            elif self.model.__class__.__name__ == 'ScalarOnScalarModel':
                optimality = self.model.compute_objective_bo(X=decoded, m=self.runs, n=self.model.Kx[0])
                return optimality

        dimensions = [(-1., 1.) for _ in range(self.latent_dim)]
        res = gp_minimize(objective, dimensions, n_calls=n_calls,
                          random_state=42, verbose=verbose, n_jobs=-1,
                          n_random_starts=n_random_starts, acq_func=acq_func, acq_optimizer=acq_optimizer)
        self.optimal_latent_var = res.x
        self.optimal_cr = res.fun
        self.optimal_des = self.decode(np.array(self.optimal_latent_var).reshape(1, -1))
        self.search_history = res.x_iters
        self.eval_history = res.func_vals
        clear_session()
        return self.optimal_cr, self.optimal_des

    def clear_memory(self):
        del self.autoencoder
        del self.encoder
        del self.decoder
        gc.collect()

    def encode(self, design):
        return self.encoder.predict(design.reshape(1, -1))

    def decode(self, latent):
        return self.decoder.predict(latent).reshape(self.runs, -1)


# Timed Block
start_time = time()
N = 12 # runs
x_base = BSplineBasis(degree=0, total_knots_num=4) # C
bs_base = PolynomialBasis(degree=2) # H
bases_pairs = [(x_base, bs_base)] 
s_on_f_model = ScalarOnFunctionModel(bases_pairs=bases_pairs)
optimizer_s_on_f = NBDO(model=s_on_f_model, latent_dim=4)
optimizer_s_on_f.compute_train_set(num_designs=1_000, runs=N)
history = optimizer_s_on_f.fit(epochs=100)
best_cr, best_des = optimizer_s_on_f.optimize()
end_time = time()

# Print statements
print(f"Time taken: {end_time - start_time:.2f} seconds")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print(f"Optimality criterion value: {best_cr}")
print(f"Optimality design: {best_des}")