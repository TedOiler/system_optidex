import numpy as np
import matplotlib.pyplot as plt


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
