import numpy as np
from typing import Optional
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# todo: how to tell when it has reached steady state? measure slopes of the curves and see if they are <.001 ?
# k ~ 0.1-10, K ~.001-100
def system_dynamics(pop0, t, components, activations, inhibitions, k_cat, K_hill, inpt=.5, basal=.5, input_node=0,):
    # todo: does E and F stay the same the entire simulation
    E = F = basal
    # number of components
    # store the updated population
    pop_update = np.zeros_like(pop0)
    # store the activation and inhibition terms
    for i in range(0, components):
        activation_term = 0
        inhibition_term = 0
        # Identify components activated by component i.
        activated_by = activations[np.where(activations[:,1] == i)[0]][:,0] if len(activations) > 0 else np.array([])
        # Identify components inhibited by component i.
        inhibited_by = inhibitions[np.where(inhibitions[:,1] == i)[0]][:,0] if len(inhibitions) > 0 else np.array([])

        # add the input term here if not added above
        if i == 0:
            activation_term += inpt * k_cat[0, i, -1] * ((1 - pop0[i]) / ((1 - pop0[i]) + K_hill[0, i, -1]))
        # Calculate the activation term if no components activate the current component i.
        elif len(activated_by) == 0 and i > input_node:
            activation_term += E * k_cat[0, i, -1] + (1-pop0[i])/ ((1-pop0[i]) + K_hill[0, i, -1])
        # Calculate the inhibition term if no components inhibit the current component i.
        if len(inhibited_by) == 0:
            inhibition_term += F * k_cat[1, i, -1] + pop0[i]/ (pop0[i]+K_hill[1, i, -1])
        # For each component that activates i, compute contribution to activation term.
        if len(activated_by) > 0:
            for act in activated_by:
                activation_term += pop0[act] * k_cat[0, i, act] * ((1-pop0[i]) / ((1-pop0[i]) + K_hill[0, i, act]))
        # For each component that inhibits i, compute contribution to inhibition term.
        if len(inhibited_by) > 0:
            for inh in inhibited_by:
                inhibition_term += pop0[inh] * k_cat[1, i, inh] * (pop0[i] / (pop0[i] + K_hill[1, i, inh]))
        # Update the population of component i by calculating the net effect of activation and inhibition.
        pop_update[i] = activation_term.copy() - inhibition_term.copy()

    return pop_update


def solve_dynamics(pop0, time, components, activations, inhibitions, k_cat, K_hill, inpt):
    # todo: maybe cant have input as an array ...
    result = odeint(system_dynamics, pop0, time,
                    args=(components, activations, inhibitions, k_cat, K_hill, inpt))
    return result


def check_steady_state(out, thresh=.001):
    # todo: compute the slope of the last 10 values and see if they are <.001
    pass


# todo: sample with latin hypercube sampling
def sample_ode_params(components, seed):
    m = components
    # input takes the same index as E or F
    # k_F and k_E terms are at the end of the k_cat and K_thresh matrices
    k_cat = seed.uniform(0.1, 10, size=(2, m+1, m+1))
    K_thresh = seed.uniform(0.001, 100, size=(2, m+1, m+1))

    return k_cat, K_thresh


def make_input_vals(init_val=.5, perc_increase=.2):
    """returns a tuple of input vals"""
    inpt = (init_val, init_val+init_val*perc_increase)
    return inpt

# todo: smooth the values instead of min/max
def compute_precision_sensitivity(inpt, out, out2, out_node=-1):
    """
    Precision: abs(((O_2 - O_1)/ O_1) / (I_2 - I_1)/I_1 )^-1
    Sensitivity: abs(((O_peak - O_1)/O_1) / (I_2 - I_1)/I_1)
    """
    # end of O_1 is steady state
    O_1 = out[:,out_node][-1]
    O_2 = out2[:,out_node][-20:].min()
    O_peak = out2[:,out_node].max()
    I_1 = inpt[0]
    I_2 = inpt[1]
    precision = abs(((O_2 - O_1)/ O_1) / (I_2 - I_1)/I_1 )**-1
    sensitivity = abs(((O_peak - O_1)/O_1) / (I_2 - I_1)/I_1)
    return precision, sensitivity

# plt.plot(out[:,0], '.')
# plt.plot(out[:,1], '.')
# plt.plot(out[:,2], '.')
# plt.show()
#
# plt.plot(out2[:,0], '.')
# plt.plot(out2[:,1], '.')
# plt.plot(out2[:,2], '.')
# plt.show()


class ODESolve:
    def __init__(
            self,
            seed,
            n_species,
            activations,
            inhibitions,
            dt,
            nt,
            max_iter_per_timestep,  # makes sure it doesn't run forever, not implemented yet
    ):
        self.activations = activations
        self.inhibitions = inhibitions
        self.n_species = n_species
        self.stabilize_tp = dt * np.arange(nt)
        self.eval_tp = dt * np.arange(2*nt)
        # todo: need to reach steady state and then add the step function -- track the slope of the counts ?
        self.inpt = make_input_vals(perc_increase=.2)

        if seed is not None:
            self.rg = np.random.default_rng(seed)

        self.k_cat, self.K_thresh = sample_ode_params(n_species, self.rg)
        self.pop0 = self.rg.uniform(0, 1, self.n_species)

    def run_ode_with_params(self):
        out = solve_dynamics(self.pop0, self.stabilize_tp, self.n_species, self.activations, self.inhibitions, self.k_cat, self.K_thresh,
                             self.inpt[0])
        # self.track_pop = out
        # todo: check oscillation -- O_peak1 > 2 *O_peak2
        # check if steady state is reached
        check_steady_state(out)

        out2 = solve_dynamics(out[-1], self.eval_tp, self.n_species, self.activations, self.inhibitions, self.k_cat, self.K_thresh,
                             self.inpt[1])

        # todo: check if there is adaptation
        precision, sensitivity = compute_precision_sensitivity(self.inpt, out, out2)
        # todo: incorporate some saving
        self.track_pop = np.concatenate([out, out2])

        if np.log10(precision) >= 1 and np.log10(sensitivity) >= 0:
            return True
        else:
            return False

    def initialize_ode_params(self, n_species):
        self.k_cat, self.K_thresh = sample_ode_params(n_species, self.rg)

    def save_dynamics(self):
        # todo: include plotting
        pass









# todo: need to add this dynamically, like check when the integration is stable and then add the step function
# def make_input(time_points, step, init_val=.5, perc_increase=.2):
#     """Input is a step function that increases by perc_increase at step*len(time_points)"""
#     inpt = np.full(len(time_points), init_val)
#     start_index = int(len(time_points) * step)
#     inpt[start_index:] = init_val + init_val * perc_increase
#     return inpt
