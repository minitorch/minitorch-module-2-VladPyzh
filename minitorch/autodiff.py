from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    left_vals = list(vals)
    left_vals[arg] -= epsilon / 2
    right_vals = list(vals)
    right_vals[arg] += epsilon / 2

    return (f(*right_vals) - f(*left_vals)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    if variable.is_constant():
        assert False, "backward from constant"

    in_degree = defaultdict(int)

    used = set()
    stack = [variable]
    in_degree[variable.unique_id] = 0
    while len(stack):
        cur_var = stack.pop()
        used.add(cur_var.unique_id)

        for var in cur_var.parents:
            if var.is_constant():
                continue
            in_degree[var.unique_id] += 1
            if var.unique_id in used:
                continue
            else:
                stack.append(var)

    in_zeros = [variable]
    ans = [variable]
    while len(in_zeros):
        cur_var = in_zeros.pop()
        for var in cur_var.parents:
            if var.is_constant():
                continue
            else:
                in_degree[var.unique_id] -= 1
                if in_degree[var.unique_id] == 0:
                    ans.append(var)
                    in_zeros.append(var)
    return ans


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leaf nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    derivatives_dict = {variable.unique_id: deriv}

    top_sort = topological_sort(variable)
    for cur_variable in top_sort:
        if cur_variable.is_leaf():
            continue
        var_n_der = cur_variable.chain_rule(derivatives_dict[cur_variable.unique_id])
        for var, der in var_n_der:
            if var.is_leaf():
                var.accumulate_derivative(der)
            else:
                if var.unique_id not in derivatives_dict:
                    derivatives_dict[var.unique_id] = der
                else:
                    derivatives_dict[var.unique_id] += der


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
