import torch

def calculate_B_spline_basis_functions(x, grid, k):
    """
    Args:
        x: torch.Tensor (num splines, num samples)
        grid: torch.Tensor (num splines, num knots)
        k: int (degree of the spline)
    Returns:
        B: torch.Tensor (num splines, num knots + k - 1, num samples)
    """
    # Extend k grid points to the left and right
    num_knots = grid.shape[1]
    distance = (grid[:, -1] - grid[:, 0]) / (num_knots-1) # (num splines, )
    for _ in range(1, k + 1):
        left_extension = grid[:, 0] - distance
        right_extension = grid[:, -1] + distance
        grid = torch.cat((left_extension.unsqueeze(1), grid, right_extension.unsqueeze(1)), dim=1)
    # Calculate B-spline basis functions
    x = x.unsqueeze(dim=1) # (num splines, 1, num samples)
    grid = grid.unsqueeze(dim=2) # (num splines, num knots + 2k, 1)

    b = (grid[:, :-1] <= x) * (x < grid[:, 1:])
    for p in range(1, k+1):
        b = (x - grid[:, :-(p + 1)]) / (grid[:, p:-1] - grid[:, :-(p + 1)]) * b[:, :-1] + (grid[:, (p + 1):] - x) / (grid[:, (p + 1):] - grid[:, 1:-p]) * b[:, 1:] # (num splines, num knots + 2k - p - 1, num samples)

    return b # (num splines, num knots + k - 1, num samples)


def coef_to_curve(x_eval, coef, grid, k):
    x = x_eval
    """
    Args:
        x: torch.Tensor (num splines, num samples)
        coef: torch.Tensor (num splines, num knots + k - 1)
        grid: torch.Tensor (num splines, num knots)
        k: int (degree of the spline)
    Returns:
        y: torch.Tensor (num splines, num samples)
    """
    b_splines = calculate_B_spline_basis_functions(x, grid, k) # (num splines, num knots + k - 1, num samples)
    y = torch.einsum('ijk,ij->ik', b_splines, coef)
    return y

def curve_to_coef(x_eval, y_eval, grid, k):
    x = x_eval
    y = y_eval
    """
    Args:       
        x: torch.Tensor (num splines, num samples)
        y: torch.Tensor (num splines, num samples)
        grid: torch.Tensor (num splines, num knots)
        k: int (degree of the spline)
    Returns:
        coef: torch.Tensor (num splines, num knots + k - 1)
    """
    b_splines = calculate_B_spline_basis_functions(x, grid, k).permute(0, 2, 1) # (num splines, num knots + k - 1, num samples)
    coef = torch.linalg.lstsq(b_splines, y.unsqueeze(2)).solution # (num splines, num knots + k - 1, 1)
    return coef.squeeze(2)