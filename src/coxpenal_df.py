"""
Degrees of freedom computation for Cox proportional hazards with penalized terms.

This module implements the degrees of freedom computation based on Bob Gray's paper
for Cox proportional hazards models with penalized terms. It handles both sparse
and non-sparse penalty terms.

Original R function: coxpenal.df
"""

from typing import List, Union, Tuple, Optional, Dict, Any, cast
import numpy as np
from numpy.typing import NDArray


def coxpenal_df(
    hmat: NDArray[np.float64],
    hinv: NDArray[np.float64],
    fdiag: NDArray[np.float64],
    assign_list: List[List[int]],
    ptype: int,
    nvar: int,
    pen1: NDArray[np.float64],
    pen2: Union[NDArray[np.float64], float],
    sparse: int
) -> Dict[str, Any]:
    """
    Compute degrees of freedom for Cox proportional hazards with penalized terms.
    
    This function implements the degrees of freedom computation based on Bob Gray's
    paper for Cox proportional hazards models with penalized terms. It handles
    different combinations of sparse and non-sparse penalty terms.
    
    Parameters
    ----------
    hmat : NDArray[np.float64]
        Right hand slice of Cholesky decomposition of H matrix
    hinv : NDArray[np.float64]
        Right hand slice of Cholesky decomposition of H-inverse
    fdiag : NDArray[np.float64]
        Diagonal of D-inverse matrix
    assign_list : List[List[int]]
        Terms information - list of lists containing variable indices for each term
    ptype : int
        Penalty type: 1 or 3 if sparse term exists, 2 or 3 if non-sparse exists
    nvar : int
        Number of non-sparse terms
    pen1 : NDArray[np.float64]
        Penalty matrix (diagonal) for sparse terms
    pen2 : Union[NDArray[np.float64], float]
        Penalty matrix for non-sparse terms
    sparse : int
        Index indicating which term is the sparse one
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'df': degrees of freedom for each term
        - 'trH': trace of H matrix for each term
        - 'var': variance matrix (if applicable)
        - 'var2': variance matrix 2 (if applicable)
        - 'fvar': diagonal of variance matrix for sparse terms (if applicable)
        - 'fvar2': diagonal of variance matrix 2 for sparse terms (if applicable)
        
    Notes
    -----
    The function handles three main cases:
    1. Only sparse terms (ptype == 1 and nvar == 0)
    2. Only dense/non-sparse terms (ptype == 2)
    3. Both sparse and non-sparse terms (ptype == 3)
    """
    
    if ptype == 1 and nvar == 0:
        # Only sparse terms
        hdiag = 1.0 / fdiag
        fvar2 = (hdiag - pen1) * fdiag**2
        df = np.sum((hdiag - pen1) * fdiag)
        trH = np.sum(fdiag)
        
        return {
            'fvar2': fvar2,
            'df': df,
            'fvar': fdiag,
            'trH': trH
        }
    
    elif ptype == 2:
        # Only dense/non-sparse terms
        # Create mask for non-zero fdiag elements
        mask = fdiag != 0
        fdiag_inv = np.where(mask, 1.0 / fdiag, 0.0)
        
        hmat_full = hmat.T @ (fdiag_inv * hmat)
        hinv_full = hinv @ (fdiag * hinv.T)
        
        # Handle different penalty matrix shapes
        if isinstance(pen2, (np.ndarray, list, tuple)):
            pen2_seq = cast(np.ndarray, pen2)
            if len(pen2_seq) == len(hmat_full):
                imat = hmat_full - pen2_seq
            else:
                imat = hmat_full - np.diag(pen2_seq)
        else:
            # pen2 is a float or scalar, broadcast to diagonal
            imat = hmat_full - np.eye(hmat_full.shape[0]) * pen2
        
        var = hinv_full @ imat @ hinv_full
        
        if len(assign_list) == 1:
            df = np.sum(imat * hinv_full)
            trH = np.sum(np.diag(hinv_full))
            return {
                'var2': var,
                'df': df,
                'trH': trH,
                'var': hinv_full
            }
        else:
            df = []
            trH = []
            d2 = np.diag(hinv_full)
            
            for i in assign_list:
                temp = _coxph_wtest(hinv_full[i, i], var[i, i])['solve']
                if isinstance(temp, np.ndarray) and temp.ndim > 1:
                    df.append(np.sum(np.diag(temp)))
                else:
                    df.append(np.sum(temp))
                trH.append(np.sum(d2[i]))
            
            return {
                'var2': var,
                'df': df,
                'trH': trH,
                'var': hinv_full
            }
    
    else:
        # Sparse terms + other variables
        nf = len(fdiag) - nvar
        nr1 = slice(0, nf)
        nr2 = slice(nf, nf + nvar)
        
        d1 = fdiag[nr1]
        d2 = fdiag[nr2]
        temp = hinv[nr1, :].T
        temp2 = hinv[nr2, :].T
        
        A_diag = d1 + np.sum(temp**2 * d2, axis=0)
        B = hinv[nr1, :] @ (d2 * temp2)
        C = hinv[nr2, :] @ (d2 * temp2)  # See notation in paper
        var2 = C - B.T @ (pen1 * B)
        
        if ptype == 3:
            # Additional work when we have penalties on both sparse and non-sparse terms
            mask = fdiag != 0
            fdiag_inv = np.where(mask, 1.0 / fdiag, 0.0)
            hmat_22 = hmat.T @ (fdiag_inv * hmat)
            temp = C - _coxph_wtest(hmat_22, np.eye(nvar))['solve']
            
            if nvar == 1:
                var2 = var2 - C * pen2 * C  # C will be 1 by 1
                temp2_val = temp * pen2
            elif isinstance(pen2, (np.ndarray, list, tuple)):
                pen2_seq = cast(np.ndarray, pen2)
                if len(pen2_seq) == nvar:
                    var2 = var2 - C @ (pen2_seq * C)  # Diagonal penalty
                    temp2_val = np.sum(np.diag(temp) * pen2_seq)
                else:
                    var2 = var2 - C @ np.array(pen2_seq).reshape(nvar, nvar) @ C
                    temp2_val = np.sum(np.diag(temp * pen2_seq))
            else:
                # pen2 is a float or scalar
                var2 = var2 - C @ (np.eye(nvar) * pen2) @ C
                temp2_val = np.sum(np.diag(temp) * pen2)
        else:
            temp2_val = 0  # temp2 contains trace[B'A^{-1}B P2], this line: P2=0
        
        df = []
        trH = []
        cdiag = np.diag(C)
        
        for i in range(len(assign_list)):
            if sparse == i:
                df.append(nf - (np.sum(A_diag * pen1) + temp2_val))
                trH.append(np.sum(A_diag))
            else:
                j = assign_list[i]
                temp = _coxph_wtest(C[j, j], var2[j, j])['solve']
                if isinstance(temp, np.ndarray) and temp.ndim > 1:
                    df.append(np.sum(np.diag(temp)))
                else:
                    df.append(np.sum(temp))
                trH.append(np.sum(cdiag[j]))
        
        return {
            'var': C,
            'df': df,
            'trH': trH,
            'fvar': A_diag,
            'var2': var2
        }


def _coxph_wtest(
    h: NDArray[np.float64],
    v: NDArray[np.float64]
) -> Dict[str, NDArray[np.float64]]:
    """
    Helper function to compute weighted test statistics.
    
    This is a simplified implementation of the R function coxph.wtest.
    In practice, this would need to be implemented based on the specific
    requirements of the Cox model.
    
    Parameters
    ----------
    h : NDArray[np.float64]
        H matrix
    v : NDArray[np.float64]
        V matrix
        
    Returns
    -------
    Dict[str, NDArray[np.float64]]
        Dictionary containing 'solve' key with the solution
    """
    # This is a placeholder implementation
    # In practice, this would solve the system H * x = V
    try:
        solve = np.linalg.solve(h, v)
    except np.linalg.LinAlgError:
        # Handle singular matrix case
        solve = np.linalg.lstsq(h, v, rcond=None)[0]
    
    return {'solve': solve}
