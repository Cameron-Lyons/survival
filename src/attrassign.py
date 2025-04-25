from typing import Any, Dict, List, Sequence


def attrassign(obj: Any, terms: Sequence[str] = None) -> Dict[str, List[int]]:
    """
    Generic attrassign function.

    Dispatches to `attrassign_lm` if `obj` appears to be a fitted linear model
    (with Patsy/StatsModels-style `design_info` and `exog`), otherwise treats
    `obj` as an R-style “assign” vector and requires `terms` to be provided.

    Args:
        obj: Either
             - a fitted linear‐model result with
               `obj.model.data.design_info` (with `.term_names` and
               `.term_name_slices`) and `obj.model.exog` (the design matrix), or
             - a sequence of integers (“assign” vector) analogous to R’s
               `attr(model_matrix, "assign")`.
        terms: If `obj` is NOT a model, `terms` must be the sequence of term
               labels (analogous to R’s `attr(terms(object), "term.labels")`).

    Returns:
        A dict mapping each term name to the list of zero‐based column indices
        in the model matrix that belong to that term.

    Raises:
        ValueError: if dispatch fails or required data is missing.
    """
    model = getattr(obj, "model", None)
    if model is not None:
        data = getattr(model, "data", None)
        design_info = getattr(data, "design_info", None) if data is not None else None
        exog = getattr(model, "exog", None)
        if design_info is not None and exog is not None:
            return attrassign_lm(obj)

    if terms is None:
        raise ValueError("`terms` must be provided when `obj` is not a model")
    if not isinstance(obj, Sequence):
        raise ValueError("`obj` must be a sequence of integers when not a model")
    return attrassign_default(obj, terms)


def attrassign_lm(model: Any) -> Dict[str, List[int]]:
    """
    Create S‐plus style assign mapping from a fitted linear model.

    This mirrors R’s `attrassign.lm`, extracting the Patsy/StatsModels design
    information to build an R‐style “assign” mapping.

    Args:
        model: A fitted StatsModels OLS (or similar) result object with:
            - `model.exog`: the design (model) matrix, and
            - `model.data.design_info`:
                - `.term_names`: List[str] of terms (excluding intercept),
                - `.term_name_slices`: Dict[str, slice] mapping each term to
                  its columns in the design matrix.

    Returns:
        A dict mapping term names (including `'(Intercept)'`) to lists of
        zero‐based column indices.
    """
    info = model.model.data.design_info
    term_labels: List[str] = info.term_names
    assign: List[int] = []
    all_terms = ["(Intercept)"] + term_labels
    for term_idx, term in enumerate(all_terms):
        sl = info.term_name_slices.get(term)
        if sl is None:
            continue
        assign.extend([term_idx] * (sl.stop - sl.start))

    return attrassign_default(assign, term_labels)


def attrassign_default(
    assign: Sequence[int], term_labels: Sequence[str]
) -> Dict[str, List[int]]:
    """
    Convert an R‐style “assign” vector and term labels into S-plus style output.

    In R:
      temp <- c("(Intercept)", term_labels)[assign + 1]
      split(seq_along(temp), factor(temp, levels=unique(temp)))

    This produces a mapping from each term name to the positions of the
    design‐matrix columns that belong to that term.

    Args:
        assign: Sequence of integers where each entry indicates which term
                (0 = intercept, 1 = first term, etc.) a column belongs to.
        term_labels: List of term labels (as from R’s `terms(...)`).

    Returns:
        A dict where keys are term names (including `'(Intercept)'`) and
        values are lists of zero-based column indices.
    """
    names = ["(Intercept)"] + list(term_labels)
    try:
        term_names = [names[i] for i in assign]
    except Exception as e:
        raise ValueError(f"Invalid assign indices: {e}")

    mapping: Dict[str, List[int]] = {}
    for idx, tname in enumerate(term_names):
        mapping.setdefault(tname, []).append(idx)
    return mapping
