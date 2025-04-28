"""
Bounded link functions for pseudo‐values, analogous to R’s blogit, bcloglog, bprobit, and blog.
"""

from dataclasses import dataclass
from typing import Callable, Union
import numpy as np
from scipy.stats import norm


ArrayLike = Union[float, np.ndarray]


@dataclass
class Link:
    """
    Represents a link function with a name identifier.

    Attributes
    ----------
    link_fun : Callable[[ArrayLike], ArrayLike]
        Function mapping μ in (0,1) (or [0,∞) for log) to the transformed scale.
    name : str
        Name of the link.
    """

    link_fun: Callable[[ArrayLike], ArrayLike]
    name: str


def blogit(edge: float = 0.05) -> Link:
    """
    Bounded logit link: clips μ to [edge, 1-edge] before applying logit.

    Parameters
    ----------
    edge : float, default=0.05
        Lower/upper bound for μ before transformation.

    Returns
    -------
    Link
        A Link object whose `link_fun` computes log(x/(1-x)) on clipped μ.
    """

    def linkfun(mu: ArrayLike) -> ArrayLike:
        x = np.clip(mu, edge, 1.0 - edge)
        return np.log(x / (1.0 - x))

    return Link(link_fun=linkfun, name="blogit")


def bcloglog(edge: float = 0.05) -> Link:
    """
    Bounded complementary log‐log link: clips μ to [edge, 1-edge] before applying cloglog.

    Parameters
    ----------
    edge : float, default=0.05
        Lower/upper bound for μ before transformation.

    Returns
    -------
    Link
        A Link object whose `link_fun` computes log(-log(1-x)) on clipped μ.
    """

    def linkfun(mu: ArrayLike) -> ArrayLike:
        x = np.clip(mu, edge, 1.0 - edge)
        return np.log(-np.log(1.0 - x))

    return Link(link_fun=linkfun, name="bcloglog")


def bprobit(edge: float = 0.05) -> Link:
    """
    Bounded probit link: clips μ to [edge, 1-edge] before applying the probit (inverse normal CDF).

    Parameters
    ----------
    edge : float, default=0.05
        Lower/upper bound for μ before transformation.

    Returns
    -------
    Link
        A Link object whose `link_fun` computes norm.ppf(x) on clipped μ.
    """

    def linkfun(mu: ArrayLike) -> ArrayLike:
        x = np.clip(mu, edge, 1.0 - edge)
        return norm.ppf(x)

    return Link(link_fun=linkfun, name="bprobit")


def blog(edge: float = 0.05) -> Link:
    """
    Bounded log link: clips μ to [edge, ∞) before applying natural log.

    Parameters
    ----------
    edge : float, default=0.05
        Lower bound for μ before transformation.

    Returns
    -------
    Link
    A Link object whose `link_fun` computes log(x) on clipped μ.
    """

    def linkfun(mu: ArrayLike) -> ArrayLike:
        x = np.maximum(mu, edge)
        return np.log(x)

    return Link(link_fun=linkfun, name="blog")
