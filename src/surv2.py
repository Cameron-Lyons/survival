import numpy as np
import pandas as pd
from typing import Union, List


class Surv2:
    """
    A class to handle survival-type data structured for time-course data.

    Parameters:
    -----------
    time : Union[List[float], np.ndarray]
        Array of time points.
    event : Union[List[str], List[int], np.ndarray]
        Array indicating event occurrence; can be categorical.
    repeated : bool
        Indicates if the data is repeated measures (default False).
    """

    def __init__(
        self,
        time: Union[List[float], np.ndarray],
        event: Union[List[str], List[int], np.ndarray],
        repeated: bool = False,
    ) -> None:

        if time is None:
            raise ValueError("Must have a time argument")

        if not isinstance(repeated, bool):
            raise ValueError("Invalid value for repeated option; must be boolean")

        self.time = np.asarray(time, dtype=float)
        self.nn = len(self.time)

        if event is None:
            raise ValueError("Must have an event argument")

        self.event = pd.Categorical(event)
        if len(self.event) != self.nn:
            raise ValueError("Time and event must have the same length")

        self.states = self.event.categories[1:].tolist()
        if any(pd.isna(self.states)) or "" in self.states:
            raise ValueError("Each state must have a non-blank name")

        self.status = self.event.codes.astype(int)
        self.data = pd.DataFrame({"time": self.time, "status": self.status})

        self.data.attrs["states"] = self.states
        self.data.attrs["repeated"] = repeated

    def __repr__(self) -> str:
        return f"Surv2(repeated={self.data.attrs['repeated']}, states={self.states}, data=\n{self.data})"

    def __str__(self) -> str:
        temp = self.data["status"]
        end = ["+"] + [f":{state}" for state in self.states]
        status_labels = [end[s + 1] if not pd.isna(s) else "?" for s in temp]
        return "\n".join([f"{t}{s}" for t, s in zip(self.data["time"], status_labels)])

    def __getitem__(self, idx: Union[int, slice, List[int]]) -> "Surv2":
        new_time = self.time[idx]
        new_event = self.event[idx]
        return Surv2(new_time, new_event, self.data.attrs["repeated"])

    def is_na(self) -> np.ndarray:
        """Check for NA values in data."""
        return pd.isna(self.data).any(axis=1).to_numpy()

    def to_matrix(self) -> np.ndarray:
        """Convert Surv2 data to matrix format."""
        return self.data.to_numpy()
