import numpy as np
import pandas as pd
import warnings
from typing import Optional, Union, List


class Surv:
    """
    A class to handle different types of survival data.

    Parameters:
    -----------
    time : Union[List[float], np.ndarray]
        The primary time variable.
    time2 : Optional[Union[List[float], np.ndarray]]
        The secondary time variable (optional, based on type).
    event : Optional[Union[List[int], List[bool], np.ndarray]]
        Event occurrence indicators.
    type : str
        Type of survival data ('right', 'left', 'counting', 'interval', 'interval2').
    origin : float
        Origin point for time measurements (default is 0).
    """

    def __init__(
        self,
        time: Union[List[float], np.ndarray],
        time2: Optional[Union[List[float], np.ndarray]] = None,
        event: Optional[Union[List[int], List[bool], np.ndarray]] = None,
        type: str = "right",
        origin: float = 0,
    ) -> None:

        if time is None:
            raise ValueError("Must have a time argument")

        self.origin = origin
        self.type = type
        self.input_attributes = {}

        self.time = np.asarray(time, dtype=float) - origin
        self.nn = len(self.time)

        self.time2 = (
            np.asarray(time2, dtype=float) - origin if time2 is not None else None
        )
        self.event = np.asarray(event) if event is not None else None

        self.process_data()

    def process_data(self) -> None:
        """Process data based on the specified survival type."""

        if self.type in ["right", "left"]:
            if self.event is None and self.time2 is not None:
                self.event = self.time2
                self.time2 = None

            if self.event is None or len(self.event) != self.nn:
                raise ValueError("Time and event/status must be same length")

            status = self.validate_event(self.event)
            self.data = pd.DataFrame({"time": self.time, "status": status})

        elif self.type == "counting":
            if self.time2 is None or self.event is None:
                raise ValueError("Counting type requires time2 and event.")

            if len(self.time2) != self.nn or len(self.event) != self.nn:
                raise ValueError("Start, stop, and event must be same length")

            invalid_times = self.time >= self.time2
            if np.any(invalid_times):
                warnings.warn("Stop time must be > start time; NA created")
                self.time[invalid_times] = np.nan

            status = self.validate_event(self.event)
            self.data = pd.DataFrame(
                {"start": self.time, "stop": self.time2, "status": status}
            )

        elif self.type in ["interval", "interval2"]:
            if self.type == "interval2":
                if self.time2 is None:
                    raise ValueError("interval2 requires time2")

                status = np.where(
                    pd.isna(self.time),
                    2,
                    np.where(
                        pd.isna(self.time2), 0, np.where(self.time == self.time2, 1, 3)
                    ),
                )

                invalid_intervals = (
                    ~pd.isna(self.time)
                    & ~pd.isna(self.time2)
                    & (self.time > self.time2)
                )
                if np.any(invalid_intervals):
                    warnings.warn("Invalid interval (start > stop), NA created")
                    status[invalid_intervals] = np.nan

                self.data = pd.DataFrame(
                    {
                        "time1": self.time,
                        "time2": np.where(status == 3, self.time2, 1),
                        "status": status,
                    }
                )
            else:
                if self.event is None:
                    raise ValueError("Interval type requires event/status")

                valid_status = [0, 1, 2, 3]
                status = np.where(np.isin(self.event, valid_status), self.event, np.nan)

                if np.any(status == 3) and self.time2 is None:
                    raise ValueError("Interval data with status=3 requires time2")

                self.data = pd.DataFrame(
                    {
                        "time1": self.time,
                        "time2": np.where(status == 3, self.time2, 1),
                        "status": status,
                    }
                )

        else:
            raise ValueError(f"Unsupported survival data type: {self.type}")

        self.data.attrs["type"] = self.type

    def validate_event(self, event: np.ndarray) -> np.ndarray:
        """
        Validate and format the event/status data.

        Parameters:
        -----------
        event : np.ndarray
            Event/status indicator array.

        Returns:
        --------
        np.ndarray
            Validated and formatted event/status array.
        """
        if np.issubdtype(event.dtype, np.bool_):
            return event.astype(int)
        elif np.issubdtype(event.dtype, np.number):
            event = event.copy()
            event[np.logical_and(~np.isnan(event), event > 1)] -= 1
            invalid = (~np.isin(event, [0, 1])) & ~np.isnan(event)
            if np.any(invalid):
                warnings.warn("Invalid status value converted to NA")
                event[invalid] = np.nan
            return event
        else:
            raise ValueError("Invalid status value; must be logical or numeric")

    def __repr__(self) -> str:
        return f"Surv(type='{self.type}', data=\n{self.data})"
