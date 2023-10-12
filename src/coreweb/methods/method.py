# -*- coding: utf-8 -*-

"""method.py

Implements the base method.
"""

import datetime
import os
import pickle
from typing import Any, Optional, Sequence, Type, Union

from heliosat.util import sanitize_dt

from ..model import SimulationBlackBox


class BaseMethod(object):
    """
    Class(object) used for fitting.

    Arguments:
        path   Optional     where to save
    Returns:
        None

    Functions:
        add_observer
        initialize
        save
        load
    """
    
    dt_0: datetime.datetime
    locked: bool
    model: Type[SimulationBlackBox]
    model_obj: SimulationBlackBox
    model_kwargs: Optional[dict]
    observers: list

    def __init__(self, path: Optional[str] = None) -> None:
        if path:
            self.locked = True
            self.load(path)
        else:
            self.locked = False

    def add_observer(
        self,
        observer: str,
        dt: Union[str, datetime.datetime, Sequence[str], Sequence[datetime.datetime]],
        dt_s: Union[str, datetime.datetime],
        dt_e: Union[str, datetime.datetime],
        dt_shift: datetime.timedelta = None,
    ) -> None:
        
        """
        Appends an observer.
    
        Arguments:
            observer          name of the observer
            dt                datetime points to be used for fitting
            dt_s              reference point prior to the fluxrope
            dt_e              reference point after the fluxrope
            dt_shift          datetime can be shifted
    
        Returns:
            None
        """
        
        self.observers.append(
            [observer, sanitize_dt(dt), sanitize_dt(dt_s), sanitize_dt(dt_e), dt_shift]
        ) # append observer with sanitized datetime (HelioSat sanitize deals with different timezones)


    def initialize(
        self,
        dt_0: Union[str, datetime.datetime],
        model: Type[SimulationBlackBox],
        model_kwargs: dict = {},
    ) -> None:
        
        """
        Initializes the Fitter.
        Sets the following properties for self:
            dt_0              sanitized launch time
            observers         empty list
            reference_frame   reference frame to work in
    
        Arguments:
            dt_0              launch time
            model             fluxrope model
            model_kwargs      dictionary containing all kwargs for the model
    
        Returns:
            None
        """
        
        if self.locked:
            raise RuntimeError("is locked")

        self.dt_0 = sanitize_dt(dt_0)
        self.model_kwargs = model_kwargs
        self.observers = []
        self.model = model

    def save(self, path: str, **kwargs: Any) -> None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        data = {
            attr: getattr(self, attr)
            for attr in self.__dict__
            if not callable(attr) and not attr.startswith("_")
        }

        for k, v in kwargs.items():
            data[k] = v

        with open(path, "wb") as fh:
            pickle.dump(data, fh)

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)

        for k, v in data.items():
            setattr(self, k, v)

    def run(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()
