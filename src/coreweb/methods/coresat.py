import datetime as datetime
from typing import Any, List, Optional, Sequence, Type, Union
import numpy as np

class custom_observer(object):

    """Handles custom data and sets the following attributes for self:
            data         dataset
        Arguments:
            data_path    where to find the data
            kwargs       any
        Returns:
            None
        """

    def __init__(self, observer:str, **kwargs: Any) -> None:
        
        self.spacecraft = observer
        
    
    def get(self, dtp: Union[str, datetime.datetime, Sequence[str], Sequence[datetime.datetime]], b_data: List[float], t_data: List[datetime.datetime],  **kwargs: Any) -> np.ndarray:
        
        
        sampling_freq = kwargs.pop("sampling_freq", None)

        if kwargs.pop("as_endpoints", False):
            # Convert dtp to datetime objects if they are strings
            if isinstance(dtp, str):
                dtp = [datetime.datetime.fromisoformat(dtp)]
            elif isinstance(dtp, Sequence) and isinstance(dtp[0], str):
                dtp = [datetime.datetime.fromisoformat(dt_str).replace(tzinfo=None) for dt_str in dtp]

            # Convert t_data to datetime objects without timezone information
            try:
                t_data = [datetime.datetime.fromisoformat(dt_str).replace(tzinfo=None) for dt_str in t_data]
            except:
                t_data = [time for time in t_data]
                
            #print(t_data)

            # Find indices where t_data falls within the specified date range
            start_time, end_time = dtp[0].replace(tzinfo=None), dtp[-1].replace(tzinfo=None)
            indices = [i for i, dt in enumerate(t_data) if start_time <= dt <= end_time]

            # Extract b_data values for the selected indices
            selected_b_data = np.array([b_data[i] for i in indices])
            selected_t_data = np.array([t_data[i] for i in indices])
            
            #print(len(selected_b_data))
            #print(len(selected_t_data))
            #print(sampling_freq)
            
            
            if sampling_freq and len(selected_t_data) > sampling_freq * 6 *2:
                # Return data at the specified sampling frequency
                selected_b_data = selected_b_data[::sampling_freq]
                selected_t_data = selected_t_data[::sampling_freq]

            return selected_t_data, selected_b_data
        
        else:
            # Initialize lists to store selected data
            selected_b_data = []
            selected_t_data = []

            # Iterate over each datetime in dtp
            for dt in dtp:
                if isinstance(dt, str):
                    dt = datetime.datetime.fromisoformat(dt).replace(tzinfo=None)
                    
                                
                if isinstance(t_data[0], str):
                    # Find the index of the closest datetime in t_data
                    closest_index = min(range(len(t_data)), key=lambda i: abs(datetime.datetime.fromisoformat(t_data[i]).replace(tzinfo=None) - dt.replace(tzinfo=None)))

                else:
                    # Find the index of the closest datetime in t_data
                    closest_index = min(range(len(t_data)), key=lambda i: abs(t_data[i].replace(tzinfo=None) - dt.replace(tzinfo=None)))

                # Find the index of the closest datetime in t_data
                #closest_index = min(range(len(t_data)), key=lambda i: abs(t_data[i] - dt))

                # Append corresponding data to the selected lists
                selected_b_data.append(b_data[closest_index])
                selected_t_data.append(t_data[closest_index])

            return np.array(selected_t_data), np.array(selected_b_data)
        
        
        
    def trajectory(self, dtp: Union[str, datetime.datetime, Sequence[str], Sequence[datetime.datetime]], pos_data: List[float], t_data: List[datetime.datetime],  **kwargs: Any) -> np.ndarray:
        
        # Initialize lists to store selected data
        selected_pos_data = []
        selected_t_data = []

        # Iterate over each datetime in dtp
        for dt in dtp:
            if isinstance(dt, str):
                dt = datetime.datetime.fromisoformat(dt)
                
            if isinstance(t_data[0], str):
                # Find the index of the closest datetime in t_data
                closest_index = min(range(len(t_data)), key=lambda i: abs(datetime.datetime.fromisoformat(t_data[i]).replace(tzinfo=None) - dt.replace(tzinfo=None)))
                
            else:
                # Find the index of the closest datetime in t_data
                closest_index = min(range(len(t_data)), key=lambda i: abs(t_data[i].replace(tzinfo=None) - dt.replace(tzinfo=None)))

            # Append corresponding data to the selected lists
            selected_pos_data.append(pos_data[closest_index])
            selected_t_data.append(t_data[closest_index])
            

        return np.array(selected_pos_data)
