import datetime
from typing import Literal, Optional
import numpy as np

from estimation.SVAR import PointIdentifiedSVAR


class RecursiveIdentification(PointIdentifiedSVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 date_frequency: Literal['D', 'W', 'M', 'Q', 'A'],
                 date_start: Optional[datetime.datetime] = None,
                 date_end: Optional[datetime.datetime] = None,
                 lag_order: Optional[int] = None,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(data=data,
                         var_names=var_names,
                         shock_names=shock_names,
                         date_frequency=date_frequency,
                         date_start=date_start,
                         date_end=date_end,
                         lag_order=lag_order,
                         constant=constant,
                         info_criterion=info_criterion)
        self.identification = 'recursive identification'

    def solve(self,
              comp_mat: Optional[np.ndarray] = None,
              cov_mat: Optional[np.ndarray] = None) -> np.ndarray:
        return np.eye(self.n_vars)
