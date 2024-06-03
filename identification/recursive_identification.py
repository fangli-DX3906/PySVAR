from typing import Literal, Optional
import numpy as np

from estimation.svar import PointIdentifiedSVAR


class RecursiveIdentification(PointIdentifiedSVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 constant: bool = True,
                 lag_order: Optional[int] = None,
                 max_lag_order: Optional[int] = 8,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic',
                 date_frequency: Literal['M', 'Q', 'A'] = 'Q',
                 date_start: Optional[str] = None):

        super().__init__(data=data,
                         var_names=var_names,
                         shock_names=shock_names,
                         constant=constant,
                         lag_order=lag_order,
                         max_lag_order=max_lag_order,
                         info_criterion=info_criterion,
                         date_frequency=date_frequency,
                         date_start=date_start)

        self.identification = 'recursive identification'

    def solve(self,
              comp_mat: Optional[np.ndarray] = None,
              cov_mat: Optional[np.ndarray] = None) -> np.ndarray:
        return np.eye(self.n_vars)
