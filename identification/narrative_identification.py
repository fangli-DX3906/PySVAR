import numpy as np
from typing import Literal, Optional

from core.svar import SetIdentifiedSVAR


class NarrativeIdentification(SetIdentifiedSVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 narrative_info: dict,
                 date_start: str,
                 constant: bool = True,
                 date_frequency: Literal['M', 'Q', 'A'] = 'Q',
                 lag_order: Optional[int] = None,
                 max_lag_order: Optional[int] = 8,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(data=data,
                         var_names=var_names,
                         shock_names=shock_names,
                         constant=constant,
                         lag_order=lag_order,
                         max_lag_order=max_lag_order,
                         info_criterion=info_criterion,
                         date_frequency=date_frequency,
                         date_start=date_start)

        self.identification = 'narrative identification'
        self.narrative_info = narrative_info
        dates, how = self.parse_narrative_info()
        self.convert_narrative_date_to_index(dates)
        self.convert_narrative_con_to_mat(how)

    def parse_narrative_info(self):
        return list(self.narrative_info.keys()), list(self.narrative_info.values())

    def convert_narrative_date_to_index(self, narrative_dates):
        index_list = []
        for date in narrative_dates:
            index_list.append(self.hd_dates.index(date))

        self.narrative_info_dates = index_list

    def convert_narrative_con_to_mat(self, narrative_cons):
        pass

    def identify(self,
                 n_rotation: int,
                 length_to_check: int = 1,
                 how: Literal['median', 'average'] = 'median',
                 seed: Optional[int] = None) -> None:
        pass
