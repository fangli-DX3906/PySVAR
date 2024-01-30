import datetime
import random
from typing import Literal, Tuple, Optional
import numpy as np
import multiprocessing
from multiprocessing import Lock, Value
from tqdm import tqdm

from estimation.SVAR import SetIdentifiedSVAR


class SignRestriction(SetIdentifiedSVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 target_signs: np.ndarray,
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
        self.identification = 'sign restriction'
        self.target_signs = target_signs
        self.n_ones = np.sum(self.target_signs == 1)
        self.n_nones = np.sum(self.target_signs == -1)
        self.num_unrestricted = self.n_vars ** 2 - self.n_ones - self.n_nones
        if self.n_ones > self.n_nones:
            self.direction = 'descend'
        else:
            self.direction = 'ascend'

    def _sort_row(self, mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = []
        for i in list(range(self.n_vars))[::-1]:
            c.append(2 ** i)
        c = np.array(c)
        mask = c * np.ones((self.n_vars, 1))
        C = np.sum(mask * mat, axis=1)
        idx = np.argsort(C)
        if self.direction == 'descend':
            idx = idx[::-1]
        return idx, mat[idx, :]

    def draw_rotation(self) -> np.ndarray:
        raw_mat = np.random.randn(self.n_vars, self.n_vars)
        Q, R = np.linalg.qr(raw_mat)
        Q = np.sign(np.diag(R)).reshape((-1, 1)) * Q
        return Q

    def _check_sign_parallel(self,
                             queue: multiprocessing.Queue,
                             counter: multiprocessing.Value,
                             lock: multiprocessing.Lock,
                             n_rotation_per_process: int,
                             length_to_check: int = 1):
        results = []
        while len(results) < n_rotation_per_process:
            D = self.draw_rotation()
            self.tool.update(rotation=D)
            self.tool.estimate_irf(length=length_to_check)
            _irfs_ = self.tool.irf
            irf_sign = np.sign(np.sum(_irfs_, axis=1).reshape((self.n_vars, self.n_vars)))
            idx, sorted_signs = self._sort_row(irf_sign)
            diff_sign = self.target_signs - sorted_signs
            if np.sum(diff_sign ** 2) == self.num_unrestricted:
                D = D[:, idx]
                results.append(D)
                with lock:
                    counter.value += 1
        queue.put(results)

    def _check_sign(self,
                    n_rotation: int,
                    length_to_check: int = 1):
        rotation_list = []
        pbar = tqdm(total=n_rotation, desc=f'Drawing {n_rotation} rotations...')
        while len(rotation_list) < n_rotation:
            D = self.draw_rotation()
            self.tool.update(rotation=D)
            self.tool.estimate_irf(length=length_to_check)
            _irfs_ = self.tool.irf
            irf_sign = np.sign(np.sum(_irfs_, axis=1).reshape((self.n_vars, self.n_vars)))
            idx, sorted_signs = self._sort_row(irf_sign)
            diff_sign = self.target_signs - sorted_signs
            if np.sum(diff_sign ** 2) == self.num_unrestricted:
                D = D[:, idx]
                rotation_list.append(D)
                pbar.update(1)
        return rotation_list

    def identify(self,
                 n_rotation: int,
                 parallel: bool = False,
                 n_process: int = 4,
                 length_to_check: int = 1,
                 seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if parallel:
            n_rotation_each = n_rotation // n_process
            n_rotation_li = [n_rotation_each for _ in range(n_process)[:-1]]
            n_rotation_li.append(n_rotation - (n_process - 1) * n_rotation_each)
            rotation_queue = multiprocessing.Queue()
            progress_counter = Value('i', 0)
            lock = Lock()

            processes_list = []
            for _, n_rotation_this_process in zip(range(n_process), n_rotation_li):
                p = multiprocessing.Process(target=self._check_sign_parallel,
                                            args=(rotation_queue, progress_counter, lock,
                                                  n_rotation_this_process, length_to_check))
                processes_list.append(p)
                p.start()

            pbar = tqdm(total=n_rotation, desc=f'Drawing {n_rotation} rotations...')
            while True:
                with lock:
                    current_progress = progress_counter.value
                pbar.update(current_progress - pbar.n)
                if current_progress >= n_rotation:
                    break

            rotation_list = []
            for _ in range(n_process):
                rotation_list.extend(rotation_queue.get())

            for p in processes_list:
                p.join()

            self.rotation_list = rotation_list[:n_rotation]
        else:
            self.rotation_list = self._check_sign(n_rotation, length_to_check)

        self.full_irf()
