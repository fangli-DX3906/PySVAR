from typing import Literal


class DateParser:
    def __init__(self,
                 start: str,
                 n_dates: int,
                 fequency: Literal['M', 'Q', 'A']):
        self.start = start
        self.n_dates = n_dates
        self.frequency = fequency
        self.date_list = [self.start]
        self.method_dict = {'M': self.make_monthly_list, 'Q': self.make_quarterly_list, 'A': self.make_annually_list}
        self.method_dict[self.frequency]()

    def make_monthly_list(self):
        while len(self.date_list) < self.n_dates:
            if int(self.date_list[-1][-2:]) == 12:
                self.date_list.append(f'{int(self.date_list[-1][:4]) + 1}01')
            else:
                if int(self.date_list[-1][-2:]) <= 8:
                    self.date_list.append(f'{int(self.date_list[-1][:4])}0{int(self.date_list[-1][-2:]) + 1}')
                elif int(self.date_list[-1][-2:]) == 9:
                    self.date_list.append(f'{int(self.date_list[-1][:4])}10')
                else:
                    self.date_list.append(f'{int(self.date_list[-1][:4])}{int(self.date_list[-1][-2:]) + 1}')

    def make_quarterly_list(self):
        if 'q' in self.date_list[0]:
            self.date_list[0] = f'{self.date_list[0][:4]}Q{self.date_list[0][-1]}'
        while len(self.date_list) < self.n_dates:
            if int(self.date_list[-1][-1]) == 4:
                self.date_list.append(f'{int(self.date_list[-1][:4]) + 1}Q1')
            else:
                self.date_list.append(f'{int(self.date_list[-1][:4])}Q{int(self.date_list[-1][-1]) + 1}')

    def make_annually_list(self):
        while len(self.date_list) < self.n_dates:
            self.date_list.append(str(int(self.date_list[-1]) + 1))


if __name__ == '__main__':
    aa = DateParser(start='202204', n_dates=24, fequency='M')
    aa.date_list

    bb = DateParser(start='2022q1', n_dates=10, fequency='Q')
    bb.date_list

    cc = DateParser(start='2022', n_dates=10, fequency='A')
    cc.date_list
