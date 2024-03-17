import os

from Scripts.Prepare import get_data
from Scripts.Prepare import get_horse_data


class prepare(object):
    def __init__(self):
        self.gd = get_data.get_data()
        self.gh = get_horse_data.get_horse_data()

        self.save_folder = 'Data/Race/'

    
    def main(self, start_year, start_mon, end_year, end_mon):
        period = self.generate_year_month_list(start_year, start_mon, end_year, end_mon)

        horse_ids = []
        for p in period:
            folder_path = self.save_folder + f'{p[0]}'
            os.makedirs(folder_path) if not os.path.exists(folder_path) else None
            df = self.gd.main(p[0], p[1], p[0], p[1])
            horse_ids.extend(list(df['horse_id']))

            df.to_pickle(f'{folder_path}/{str(p[1]).zfill(2)}.pkl')

        horse_ids = list(set(horse_ids))
        self.gh.main(horse_ids)


    def generate_year_month_list(self, start_year, start_mon, end_year, end_mon):
        year_month_list = []
        while (start_year, start_mon) <= (end_year, end_mon):
            year_month_list.append([start_year, start_mon])
            start_mon += 1
            if start_mon > 12:
                start_year += 1
                start_mon = 1
        return year_month_list