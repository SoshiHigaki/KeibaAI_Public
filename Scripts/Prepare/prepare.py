import os

from Scripts.Prepare import get_data
from Scripts.Prepare import get_horse_data
from Scripts.Prepare import get_jockey_data
from Scripts.Prepare import get_trainer_data

class Prepare(object):
    def __init__(self):
        self.gd = get_data.Get_Data()
        self.gh = get_horse_data.Get_Horse_Data()
        self.gj = get_jockey_data.Get_Jockey_Data()
        self.gt = get_trainer_data.Get_Trainer_Data()

        self.save_folder = 'Data/Race/'
        self.return_folder = 'Data/Return/'

        self.test_ratio = 0.2

    
    def main(self, start_year, start_mon, end_year, end_mon):
        period = self.generate_year_month_list(start_year, start_mon, end_year, end_mon)

        horse_ids = []
        for p in period:
            train_folder_path = f'{self.save_folder}/Train/{p[0]}/'
            test_folder_path = f'{self.save_folder}/Test/{p[0]}/'
            return_folder_path = f'{self.return_folder}/{p[0]}/'
            os.makedirs(train_folder_path) if not os.path.exists(train_folder_path) else None
            os.makedirs(test_folder_path) if not os.path.exists(test_folder_path) else None
            os.makedirs(return_folder_path) if not os.path.exists(return_folder_path) else None

            train_df, test_df, r_df = self.gd.main(p[0], p[1], p[0], p[1])
            horse_ids.extend(list(train_df['horse_id']))
            horse_ids.extend(list(test_df['horse_id']))

            train_df.to_pickle(f'{train_folder_path}/{str(p[1]).zfill(2)}.pkl')
            test_df.to_pickle(f'{test_folder_path}/{str(p[1]).zfill(2)}.pkl')

            r_df.to_pickle(f'{return_folder_path}/{str(p[1]).zfill(2)}.pkl')

        horse_ids = list(set(horse_ids))
        self.gh.main(horse_ids)
        
        self.gj.main(start_year-1, end_year)
        self.gt.main(start_year-1, end_year)

    def generate_year_month_list(self, start_year, start_mon, end_year, end_mon):
        year_month_list = []
        while (start_year, start_mon) <= (end_year, end_mon):
            year_month_list.append([start_year, start_mon])
            start_mon += 1
            if start_mon > 12:
                start_year += 1
                start_mon = 1
        return year_month_list