import pickle
import inflection
import datetime
import math
import pandas as pd
import numpy  as np


class Rossmann(object):

    def __init__(self):
        self.home_path                        = 'C:/Users/andre/repos/ds-em-producao/'
        self.competition_distance_scaler      = pickle.load(open(self.home_path + 'parameters/competition_distance_scaler.pkl', 'rb'))
        self.competition_number_monhts_scaler = pickle.load(open(self.home_path + 'parameters/competition_number_months_scaler.pkl', 'rb'))
        self.promo_number_weeks_scaler        = pickle.load(open(self.home_path + 'parameters/promo_number_weeks_scaler.pkl', 'rb'))
        self.store_type_scaler                = pickle.load(open(self.home_path + 'parameters/store_type_scaler.pkl', 'rb'))
        self.year_scaler                      = pickle.load(open(self.home_path + 'parameters/year_scaler.pkl', 'rb'))

    def data_cleaning(self, df):

        # Saving columns name to a new variable
        cols_name = df.columns.to_list()

        # Create a lambda function to pass columns name to snake case
        transform_snake_case = lambda x: inflection.underscore(x)

        # Apply function and save new names on another list
        cols_name_new = list(map(transform_snake_case, cols_name))
        
        # Renaming
        df.columns = cols_name_new 
        df['date'] = pd.to_datetime(df['date'])

        # fill na
        # competition_distance (fill out with extreme values, assuming that null values does not have close competitor)
        df['competition_distance'] = df['competition_distance'].apply(lambda x: 200000.0 if math.isnan(x) else x)
        
        # competition_open_since_month 
        df['competition_open_since_month'] = df.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month'])
                                                      else x['competition_open_since_month'], axis=1)
        
        # competition_open_since_year
        df['competition_open_since_year'] = df.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year'])
                                                     else x['competition_open_since_year'], axis=1)
        
        # promo2_since_week
        df['promo2_since_week'] = df.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week'])
                                                        else x['promo2_since_week'], axis = 1)
        
        # promo2_since_year
        df['promo2_since_year'] = df.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year'])
                                                        else x['promo2_since_year'], axis = 1)
        
        # promo_interval
        months = {1: 'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11: 'Nov', 12: 'Dec'}
        df['promo_interval'].fillna(0, inplace=True)
        df['month_map'] = df['date'].dt.month.map(months)
        
        df['is_promo'] = df.apply(lambda x: 0 if x['promo_interval'] == 0 
                                  else 1 if x['month_map'] in x['promo_interval'].split(',') 
                                  else 0, axis=1)

        # Change data types from float to int
        df['competition_open_since_month'] = df['competition_open_since_month'].astype('int64')
        df['competition_open_since_year'] = df['competition_open_since_year'].astype('int64')
        df['promo2_since_week'] = df['promo2_since_week'].astype('int64')
        df['promo2_since_year'] = df['promo2_since_year'].astype('int64')

        return df

    def feature_engineering(self, df):

        # year
        df['year'] = df['date'].dt.year
            
        # month
        df['month'] = df['date'].dt.month
            
        # week
        df['week'] = df['date'].dt.isocalendar().week
            
        # day
        df['day'] = df['date'].dt.day
            
        # year-week
        df['year_week'] = df['date'].dt.strftime('%Y-%W')
            
        # Competition since date
        df['competition_since_date'] = df.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], 
                                                                            month=x['competition_open_since_month'], 
                                                                            day=1), axis=1)
        
        # Number of months in competition
        df['competition_number_months'] = (((df['date'] - df['competition_since_date']).dt.days)/30).astype(int)
            
        # Promo since
        df['promo_since'] = df['promo2_since_year'].astype(str) + '-' + df['promo2_since_week'].astype(str)
        df['promo_since'] = df['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
            
        # Number of weeks in promo
        df['promo_number_weeks'] = (((df['date'] - df['promo_since']).dt.days)/7).astype(int)
            
        # Assortment
        df['assortment'] = df['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')
            
        # State Holiday
        df['state_holiday'] = df['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 
                                                                  'easter_holiday' if x == 'b' else 
                                                                  'christmas' if x == 'c' else 
                                                                  'no_holiday')
        
        # Feature Filtering
        df = df[df['open'] != 0].reset_index(drop=True)
        cols = ['open', 'month_map', 'promo_interval']
        df.drop(columns=cols, axis=1, inplace=True)

        return df

    def data_preparation(self, df):

        # competition_distance
        df['competition_distance'] = self.competition_distance_scaler.fit_transform(df[['competition_distance']].values)
        
        # year
        df['year'] = self.year_scaler.fit_transform(df[['year']].values)
        
        # competition_number_months
        df['competition_number_months'] =  self.competition_number_monhts_scaler.fit_transform(df[['competition_number_months']].values)
        
        # promo_number_weeks
        df['promo_number_weeks'] = self.promo_number_weeks_scaler.fit_transform(df[['promo_number_weeks']].values)

        # state_holiday
        df = pd.get_dummies(df, prefix=['state_holiday'], columns=['state_holiday'])
            
        # store_type
        df['store_type'] = self.store_type_scaler.fit_transform(df['store_type'])
            
        # assortment
        df['assortment'] = df['assortment'].apply(lambda x: 1 if x == 'basic' else 2 if x == 'extra' else 3)
        
        # day
        df['day_sin'] = df['day'].apply(lambda x: np.sin(x * (2 * np.pi/30)))
        df['day_cos'] = df['day'].apply(lambda x: np.cos(x * (2 * np.pi/30)))
            
        # month
        df['month_sin'] = df['month'].apply(lambda x: np.sin(x * (2 * np.pi/12)))
        df['month_cos'] = df['month'].apply(lambda x: np.cos(x * (2 * np.pi/12)))
            
        # day_of_week
        df['day_of_week_sin'] = df['day_of_week'].apply(lambda x: np.sin(x * (2 * np.pi/7)))
        df['day_of_week_cos'] = df['day_of_week'].apply(lambda x: np.cos(x * (2 * np.pi/7)))
            
        # week
        df['week_sin'] = df['week'].apply(lambda x: np.sin(x * (2 * np.pi/52)))
        df['week_cos'] = df['week'].apply(lambda x: np.cos(x * (2 * np.pi/52)))

        selected_cols = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_year', 'promo2',
                         'promo2_since_year', 'competition_number_months', 'promo_number_weeks', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                         'day_of_week_sin', 'day_of_week_cos', 'week_sin', 'week_cos', 'year']
        
        return df.loc[:, selected_cols]

    def get_prediction(self, model, original_data, test_data):

        # prediction
        pred = model.predict(test_data)

        # append predictions to the main dataset
        original_data['prediction'] = np.exmp1(pred)

        return original_data.to_json(orient='records', date_format='iso')