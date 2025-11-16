# preprocessing_data.py
import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self, unprop_path, maneuvers_path):
        self.unprop_df = pd.read_csv(unprop_path, parse_dates=[0], index_col=0)
        self.maneuvers_df = self.get_maneuvers_dataframe(maneuvers_path)


    # setter
    def set_unprop_data(self, unprop_df):
        self.unprop_df = unprop_df
    def set_maneuvers_data(self, maneuvers_df):
        self.maneuvers_df = maneuvers_df
    # getter
    def get_unprop_data(self):
        return self.unprop_df

    def get_maneuvers_data(self):
        return self.maneuvers_df

    # Function to parse maneuvers file and create a DataFrame
    def get_maneuvers_dataframe(self, maneuvers_path):
    # Create maneuvers dataframe, store information
        maneuvers_list = []
        with open(maneuvers_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                # --- Parse General Maneuver Info (mostly fixed-width) ---
                sat_id = line[0:5].strip()
                
                # Maneuver Start Time
                y1 = int(line[6:10])
                doy1 = int(line[11:14])
                h1 = int(line[15:17])
                m1 = int(line[18:20])
                start_time = pd.to_datetime(f'{y1}-{doy1}', format='%Y-%j') + pd.Timedelta(hours=h1, minutes=m1)

                # Maneuver End Time
                y2 = int(line[21:25])
                doy2 = int(line[26:29])
                h2 = int(line[30:32])
                m2 = int(line[33:35])
                end_time = pd.to_datetime(f'{y2}-{doy2}', format='%Y-%j') + pd.Timedelta(hours=h2, minutes=m2)

                maneuver_type = line[36:39].strip()
                param_type = line[40:43].strip()
                num_burns = int(line[44])

                # --- Parse Info for Each Burn ---
                burn_data_block = line[46:]
                for i in range(num_burns):
                    k = i * 232  # Offset for each burn block

                    # Median Burn Time
                    burn_y = int(burn_data_block[k+0:k+4])
                    burn_doy = int(burn_data_block[k+5:k+8])
                    burn_h = int(burn_data_block[k+9:k+11])
                    burn_m = int(burn_data_block[k+12:k+14])
                    burn_s_ms = float(burn_data_block[k+15:k+21])
                    burn_seconds = int(burn_s_ms)
                    burn_microseconds = int((burn_s_ms - burn_seconds) * 1e6)
                    median_time = pd.to_datetime(f'{burn_y}-{burn_doy}', format='%Y-%j') + \
                                pd.Timedelta(hours=burn_h, minutes=burn_m, seconds=burn_seconds, microseconds=burn_microseconds)

                    # Burn parameters
                    duration = float(burn_data_block[k+22:k+42])
                    dv1 = float(burn_data_block[k+43:k+63])
                    dv2 = float(burn_data_block[k+64:k+84])
                    dv3 = float(burn_data_block[k+85:k+105])
                    acc1 = float(burn_data_block[k+106:k+126])
                    acc2 = float(burn_data_block[k+127:k+147])
                    acc3 = float(burn_data_block[k+148:k+168])
                    delta_acc1 = float(burn_data_block[k+169:k+189])
                    delta_acc2 = float(burn_data_block[k+190:k+210])
                    delta_acc3 = float(burn_data_block[k+211:k+231])

                    maneuvers_list.append({
                        'sat_id': sat_id,
                        'maneuver_start': start_time,
                        'maneuver_end': end_time,
                        'maneuver_type': maneuver_type,
                        'param_type': param_type,
                        'total_burns_in_maneuver': num_burns,
                        'burn_number': i + 1,
                        'burn_median_time': median_time,
                        'burn_duration_sec': duration,
                        'dv_radial': dv1,
                        'dv_along_track': dv2,
                        'dv_cross_track': dv3,
                        'acc_radial_10e-6ms2': acc1,
                        'acc_along_track_10e-6ms2': acc2,
                        'acc_cross_track_10e-6ms2': acc3,
                        'delta_acc_radial_10e-6ms2': delta_acc1,
                        'delta_acc_along_track_10e-6ms2': delta_acc2,
                        'delta_acc_cross_track_10e-6ms2': delta_acc3
                    })
        maneuvers_df = pd.DataFrame(maneuvers_list)
        return maneuvers_df
    
    # Function to add maneuver status to unpropagated data
    def add_maneuver_status(self):
        maneuvers_df = self.maneuvers_df
        # Initialize maneuver status column
        self.unprop_df['maneuver_status'] = 0

        for _, maneuver in maneuvers_df.iterrows():
            median_time = maneuver['burn_median_time']
            after_median_indices = self.unprop_df.index[self.unprop_df.index >= median_time]
            
            # Mark first dates after the maneuver as 1 
            if not after_median_indices.empty:
                first_date_after = after_median_indices[0]
                self.unprop_df.loc[first_date_after, 'maneuver_status'] = 1
    
    # preprocessing step
    def preprocess_data(self):       
        # Remove the time part from the index for daily frequency
        self.unprop_df.index = self.unprop_df.index.normalize()
        # Ensure index is unique and the value as the mean for that day
        self.unprop_df = self.unprop_df.groupby(self.unprop_df.index).mean()
        # Create a full date range to identify missing days
        full_date_range = pd.date_range(start=self.unprop_df.index.min(), end=self.unprop_df.index.max(), freq='D')
        # Reindex to the full date range
        self.unprop_df = self.unprop_df.reindex(full_date_range)
        # Fill missing maneuver status with 0 (no maneuver)
        self.unprop_df['maneuver_status'] = self.unprop_df['maneuver_status'].fillna(0)
        # Fill other missing values with interpolation
        self.unprop_df = self.unprop_df.interpolate(method='time')
    
    # create new features
    def create_features(self):
        mu = 398600.4418  # Earth's gravitational parameter in km^3/s^2
        # The 'Brouwer mean motion' is given in revolutions per day.
        # Convert revolutions per day to radians per second for the formula.
        n_rev_per_day = self.unprop_df["Brouwer mean motion"]
        n_rad_per_sec = n_rev_per_day * (2 * np.pi) / 86400
        
        # Calculate semi-major axis 'a' in km
        a = (mu / n_rad_per_sec**2)**(1/3)
        
        e = self.unprop_df["eccentricity"]
        # Ensure the argument of sqrt is non-negative to avoid warnings
        self.unprop_df["specific_angular_momentum"] = np.sqrt(mu * a * np.maximum(0, 1 - e**2))

        # create specific orbital energy feature
        self.unprop_df["specific_orbital_energy"] = -mu / (2 * a)
    
        
    def processed_data(self):
        self.add_maneuver_status()
        self.preprocess_data()
        self.create_features()

    # Function to select relevant features
    def select_features(self, features_list):
        unprop_df = self.unprop_df.copy()
        return unprop_df[features_list + ['maneuver_status']]


def main():
    # Paths
    dir = "./satellite_data/"
    unprop_dir = dir + "orbital_elements/"
    man_dir = dir + "manoeuvres/"

    cs2_unprop_path = unprop_dir + "unpropagated_elements_CryoSat-2.csv"
    cs2_maneuvers_path = man_dir + "cs2man.txt"

    # Preprocess CryoSat-2 data
    cs2_preprocessor = Preprocessor(cs2_unprop_path, cs2_maneuvers_path)
    cs2_preprocessor.processed_data()
    cs2_unprop_final = cs2_preprocessor.select_features(
        ["inclination", "Brouwer mean motion", "specific_angular_momentum"]
    )
    print(cs2_unprop_final.head())
    print(cs2_unprop_final.info())
    print(cs2_unprop_final.describe())
    print(cs2_preprocessor.get_maneuvers_data().head())

if __name__ == "__main__":
    main()
