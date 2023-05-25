from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import datetime
import pytz
from tqdm import tqdm
import scipy.stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
import seaborn as sns
import re
from matplotlib.ticker import MultipleLocator
warnings.filterwarnings('ignore', category=FutureWarning)
import RegressionModeling as rm
import PercentileIndex as PI
import os
import matplotlib as mpl
from matplotlib.colors import ListedColormap

def parse_csv_file(filename):

    def get_area_name(area):
        for key, value in area_mappings.items():
            if area.startswith(key):
                return value
        return area

    area_mappings = {
            '13': 'Cabrillo',
            '14': 'Redondo',
            '15': 'Hermosa-Manhattan',
            '16': 'Manhattan',
            '17': 'ElSegundo-PlayaDelRey',
            '23': 'Venice',
            '25': 'SantaMonica',
            '26': 'SantaMonica',
            '27': 'WillRogers',
            '33': 'Malibu',
            '34': 'Zuma'
    }

    df = pd.read_csv(filename)
    pacific_tz = pytz.timezone('US/Pacific')
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Area'] = df['Area'].apply(lambda x: get_area_name(x))
    df = df.dropna(subset=['Attendance']).loc[df['Date'].dt.year != 2020]  # Remove rows with empty attendance
    df = df[df['Holiday'] == '0']
    
    # Attendance summary calculation
    total_attd = df['Attendance'].sum()
    julian_day_values = df['JulianDay']
    selected_days_df = df[(julian_day_values >= 125) & (julian_day_values <= 275)]

    selected_days_attd = selected_days_df['Attendance'].sum()
    other_days_attd = total_attd - selected_days_attd

    selected_days_ratio = selected_days_attd / total_attd
    other_days_ratio = other_days_attd / total_attd

    print("Sum of attendance from JulianDay 125 to 275:", selected_days_attd)
    print("Sum of attendance for the rest of the days:", other_days_attd)
    print("Total attendance:", total_attd)
    print("Selected days attendance ratio:", selected_days_ratio)
    print("Other days attendance ratio:", other_days_ratio)
    
    df = df[(julian_day_values >= 125) & (julian_day_values <= 275)]
    
    df['JulianFactor'] = 75 - (np.abs(df['JulianDay'] - 200))
    # 75 - |JulianDay - 200|
    
    df = df.groupby(['Area', 'Date']).agg({'Attendance': 'sum', 'KLAXMaxT': np.nanmean, 'KLAXPrecip': np.nanmean, 'KVNYMaxT': np.nanmean, 'KVNYPrecip': np.nanmean, 'JulianDay': np.nanmean, 'DayOfWeek': np.nanmean, 'KLAXAvgWS': np.nanmean, 'JulianFactor': np.nanmean}).reset_index()
    
    areas = df['Area'].unique()
    start_date = datetime.date(2017, 1, 1)  # Specify the date class explicitly
    end_date = datetime.date(2021, 12, 31)  # Specify the date class explicitly
    years = end_date.year - start_date.year  # Calculate the number of years
    lst = list(range(125, 276)) * years  # Repeat the Julian Day range for each year
    new_df = pd.DataFrame(columns=df.columns)
    
    for area in areas:
        data_area = df[df['Area'] == area]
        data_jd = data_area['JulianDay'].values  # Extract JulianDay values for the area
        i = 0
        for jd in lst:
            if i < len(data_jd) and jd == data_jd[i]:
                new_row = data_area.iloc[i]
                i += 1
            else:
                new_row = pd.Series([np.nan] * len(df.columns), index=df.columns)
                new_row['Area'] = area
                new_row['JulianDay'] = jd
            new_df = new_df.append(new_row, ignore_index=True)
            
    new_df['JulianDay'] = pd.to_numeric(new_df['JulianDay'], errors='coerce')
    new_df['KLAXAvgWS'] = pd.to_numeric(new_df['KLAXAvgWS'], errors='coerce')
    new_df['JulianFactor'] = pd.to_numeric(new_df['JulianFactor'], errors='coerce')
    #return df
    return new_df


    
def split_data_frame(df):
    df['DayType'] = np.where(df['DayOfWeek'].isin([5,6]), 'Weekend', 'Weekday')
    df[['KLAXPrecip', 'KVNYPrecip']] = df[['KLAXPrecip', 'KVNYPrecip']].astype(float)
    #df['KLAXPrecip'] = (df['KLAXPrecip'] > 0.0).astype(int)
    #df['KVNYPrecip'] = (df['KVNYPrecip'] > 0.0).astype(int)
    
    df = df.groupby('Area').filter(lambda x: len(x) >= 100 and x['Attendance'].max() <= 250000)

    df_scaled = df.copy()
    scaler = StandardScaler()

    variables = ['KLAXMaxT', 'KVNYMaxT', 'KLAXAvgWS', 'JulianFactor']
    print('\nCopying df and standardizing values')
    for var in variables:
        data = df[var]
        
        scaler.fit(data.values.reshape(-1,1))
        data_mean = scaler.mean_[0]
        data_std = scaler.scale_[0]

        statistics[var] = {'mean':(data_mean), 'std':(data_std)}

        print(f"{var} {statistics[var]}")
        print()

        df_scaled[var] = scaler.transform(data.values.reshape(-1,1)).flatten()
    return df, df_scaled


def filter_areas(grouped_df, variables, secondary_y_variables, grouped_df_scaled):
    areas = grouped_df['Area'].unique()
    areas_to_keep = []

    for i in range(0, len(areas), 4):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.subplots_adjust(hspace=0.4, top=0.85)
        for j, ax in enumerate(axs.flatten()):
            idx = i + j
            if idx < len(areas):
                area = areas[idx]
                mask = grouped_df['Area'] == area
                data = grouped_df.loc[mask]

                # Plot the variables on the primary y-axis
                for idx, variable in enumerate(variables):
                    ax.scatter(data['JulianDay'], data[variable], label=variable, marker=['^', '^'][idx], s=7,
                               color=['orange', 'red'][idx])

                # Create a secondary y-axis and plot the variables on it
                ax2 = ax.twinx()
                for idx, variable in enumerate(secondary_y_variables):
                    ax2.scatter(data['JulianDay'], data[variable], label=f'{variable}', linestyle='--', marker='o',
                                s=7, color=['blue'][idx])

                # Combine legends for primary and secondary y-axes
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, markerscale=1.5, loc='best', fontsize=14)

                ax.set_title(f'{i + j + 1}. Area {area}: {", ".join(variables + secondary_y_variables)}', fontweight='bold')
                ax.set_xlabel('Julian Day')
                ax.set_ylabel(f'{", ".join(variables)}')
                ax2.set_ylabel(f'{", ".join(secondary_y_variables)}')

        plt.tight_layout()
        plt.show()

        while True:
            keep_input = input("Enter the numbers of the areas you want to keep (separated by commas): ").strip()
            try:
                keep_numbers = list(map(int, keep_input.split(','))) if keep_input else []
                if all(num <= len(areas) for num in keep_numbers):
                    break
                else:
                    print("Invalid input, please try again.")
            except ValueError:
                print("Invalid input, please try again.")

        areas_to_keep.extend([areas[num - 1] for num in keep_numbers])

    print(areas_to_keep)

    for area in areas:
        if area not in areas_to_keep:
            mask = grouped_df['Area'] == area
            grouped_df = grouped_df.loc[~mask]
            grouped_df_scaled = grouped_df_scaled.loc[~mask]

    return grouped_df, grouped_df_scaled

def make_plots(grouped_df, x_variable, y_variables, y2_variables, nameOutput, y_min=None, y_max=None, alpha=1.0):
    colors_map = {'KLAXMaxT': 'red', 'KVNYMaxT': 'orange', 'Attendance': 'black','KLAXPrecip': 'green','KVNYPrecip': 'blue', 'KLAXAvgWS': 'purple', 'JulianFactor' : 'teal'}
    
    buffer = 0.05

    global_y_min = y_min if y_min is not None else grouped_df[y_variables].min().min() * (1 - buffer)
    global_y_max = y_max if y_max is not None else grouped_df[y_variables].max().max() * (1 + buffer)

    if y2_variables:
        global_y2_max = grouped_df[y2_variables].max().max() * (1 + buffer)
        global_y2_min = grouped_df[y2_variables].min().min() * (1 - buffer)

    start_year = grouped_df['Date'].min().year
    end_year = grouped_df['Date'].max().year
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.55, top=0.83)
    
    x_min, x_max = grouped_df[x_variable].agg(['min', 'max']) + (grouped_df[x_variable].max() - grouped_df[x_variable].min()) * np.array([-buffer, buffer])
    
    

    label_map = {
    'KLAXMaxT': 'KLAX Max Temp',
    'KLAXAvgWS': 'KLAX Average Wind Speed',
    'KLAXPrecip': 'KLAX Precipitation',
    'KVNYPrecip': 'KVNY Precipitation',
    'KVNYMaxT': 'KVNY Max Temp',
    'JulianDay': 'Julian Day',
    'JulianFactor': 'Day of Year Factor',
    'Attendance': 'Attendance',
    None: None
    }

    y_variablesLabel = []
    y2_variablesLabel = []

    if x_variable:
        x_variableLabel = label_map.get(x_variable)


    if y_variables:
        if y_variables[0]:
            for var in y_variables:
                y_variablesLabel.append(label_map.get(var))
                
    
    if y2_variables:
        if y2_variables[0]:
            for var in y2_variables:
                y2_variablesLabel.append(label_map.get(var))
    
    titleLabels = []
    if y_variablesLabel:
        titleLabels.extend(y_variablesLabel)
    if y2_variablesLabel:
        titleLabels.extend(y2_variablesLabel)
    titleLabels = [label for label in titleLabels if label is not None]
    titleString = ', '.join(titleLabels)


    print(f'Title Guidance: {titleString} vs {x_variableLabel}\n{start_year}-{end_year} (excluding 2020)\n')
    setTitle = input('Set title: ')
    if setTitle =='':
        setTitle = f'{titleString} vs {x_variableLabel}\n{start_year}-{end_year} (excluding 2020)\n'
        
    loc = (grouped_df['Area'].unique())
    if len(loc) > 3:
        loc = 'All Locations'
    else:
        loc = loc[0]
    fig.suptitle(f'{loc} - {setTitle}', fontweight='bold', fontsize=20)

    if len(y_variablesLabel) > 1:
        y_variablesLabel = ', '.join(y_variablesLabel)
    else:
        y_variablesLabel = ''.join(y_variablesLabel)

    print(f'ylabel guidance: {y_variablesLabel}')
    ylabel = input('Set ylabel: ')
    if ylabel == '':
        ylabel = f'{y_variablesLabel}'    
        
    if len(y2_variablesLabel) > 1:
        y2_variablesLabel = ', '.join(y2_variablesLabel)
    else:
        y2_variablesLabel = ''.join(y2_variablesLabel)
        
    if y2_variables:
        print(f'y2label guidance: {y2_variablesLabel}')
        y2label = input('Set y2label: ')
        if y2label == '':
            y2label = f'{y2_variablesLabel}'
            
    
    print(f'xlabel guidance: {x_variableLabel}')
    xlabel = input('Set xlabel: ')
    if xlabel == '':
        xlabel = f'{x_variableLabel}' 
    
    for ax, (day_type, title) in zip(axs, [('Weekend', 'Weekend'), ('Weekday', 'Weekday')]):
        data = grouped_df.copy()
    
        # Set variables to null if they are not of the current day type
        data.loc[data['DayType'] != day_type, x_variable] = np.nan
        data.loc[data['DayType'] != day_type, y_variables] = np.nan
        data.loc[data['DayType'] != day_type, 'Attendance'] = np.nan
        
        dataNoNone = data.dropna(subset=[x_variable] + y_variables)
        

        data = data.sort_values(by=[x_variable])
        dataNoNone = dataNoNone.sort_values(by=[x_variable])

        ax.set_title(f'{title}\n', fontweight='bold', fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.grid(True, alpha=0.5)
        
        ax.set_xlim(x_min, x_max)

        for y_variable in y_variables:
            color = colors_map[y_variable]
            if color == 'black':
                    color = colors_map[x_variable]
            ax.scatter(data[x_variable], data[y_variable], marker='^', color=color, label=y_variable, alpha=0.7, zorder=1)

            #z = np.polyfit(data[x_variable], data[y_variable], 3)
            #f = np.poly1d(z)
            #ax.plot(data[x_variable], f(data[x_variable]), color='black', linewidth=4, zorder=3)
            #ax.plot(data[x_variable], f(data[x_variable]), color=color, linewidth=2, zorder=4)

        handles, labels = ax.get_legend_handles_labels()
        
        if y2_variables:
            ax2 = ax.twinx()
            ax2.set_ylabel(y2label, fontsize=16)
            for y2_variable in y2_variables:
                color = colors_map[y2_variable]
                ax2.scatter(data[x_variable], data[y2_variable], marker='o', color=color, alpha=alpha, label=y2_variable, zorder=2)

            handles2, labels2 = ax2.get_legend_handles_labels()
            handles.extend(handles2)
            labels.extend(labels2)

        # Set x and y limits for both subplots using global min and max values
        ax.set_ylim(global_y_min, global_y_max)
        if y2_variables:
            ax2.set_ylim(global_y2_min, global_y2_max)
    
        # Add correlation for both temperatures vs Attd to the top right of the plot
        print(x_variable, y_variables)
        corr_data = {y: scipy.stats.pearsonr(dataNoNone[x_variable], dataNoNone[y]) for y in y_variables}
        textstr = '     '.join(f'{y} Correlation r: {corr:.2f}' for y, (corr, _) in corr_data.items())
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        # Calculate position of annotation based on maximum x and y values of the data
        ax.text(0.5, 1.08, f"{textstr}", transform=ax.transAxes, fontweight='normal', fontsize=14, va='top', ha='center')
        
    # Set the n_col to 3 so that each label is side by side in the legend
    plt.legend(handles, labels, loc='center', ncol=3, markerscale=1.5, fontsize=14, bbox_to_anchor=(0.5, 1.28))

    plt.savefig(nameOutput)
    plt.close()

       
def prepare_data(grouped_df, day_type):
    cols_to_keep = ['JulianDay', 'JulianFactor', 'DayOfWeek','Area']
    data = grouped_df.copy()
    data.loc[data['DayType'] != day_type, ~data.columns.isin(cols_to_keep)] = np.nan
    data = data.reset_index().rename(columns={"index": "Date"})


    # One-hot encode day of the week
    #one_hot_encoder = OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')
    #day_of_week_one_hot = one_hot_encoder.fit_transform(data["DayOfWeek"].values.reshape(-1, 1))
    #dayNames = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    #for i, category in enumerate(one_hot_encoder.categories_[0]):
    #    if dayNames[category] not in ['Saturday', 'Sunday']:
    #        data[f"{dayNames[category]}"] = day_of_week_one_hot[:, i]

    # Drop unnecessary columns
    columns_to_drop = ["Date", "Holiday", "Month", "Year", "DayType", "DayOfWeek",'Monday','Tuesday','Wednesday','Thursday','Friday']
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data = data.drop(existing_columns_to_drop, axis=1)

    return data


filename = 'LFDataPython.csv'
julianFactorReplacement = f'75-|DayOfYear-200|'
statistics={}
df = parse_csv_file(filename)
grouped_df_orig, grouped_df_scaled_orig = split_data_frame(df)

## Correlations Attd vs variable
#make_plots(grouped_df, 'KLAXMaxT', ['Attendance'], [''], 'CorrPlots/KLAXTempCorrPlot.png', None, None)
#make_plots(grouped_df, 'KVNYMaxT', ['Attendance'], [''], 'CorrPlots/KVNYTempCorrPlot.png', None, None)
#make_plots(grouped_df, 'KLAXAvgWS', ['Attendance'], [''], 'CorrPlots/WSCorrPlot.png', None, None)
#make_plots(grouped_df, 'JulianFactor', ['Attendance'], [''], 'CorrPlots/JulianFactorCorrPlot.png', None, None)
#
##Attd vs variables over days
#make_plots(grouped_df, 'JulianDay', ['KLAXMaxT', 'KVNYMaxT'], ['Attendance'], 'JulianDayPlots/TempWeekPlot.png', 50, 120, 0.3)

filter_choice=""
while filter_choice not in ["yes", "no"]: 
    filter_choice = input("Do you want to filter out areas? (yes/no): ").strip().lower()
    if filter_choice not in ["yes", "no"]:
        print('Try again. Not an option.')

if filter_choice == "yes":
    grouped_df, grouped_df_scaled = filter_areas(grouped_df_orig, ['KLAXMaxT', 'KVNYMaxT'], ['Attendance'], grouped_df_scaled_orig)
    
    # Correlations Attd vs variable
    if not os.path.isfile('CorrPlots/fKLAXTempCorr.png'):
        make_plots(grouped_df, 'KLAXMaxT', ['Attendance'], None, 'CorrPlots/fKLAXTempCorr.png', None, None, 1)
    if not os.path.isfile('CorrPlots/fKVNYTempCorr.png'):
        make_plots(grouped_df, 'KVNYMaxT', ['Attendance'], None, 'CorrPlots/fKVNYTempCorr.png', None, None, 1)
    if not os.path.isfile('CorrPlots/fKLAXWSCorr.png'):
        make_plots(grouped_df, 'KLAXAvgWS', ['Attendance'], None, 'CorrPlots/fKLAXWSCorr.png', None, None, 1)
    if not os.path.isfile('CorrPlots/fJulianFactorCorr.png'):
        make_plots(grouped_df, 'JulianFactor', ['Attendance'], None, 'CorrPlots/fJulianFactorCorr.png', None, None, 1)
    if not os.path.isfile('CorrPlots/fKLAXPrecipCorr.png'):
        make_plots(grouped_df, 'KLAXPrecip', ['Attendance'], None, 'CorrPlots/fKLAXPrecipCorr.png', None, None, 1)
    
    #Attd vs variables over days
    if not os.path.isfile('JulianDayPlots/fTempWeek.png'):
        make_plots(grouped_df, 'JulianDay', ['KLAXMaxT', 'KVNYMaxT'], ['Attendance'], 'JulianDayPlots/fTempWeek.png', 50, 120, 0.7)

elif filter_choice == "no":
    grouped_df = grouped_df_orig
    grouped_df_scaled = grouped_df_scaled_orig
    print('Regression Time!')

#Prepare data
dataWeekend = prepare_data(grouped_df, 'Weekend')
dataWeekday = prepare_data(grouped_df, 'Weekday')

dataWeekend_orig = prepare_data(grouped_df_orig, 'Weekend')
dataWeekday_orig = prepare_data(grouped_df_orig, 'Weekday')

#Drop Nones for regression modeling
weekendNoNone = dataWeekend.dropna()
weekdayNoNone = dataWeekday.dropna()

# Perform regression for both weekends and weekdays
weekend, score_name = rm.perform_regression(weekendNoNone, statistics, 1)
weekday, score_name = rm.perform_regression(weekdayNoNone, statistics, 0)


# Display results
print("\nWeekend:")
for model_name, model_data in weekend.items():
    print(f"{model_data['Index']}. {model_name}: {score_name} = {model_data[score_name]:.4f}")
    print(model_data['Display Equation'])
    print(model_data['Simple Equation'])
    print()

print("\nWeekday:")
for model_name, model_data in weekday.items():
    print(f"{model_data['Index']}. {model_name}: {score_name} = {model_data[score_name]:.4f}")
    print(model_data['Display Equation'])
    print(model_data['Simple Equation'])
    print()

# Display equations
while True:
    equation_choice = input("\nEnter the index number of the model for which you want to see the equation: ").strip()
    try:
        equation_choice = int(equation_choice)
        break
    except ValueError:
        print("Invalid input, please try again.")

for model_data in weekend.values():
    if model_data['Index'] == equation_choice:
        print(f"\nWeekend {model_data['Display Equation']}")
        print(f"Weekend {model_data['Simple Equation']}")
        scoreEnd = f"{model_data[score_name]:.4f}"
        break

for model_data in weekday.values():
    if model_data['Index'] == equation_choice:
        print(f"\nWeekday {model_data['Display Equation']}")
        print(f"Weekday {model_data['Simple Equation']}")
        scoreDay = f"{model_data[score_name]:.4f}"
        break

grouped_df_scaled.reset_index(inplace=True)
print('Plotting:',grouped_df_scaled['Area'].unique())


for area_choice in grouped_df_scaled['Area'].unique():
    area_data_weekend = dataWeekend.loc[dataWeekend['Area'] == area_choice]
    area_data_weekday = dataWeekday.loc[dataWeekday['Area'] == area_choice]

    top_equation_end = list(weekend.values())[0]['Simple Equation']
    top_equation_day = list(weekday.values())[0]['Simple Equation']
    
    top_equation_end_display = top_equation_end.replace('JulianFactor', julianFactorReplacement) 
    top_equation_day_display = top_equation_day.replace('JulianFactor', julianFactorReplacement) 
    

    # Extract variable names from equation, excluding "abs" function
    variables_end = re.findall(r'\b(?!abs)[A-Za-z]+\s?[A-Za-z]*\b', top_equation_end.split('=', 1)[-1])
    variables_day = re.findall(r'\b(?!abs)[A-Za-z]+\s?[A-Za-z]*\b', top_equation_day.split('=', 1)[-1])

    variables_values_end = {}
    for var in variables_end:
        variables_values_end[var] = area_data_weekend[var].values.reshape(-1, 1)
    
    variables_values_day = {}
    for var in variables_day:
        variables_values_day[var] = area_data_weekday[var].values.reshape(-1, 1)


    # Evaluate the expressions on the right side of the equations using the variable values
    predicted_y_end = eval(top_equation_end.split('=', 1)[-1], variables_values_end)
    predicted_y_day = eval(top_equation_day.split('=', 1)[-1], variables_values_day)

    # Plot the actual and predicted attendance for the chosen area
    x_weekend = area_data_weekend['JulianDay'].values.reshape(-1, 1)
    y_weekend = area_data_weekend['Attendance'].values.reshape(-1, 1)
    x_weekday = area_data_weekday['JulianDay'].values.reshape(-1, 1)
    y_weekday = area_data_weekday['Attendance'].values.reshape(-1, 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.5, top=0.85)

    size = 15
    labelSize = 16
    figtitleSize = 20

    ax1.scatter(x_weekend, y_weekend, color='b', label='Observed Attendance', marker='o', s=size+10)
    ax1.scatter(x_weekend, predicted_y_end, color='r', label='Predicted Attendance', marker='^', s=size)
    ax1.set_xlabel('Day of Year', fontsize=labelSize)
    ax1.set_ylabel('Attendance', fontsize=labelSize)
    ax1.set_title('Weekend\n\n', loc='center', fontweight='bold', fontsize=labelSize)
    ax1.text(0.5, 1.17, f"{top_equation_end_display}\n{score_name} Score: {scoreEnd}", transform=ax1.transAxes, fontweight='normal', fontsize=14, va='top', ha='center')
    ax1.legend(markerscale=1.5, loc='upper left', fontsize=13)
    ax1.grid(True, alpha=0.5)

    ax2.scatter(x_weekday, y_weekday, color='b', label='Observed Attendance', marker='o', s=size+10)
    ax2.scatter(x_weekday, predicted_y_day, color='r', label='Predicted Attendance', marker='^', s=size)
    ax2.set_xlabel('Day of Year', fontsize=labelSize)
    ax2.set_ylabel('Attendance', fontsize=labelSize)
    ax2.set_title('Weekday\n\n', loc='center', fontweight='bold', fontsize=labelSize)
    ax2.text(0.5, 1.17, f"{top_equation_day_display}\n{score_name} Score: {scoreDay}", transform=ax2.transAxes, fontweight='normal', fontsize=14, va='top', ha='center')
    ax2.legend(markerscale=1.5, loc='upper left', fontsize=13)
    ax2.grid(True, alpha=0.5)

    plt.suptitle(f"Observed vs Predicted Attendance for {area_choice}", fontweight='bold', fontsize = figtitleSize)
    plt.savefig(f'PredictedAttdAreas/{area_choice}Pred.png')
    plt.close()



    percentileEnd = PI.raw2percent(predicted_y_end, area_choice)
    percentileDay = PI.raw2percent(predicted_y_day, area_choice)
    
    indexEnd = PI.percent2index(predicted_y_end, area_choice)
    indexDay = PI.percent2index(predicted_y_day, area_choice)

otherloc = ['Zuma','Manhattan']


for location in otherloc:
    newattEnd = PI.percent2raw(percentileEnd,location)
    newattDay = PI.percent2raw(percentileDay,location)
    
    newIndexEnd = PI.percent2index(percentileEnd, location)
    newIndexDay = PI.percent2index(percentileDay, location)
    
    # Get Julian Day values for reference location
    Julian_end_ref = np.ravel(x_weekend)
    Julian_day_ref = np.ravel(x_weekday)
    
    # Filter data to include only common Julian Day values
    dataWeekend_f = dataWeekend_orig.loc[(dataWeekend_orig['Area'] == location)].reset_index(drop=True)
    dataWeekday_f = dataWeekday_orig.loc[(dataWeekday_orig['Area'] == location)].reset_index(drop=True)

    obs_end = dataWeekend_f['Attendance']
    obs_day = dataWeekday_f['Attendance']
    
    Julian_end = dataWeekend_f['JulianDay']
    Julian_day = dataWeekday_f['JulianDay']
    
    obsIndexEnd = PI.percent2index(PI.raw2percent(obs_end, location),location)
    obsIndexDay = PI.percent2index(PI.raw2percent(obs_day, location),location)

    # Plot pred attd bars with obs attd
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.50, top=0.90)
    plt.suptitle(f'{location} - Observed Attendance vs Predicted Attendance Range', ha = 'center', fontweight='bold', fontsize=20)
    
 
    # Set axis labels and ticks
    ax1.set_ylabel('Attendance', fontsize=16)
    ax1.set_xlabel('Day of Year', fontsize=16)
    ax1.set_title('Weekend', fontweight='bold', fontsize=16, loc = 'center')
    ax1.grid(True, alpha=0.5)
    
    ax2.set_ylabel('Attendance', fontsize=16)
    ax2.set_xlabel('Day of Year', fontsize=16)
    ax2.set_title('Weekday', fontweight='bold', fontsize=16, loc = 'center')
    ax2.grid(True, alpha=0.5)

    bar_width=0.6
    
    for i in range(len(Julian_end_ref)):
        if isinstance(newattEnd[i], str):
            b = int(newattEnd[i].split('-')[0])
            t = int(newattEnd[i].split('-')[1])
            ax1.bar(Julian_end_ref[i], t - b, bottom=b, width=bar_width, align='center', edgecolor='white', color='red', alpha=0.7, label='Predicted Attendance Range' if i == 1 else None)
    
    ax1.scatter(Julian_end, obs_end, color = 'black', label='Observed Attendance', zorder=2, alpha=0.7)

    for i in range(len(Julian_day_ref)):
        if isinstance(newattDay[i], str):
            b = int(newattDay[i].split('-')[0])
            t = int(newattDay[i].split('-')[1])
            ax2.bar(Julian_day_ref[i], t - b, bottom=b, width=bar_width, align='center', edgecolor='white', color='red', alpha=0.7)
    
    ax2.scatter(Julian_day, obs_day, color = 'black', label='Observed Attendance', zorder=2, alpha=0.7)
    
    handles, labels = ax1.get_legend_handles_labels()
    plt.legend(handles, labels, loc='center', ncol=2, markerscale=1, fontsize=14, bbox_to_anchor=(0.5, 1.20))
    
    plt.savefig(f'OtherLocPredAreas/{location}PredAttdByPercentile.png')
    plt.close()
    
    # Create figure for index comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.5, top=0.85)
    plt.suptitle(f'Coastal Operations and Safety Index (COSI) \n{location} Beach - Observed Index vs Predicted Index', ha='center', fontweight='bold', fontsize=20)
    
    # Set axis labels and ticks
    ax1.set_ylabel('COSI', fontsize=16)
    ax1.set_xlabel('Day of Year', fontsize=16)
    ax1.set_title(f'Weekend', fontweight='bold', fontsize=16, loc='center')
    ax1.grid(True, alpha=0.5)
    ax1.set_ylim(0, 10)  # Set y-axis limits
    
    ax2.set_ylabel('COSI', fontsize=16)
    ax2.set_xlabel('Day of Year', fontsize=16)
    ax2.set_title(f'Weekday', fontweight='bold', fontsize=16, loc='center')
    ax2.grid(True, alpha=0.5)
    ax2.set_ylim(0, 10)  # Set y-axis limits
    
    
    ax1.scatter(Julian_end, obsIndexEnd, color='black', label='Observed Index', zorder=1, alpha=0.7, s=25)
    ax1.scatter(Julian_end, newIndexEnd, color='red', label='Predicted Index', zorder=2, alpha=0.7, s=15)
    
    ax2.scatter(Julian_day, obsIndexDay, color='black', label='Observed Index', zorder=1, alpha=0.7, s=25)
    ax2.scatter(Julian_day, newIndexDay, color='red', label='Predicted Index', zorder=2, alpha=0.7, s=15)
    
    
    handles, labels = ax1.get_legend_handles_labels()
    plt.legend(handles, labels, loc='center', ncol=2, markerscale=1.5, fontsize=14, bbox_to_anchor=(0.5, 1.20))
    
    plt.savefig(f'OtherLocPredAreas/{location}PredIndexByPercentile.png')
    plt.close()
    
    

    