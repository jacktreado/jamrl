from datetime import datetime
import os

def create_camp_plot_folder(
    camp_name : str,
    camp_loc : str = "/Users/jacktreado/Dresden/Projects/JamRL/campaigns"
) -> str:
    """createFolder: run to create data folder if it does not exist already
    """
    # create plot_loc
    camp_dir = os.path.join(camp_loc, camp_name)
    if not os.path.exists(camp_dir):
        raise FileNotFoundError(f"No camp_dir {camp_dir} found for campaign {camp_name}")
    else:
        plot_loc = os.path.join(camp_dir, "plots/")
        if not os.path.isdir(plot_loc):
            print(f"HELLO WORLD! Creating `plots` directory for campaign {camp_name} in {plot_loc}")
            os.mkdir(plot_loc)
    
    # Save directories determined by date
    day_str             = datetime.today().strftime('%d')
    month_str           = datetime.today().strftime('%m')
    year_str            = datetime.today().strftime('%Y')

    # -- Year
    year_dir            = plot_loc + year_str + '/'
    if not os.path.isdir(year_dir):
        print("HAPPY NEW YEAR...Year directory ", year_dir, " DOES NOT EXIST, so making!")
        os.mkdir(year_dir)
    else:
        print("It is the year ", year_str, " and the correct year directory exists, so saving in this year's directory...")

    # -- Month
    month_dir            = year_dir + year_str + '-' + month_str + '/'
    if not os.path.isdir(month_dir):
        print("NEW MONTH FOUND!...Month directory ", month_dir, " DOES NOT EXIST, so making!")
        os.mkdir(month_dir)
    else:
        print("It is the month ", year_str + "-" + month_str, " and the correct month directory exists, so saving in this month's directory...")
        
    # -- Day
    day_dir             = month_dir + year_str + '-' + month_str + '-' + day_str + '/'
    if not os.path.isdir(day_dir):
        print("TODAY IS A NEW DAY!...Plotting directory ", day_dir, " DOES NOT EXIST, so making!")
        os.mkdir(day_dir)
    else:
        print("It is the day ", year_str + '-' + month_str + '-' + day_str, " and the correct directory exists, so saving in this day's directory...")
        
    
    # -- Return day dir 
    return day_dir