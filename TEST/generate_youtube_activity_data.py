# -- IMPORTS

import pandas as pd
import random
import os
from datetime import datetime, timedelta

# -- STATEMENTS

script_folder_path = os.path.dirname( os.path.abspath( __file__ ) )
data_folder_path = os.path.join( script_folder_path, "data" )
subject_file_path = os.path.join( data_folder_path, "youtube_video_subject_data.csv" )

subject_data_frame = pd.read_csv( subject_file_path )
subject_array = []

for _, subject_row in subject_data_frame.iterrows():
    category = subject_row[ "category" ]
    subject = subject_row[ "subject" ]
    subject_array.append( ( category, subject ) )

if not subject_array:
    raise ValueError( f"No subjects found in {subject_file_path}" )

print( f"Loaded {len( subject_array )} subjects from {subject_file_path}" )

start_day = 100
end_day = 300
year = 2025
duration_option_array = [ 15, 20, 30, 45, 60 ]

time_pattern_by_name_map = 
    {
        "morning": [ ( 6, 0 ), ( 7, 30 ), ( 8, 15 ), ( 9, 0 ), ( 10, 30 ), ( 11, 45 ) ],
        "afternoon": [ ( 12, 0 ), ( 13, 20 ), ( 14, 10 ), ( 15, 30 ), ( 16, 45 ) ],
        "evening": [ ( 17, 30 ), ( 18, 45 ), ( 19, 15 ), ( 20, 0 ), ( 20, 45 ), ( 21, 30 ) ],
        "night": [ ( 22, 0 ), ( 22, 30 ), ( 23, 15 ) ]
    }

activity_array = []

for day in range( start_day, end_day + 1 ):
    start_date = datetime( year, 1, 1 )
    target_date = start_date + timedelta( days = day )
    weekday = target_date.weekday()

    if weekday in [ 5, 6 ]:
        video_count = random.randint( 2, 5 )
    else:
        video_count = random.randint( 1, 4 )

    if weekday in [ 5, 6 ]:
        pattern_distribution = [ "morning" ] * 2 + [ "afternoon" ] * 2 + [ "evening" ] * 1 + [ "night" ] * 0
    elif weekday == 0:
        pattern_distribution = [ "morning" ] * 1 + [ "afternoon" ] * 2 + [ "evening" ] * 2 + [ "night" ] * 1
    else:
        pattern_distribution = [ "morning" ] * 0 + [ "afternoon" ] * 1 + [ "evening" ] * 3 + [ "night" ] * 1

    for _ in range( video_count ):
        pattern = random.choice( pattern_distribution )
        hour, minute = random.choice( time_pattern_by_name_map[ pattern ] )

        minute += random.randint( -15, 15 )

        if minute < 0:
            minute = 0
        elif minute >= 60:
            minute = 55

        minute = round( minute / 5 ) * 5
        minute = min( 55, minute )
        duration = random.choice( duration_option_array )
        duration = round( duration / 5 ) * 5
        duration = max( 5, duration )

        category, subject = random.choice( subject_array )

        activity_array.append( 
            {
                "year": year,
                "year_day": day,
                "week_day": weekday,
                "hour": hour,
                "minute": minute,
                "duration": duration,
                "category": category,
                "subject": subject
            } 
            )

activity_data_frame = pd.DataFrame( activity_array )
activity_data_frame = activity_data_frame.sort_values( [ "year_day", "hour", "minute" ] ).reset_index( drop = True )

output_file = os.path.join( data_folder_path, "youtube_video_activity_data.csv" )

with open( output_file, "w", newline = "", encoding = "utf-8" ) as f:
    f.write( '"year","year_day","week_day","hour","minute","duration","category","subject"\n' )

    for _, activity_row in activity_data_frame.iterrows():
        category = str( activity_row[ "category" ] ).replace( '"', '""' )
        subject = str( activity_row[ "subject" ] ).replace( '"', '""' )
        f.write( f'{int(activity_row[ "year" ])},{int(activity_row[ "year_day" ])},{int(activity_row[ "week_day" ])},'
                f'{int(activity_row[ "hour" ])},{int(activity_row[ "minute" ])},{int(activity_row[ "duration" ])},'
                f'"{category}","{subject}"\n' )

print( f"Generated {len( activity_data_frame )} activity records for days {start_day}-{end_day}" )
print( f"Output saved to {output_file}" )

