# -- IMPORTS

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# -- STATEMENTS

script_folder_path = os.path.dirname( os.path.abspath( __file__ ) )
data_folder_path = os.path.join( script_folder_path, "data" )
input_file_path = os.path.join( data_folder_path, "daily_activity_data.csv" )

activity_data_frame = pd.read_csv( input_file_path )

new_activity_array = []

work_activity_array = 
    [
        ( "developing application", 120 ),
        ( "code review", 90 ),
        ( "debugging", 60 ),
        ( "writing documentation", 90 ),
        ( "attending a meeting", 240 ),
        ( "planning sprint", 60 ),
    ]

leisure_activity_array = 
    [
        ( "watching television", 120 ),
        ( "watching tv series", 180 ),
        ( "playing video game", 120 ),
        ( "reading", 60 ),
        ( "browsing internet", 45 ),
        ( "listening to music", 30 ),
    ]

activity_array = activity_data_frame.groupby( [ "year", "year_day" ] ).first().reset_index()

for _, activity_row in activity_array.iterrows():
    year = int( activity_row[ "year" ] )
    year_day = int( activity_row[ "year_day" ] )
    start_date = datetime( year, 1, 1 )
    target_date = start_date + timedelta( days = year_day )
    week_day = target_date.weekday()
    is_workday = week_day < 5
    is_weekend = week_day >= 5

    activity_array = []

    if is_workday:
        activity_array.append( ( 390, 15, "hygiene", "taking a shower" ) )
        activity_array.append( ( 405, 5, "hygiene", "brushing teeth" ) )
        activity_array.append( ( 410, 10, "meal", "drinking coffee" ) )
        activity_array.append( ( 420, 15, "meal", "eating breakfast" ) )
        activity_array.append( ( 435, 30, "transport", "driving" ) )
        activity_array.append( ( 465, 120, "work", "developing application" ) )
        activity_array.append( ( 585, 15, "meal", "drinking coffee" ) )
        activity_array.append( ( 600, 90, "work", "code review" ) )
        activity_array.append( ( 690, 15, "meal", "eating snack" ) )
        activity_array.append( ( 705, 15, "work", "debugging" ) )
        activity_array.append( ( 720, 30, "meal", "eating lunch" ) )
        activity_array.append( ( 750, 240, "work", "attending a meeting" ) )
        activity_array.append( ( 990, 15, "meal", "drinking coffee" ) )
        activity_array.append( ( 1005, 15, "work", "writing documentation" ) )
        activity_array.append( ( 1020, 30, "transport", "driving" ) )
        activity_array.append( ( 1050, 30, "meal", "eating dinner" ) )

        if ( year_day + week_day ) % 3 == 0:
            activity_array.append( ( 1080, 120, "leisure", "playing video game" ) )
        elif ( year_day + week_day ) % 3 == 1:
            activity_array.append( ( 1080, 120, "leisure", "watching tv series" ) )
        else:
            activity_array.append( ( 1080, 120, "leisure", "watching television" ) )

        if ( year_day + week_day ) % 2 == 0:
            activity_array.append( ( 1200, 120, "leisure", "reading" ) )
        else:
            activity_array.append( ( 1200, 120, "leisure", "browsing internet" ) )

        activity_array.append( ( 1320, 5, "hygiene", "brushing teeth" ) )
        activity_array.append( ( 1325, 25, "leisure", "listening to music" ) )
        activity_array.append( ( 1350, 480, "rest", "sleeping" ) )

    else:

        activity_array.append( ( 450, 15, "hygiene", "taking a shower" ) )
        activity_array.append( ( 465, 5, "hygiene", "brushing teeth" ) )
        activity_array.append( ( 470, 10, "meal", "drinking coffee" ) )
        activity_array.append( ( 480, 15, "meal", "eating breakfast" ) )

        if week_day == 5:
            if ( year_day ) % 3 == 0:
                activity_array.append( ( 495, 90, "sport", "playing tennis" ) )
            elif ( year_day ) % 3 == 1:
                activity_array.append( ( 495, 90, "sport", "running" ) )
            else:
                activity_array.append( ( 495, 90, "sport", "going to gym" ) )

            activity_array.append( ( 585, 15, "hygiene", "taking a shower" ) )
            activity_array.append( ( 600, 15, "meal", "drinking coffee" ) )

            if ( year_day ) % 2 == 0:
                activity_array.append( ( 615, 105, "leisure", "reading" ) )
            else:
                activity_array.append( ( 615, 105, "leisure", "browsing internet" ) )
        else:
            if ( year_day ) % 2 == 0:
                activity_array.append( ( 495, 120, "leisure", "reading" ) )
            else:
                activity_array.append( ( 495, 120, "leisure", "browsing internet" ) )

            activity_array.append( ( 615, 15, "meal", "eating snack" ) )
            activity_array.append( ( 630, 90, "leisure", "listening to music" ) )

        activity_array.append( ( 720, 30, "meal", "eating lunch" ) )

        if ( year_day ) % 4 == 0:
            activity_array.append( ( 750, 120, "leisure", "playing video game" ) )
        elif ( year_day ) % 4 == 1:
            activity_array.append( ( 750, 120, "leisure", "watching tv series" ) )
        elif ( year_day ) % 4 == 2:
            activity_array.append( ( 750, 120, "leisure", "watching television" ) )
        else:
            activity_array.append( ( 750, 120, "leisure", "reading" ) )

        activity_array.append( ( 870, 15, "meal", "eating snack" ) )

        if ( year_day ) % 3 == 0:
            activity_array.append( ( 900, 180, "leisure", "watching tv series" ) )
        elif ( year_day ) % 3 == 1:
            activity_array.append( ( 900, 180, "leisure", "playing video game" ) )
        else:
            activity_array.append( ( 900, 180, "leisure", "browsing internet" ) )

        activity_array.append( ( 1080, 30, "meal", "eating dinner" ) )

        if ( year_day ) % 2 == 0:
            activity_array.append( ( 1110, 120, "leisure", "watching tv series" ) )
        else:
            activity_array.append( ( 1110, 120, "leisure", "playing video game" ) )

        if ( year_day ) % 2 == 0:
            activity_array.append( ( 1230, 90, "leisure", "reading" ) )
        else:
            activity_array.append( ( 1230, 90, "leisure", "listening to music" ) )

        activity_array.append( ( 1320, 5, "hygiene", "brushing teeth" ) )
        activity_array.append( ( 1325, 25, "leisure", "browsing internet" ) )
        activity_array.append( ( 1350, 540, "rest", "sleeping" ) )

    for day_minute_index, duration, category, subject in activity_array:
        hour = day_minute_index // 60
        minute = day_minute_index % 60
        minute = round( minute / 5 ) * 5
        minute = min( 55, minute )
        duration = round( duration / 5 ) * 5
        duration = max( 5, duration )

        new_activity_array.append( {
            "year": year,
            "year_day": year_day,
            "week_day": week_day,
            "hour": hour,
            "minute": minute,
            "duration": duration,
            "category": category,
            "subject": subject
        } )

new_activity_data_frame = pd.DataFrame( new_activity_array )
new_activity_data_frame[ "sort_time" ] = new_activity_data_frame[ "hour" ] * 60 + new_activity_data_frame[ "minute" ]
new_activity_data_frame = new_activity_data_frame.sort_values( [ "year", "year_day", "sort_time" ] ).reset_index( drop = True )
new_activity_data_frame = new_activity_data_frame.drop( "sort_time", axis = 1 )

output_file_path = os.path.join( data_folder_path, "daily_activity_data.csv" )
with open( output_file_path, 'w', newline = '', encoding = 'utf-8' ) as f:
    f.write( '"year","year_day","week_day","hour","minute","duration","category","subject"\n' )

    for _, activity_row in new_activity_data_frame.iterrows():
        category = str( activity_row[ "category" ] ).replace( '"', '""' )
        subject = str( activity_row[ "subject" ] ).replace( '"', '""' )
        f.write( f'{int(activity_row[ "year" ])},{int(activity_row[ "year_day" ])},{int(activity_row[ "week_day" ])},'
                f'{int(activity_row[ "hour" ])},{int(activity_row[ "minute" ])},{int(activity_row[ "duration" ])},'
                f'"{category}","{subject}"\n' )

print( f"Updated activity data with {len( new_activity_data_frame )} activities across {new_activity_data_frame[ "year_day" ].nunique()} days" )
print( f"Sleep: 480 minutes (8 hours) on workdays, 540 minutes (9 hours) on weekends" )
print( f"Output saved to {output_file_path}" )
