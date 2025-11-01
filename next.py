# -- IMPORTS

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

from darts import TimeSeries, concatenate
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import torch

# -- CONSTANTS

MINUTES_PER_DAY = 1440
TIME_RESOLUTION_MIN = 5

# -- FUNCTIONS

def is_leap_year( year ):
    return ( year % 4 == 0 and year % 100 != 0 ) or ( year % 400 == 0 )


def days_in_year( year ):
    return 366 if is_leap_year( year ) else 365


def load_subject_table( filepath ):
    try:
        subject_data_frame = pd.read_csv( filepath, header = 0 )
    except pd.errors.EmptyDataError:
        return pd.DataFrame( columns = [ "category", "subject" ] )

    subject_data_frame = subject_data_frame.astype( {
        "category": str,
        "subject": str,
    } )

    return set( zip( subject_data_frame[ "category" ], subject_data_frame[ "subject" ] ) ), subject_data_frame


def load_activity_table( filepath ):
    if not os.path.isfile( filepath ) or os.path.getsize( filepath ) == 0:
        return pd.DataFrame( columns = [ "year", "year_day", "week_day", "hour", "minute", "duration", "category", "subject" ] )

    with open( filepath, 'r' ) as file_handle:
        first_line = file_handle.readline().strip()

    try:
        activity_data_frame = pd.read_csv( filepath, header = 0 )
    except pd.errors.EmptyDataError:
        return pd.DataFrame( columns = [ "year", "year_day", "week_day", "hour", "minute", "duration", "category", "subject" ] )

    type_dictionary = {
        "year": int,
        "year_day": int,
        "week_day": int,
        "hour": int,
        "minute": int,
        "duration": int,
        "category": str,
        "subject": str,
    }

    activity_data_frame = activity_data_frame.astype( type_dictionary )

    return activity_data_frame


def split_activities_spanning_midnight( activity_data_frame ):
    if activity_data_frame.empty:
        return activity_data_frame
    
    split_activity_row_array = []
    
    for _, activity_row in activity_data_frame.iterrows():
        start_minute_index = activity_row[ "hour" ] * 60 + activity_row[ "minute" ]
        duration_minutes = activity_row[ "duration" ]
        end_minute_index = start_minute_index + duration_minutes
        
        if end_minute_index <= MINUTES_PER_DAY:
            split_activity_row_array.append( activity_row.to_dict() )
        else:
            part_one_duration = MINUTES_PER_DAY - start_minute_index
            
            if part_one_duration > 0:
                part_one = activity_row.copy()
                part_one[ "duration" ] = part_one_duration
                split_activity_row_array.append( part_one.to_dict() )
            
            part_two_duration = duration_minutes - part_one_duration
            
            if part_two_duration > 0:
                part_two = activity_row.copy()
                part_two[ "hour" ] = 0
                part_two[ "minute" ] = 0
                part_two[ "duration" ] = part_two_duration
                part_two[ "year_day" ] = activity_row[ "year_day" ] + 1
                try:
                    year_value = int( activity_row[ "year" ] )
                    base_date = datetime( year_value, 1, 1 ) + timedelta( days = int( part_two[ "year_day" ] ) )
                    part_two[ "week_day" ] = base_date.weekday()
                except ( ValueError, OverflowError ):
                    part_two[ "week_day" ] = ( activity_row[ "week_day" ] + 1 ) % 7
                split_activity_row_array.append( part_two.to_dict() )
    
    if split_activity_row_array:
        return pd.DataFrame( split_activity_row_array )
    else:
        return activity_data_frame.copy()


def save_activity_table( activity_data_frame, filepath ):
    if activity_data_frame.empty:
        open( filepath, 'w' ).close()
        return

    column_name_array = [ "year", "year_day", "week_day", "hour", "minute", "duration", "category", "subject" ]

    output_activity_data_frame = activity_data_frame[ column_name_array ].copy()

    output_activity_data_frame[ "year" ] = output_activity_data_frame[ "year" ].astype( int )
    output_activity_data_frame[ "year_day" ] = output_activity_data_frame[ "year_day" ].astype( int )
    output_activity_data_frame[ "week_day" ] = output_activity_data_frame[ "week_day" ].astype( int )
    output_activity_data_frame[ "hour" ] = output_activity_data_frame[ "hour" ].astype( int )
    output_activity_data_frame[ "minute" ] = output_activity_data_frame[ "minute" ].astype( int )
    output_activity_data_frame[ "duration" ] = output_activity_data_frame[ "duration" ].astype( int )

    output_activity_data_frame.to_csv(
        filepath,
        index = False,
        header = True,
        quoting = 2,
        quotechar = '"',
        escapechar = None,
        doublequote = True,
        lineterminator = '\n'
    )


def validate_activity_table( activity_data_frame, valid_subjects ):
    if activity_data_frame.empty:
        return

    for _, activity_row in activity_data_frame.iterrows():
        if pd.isna( activity_row[ "year" ] ) or not isinstance( activity_row[ "year" ], ( int, np.integer ) ):
            raise ValueError( f"Invalid year: {activity_row['year']}" )
        year_value = int( activity_row[ "year" ] )
        
        if year_value < 1900 or year_value > 2100:
            raise ValueError( f"Year out of reasonable range: {year_value}" )

    for _, activity_row in activity_data_frame.iterrows():
        year_value = int( activity_row[ "year" ] )
        maximum_year_day = days_in_year( year_value ) - 1
        
        if not ( 0 <= activity_row[ "year_day" ] <= maximum_year_day ):
            raise ValueError( f"Invalid year_day: {activity_row['year_day']} (max allowed: {maximum_year_day} for year {year_value})" )

        if not ( 0 <= activity_row[ "week_day" ] <= 6 ):
            raise ValueError( f"Invalid week_day: {activity_row['week_day']}" )

        if not ( 0 <= activity_row[ "hour" ] <= 23 ):
            raise ValueError( f"Invalid hour: {activity_row['hour']}" )

        if not ( 0 <= activity_row[ "minute" ] <= 55 ):
            raise ValueError( f"Invalid minute: {activity_row['minute']}" )

        if not ( 5 <= activity_row[ "duration" ] <= MINUTES_PER_DAY ):
            raise ValueError( f"Invalid duration: {activity_row['duration']}" )

        if activity_row[ "minute" ] % TIME_RESOLUTION_MIN != 0:
            raise ValueError( f"minute must be multiple of {TIME_RESOLUTION_MIN}: {activity_row['minute']}" )

        if activity_row[ "duration" ] % TIME_RESOLUTION_MIN != 0:
            raise ValueError( f"duration must be multiple of {TIME_RESOLUTION_MIN}: {activity_row['duration']}" )

        try:
            date_value = datetime( year_value, 1, 1 ) + timedelta( days = int( activity_row[ "year_day" ] ) )
            actual_weekday = date_value.weekday()
            
            if int( activity_row[ "week_day" ] ) != actual_weekday:
                raise ValueError( f"week_day mismatch: {activity_row['week_day']} does not match actual weekday {actual_weekday} for year_day {activity_row['year_day']} in year {year_value}" )
        except ( ValueError, OverflowError ) as error:
            raise ValueError( f"Invalid date calculation for year_day {activity_row['year_day']} in year {year_value}: {error}" )

        if ( activity_row[ "category" ], activity_row[ "subject" ] ) not in valid_subjects:
            raise ValueError( f"Invalid subject pair: ({activity_row['category']}, {activity_row['subject']})" )


def enforce_no_overlap_and_bounds( activity_row_array ):
    if not activity_row_array:
        return []
    
    activity_row_array.sort( key = lambda activity_row: activity_row[ "hour" ] * 60 + activity_row[ "minute" ] )
    valid_activity_row_array = []
    current_end_minute_index = 0
    
    for activity_row in activity_row_array:
        start_minute_index = activity_row[ "hour" ] * 60 + activity_row[ "minute" ]
        duration_minutes = activity_row[ "duration" ]
        
        if start_minute_index < current_end_minute_index:
            continue
        valid_activity_row_array.append( activity_row )
        current_end_minute_index = start_minute_index + duration_minutes
    
    return valid_activity_row_array


def create_dense_timeline( activity_data_frame ):
    if activity_data_frame.empty:
        raise ValueError( "No valid training data." )

    activity_data_frame[ "label" ] = activity_data_frame[ "category" ] + "||" + activity_data_frame[ "subject" ]
    label_string_array = sorted( set( activity_data_frame[ "label" ] ) )
    label_index_by_label_string_map = { label_string: label_index for label_index, label_string in enumerate( label_string_array ) }
    activity_data_frame[ "label_id" ] = activity_data_frame[ "label" ].map( label_index_by_label_string_map )

    if not activity_data_frame[ "year" ].isna().all():
        year_value_array = activity_data_frame[ "year" ].dropna()
        
        if len( year_value_array ) > 0:
            mode_value_array = year_value_array.mode()
            
            if len( mode_value_array ) > 0:
                actual_year = int( mode_value_array.iloc[ 0 ] )
            else:
                actual_year = int( year_value_array.iloc[ 0 ] )
        else:
            actual_year = 2023
    else:
        actual_year = 2023

    label_index_by_minute_index_by_day_index_map = {}
    
    for _, activity_row in activity_data_frame.iterrows():
        day_index = activity_row[ "year_day" ]
        start_minute_index = activity_row[ "hour" ] * 60 + activity_row[ "minute" ]
        duration_minutes = activity_row[ "duration" ]
        label_index = activity_row[ "label_id" ]
        
        if day_index not in label_index_by_minute_index_by_day_index_map:
            label_index_by_minute_index_by_day_index_map[ day_index ] = {}
        
        end_minute_index = start_minute_index + duration_minutes
        end_minute_index = min( end_minute_index, MINUTES_PER_DAY )
        
        for minute_index in range( start_minute_index, end_minute_index ):
            label_index_by_minute_index_by_day_index_map[ day_index ][ minute_index ] = label_index

    timeline_record_array = []
    minimum_day_index = activity_data_frame[ "year_day" ].min()
    data_frame_maximum_day_index = activity_data_frame[ "year_day" ].max()
    map_maximum_day_index = max( label_index_by_minute_index_by_day_index_map.keys() ) if label_index_by_minute_index_by_day_index_map else data_frame_maximum_day_index
    maximum_day_index = max( data_frame_maximum_day_index, map_maximum_day_index )

    for day_index in range( minimum_day_index, maximum_day_index + 1 ):
        base_date = datetime( actual_year, 1, 1 ) + timedelta( days = day_index )
        label_index_by_minute_index_map = label_index_by_minute_index_by_day_index_map.get( day_index, {} )
        
        for minute_index in range( 0, MINUTES_PER_DAY, TIME_RESOLUTION_MIN ):
            label_index = label_index_by_minute_index_map.get( minute_index, -1 )
            timeline_record = {
                "timestamp": base_date + timedelta( minutes = minute_index ),
                "week_day": base_date.weekday(),
            }
            
            for label_string, mapped_label_index in label_index_by_label_string_map.items():
                timeline_record[ label_string ] = 1.0 if mapped_label_index == label_index else 0.0
            timeline_record_array.append( timeline_record )

    dense_timeline_data_frame = pd.DataFrame( timeline_record_array ).set_index( "timestamp" ).sort_index()
    return dense_timeline_data_frame, label_index_by_label_string_map, { mapped_label_index: label_string for label_string, mapped_label_index in label_index_by_label_string_map.items() }, actual_year


def train_model( model_name, subject_table_path, activity_table_path ):
    valid_subjects, _ = load_subject_table( subject_table_path )
    activity_data_frame = load_activity_table( activity_table_path )
    validate_activity_table( activity_data_frame, valid_subjects )

    if activity_data_frame.empty:
        print( "No valid activity data to train on." )
        return

    split_activity_data_frame = split_activities_spanning_midnight( activity_data_frame )
    validate_activity_table( split_activity_data_frame, valid_subjects )

    dense_timeline_data_frame, label_index_by_label_string_map, label_string_by_index_map, actual_year = create_dense_timeline( split_activity_data_frame )

    label_string_array = [ label_string_by_index_map[ label_index ] for label_index in range( len( label_string_by_index_map ) ) ]
    target_series = TimeSeries.from_dataframe( dense_timeline_data_frame[ label_string_array ], freq = f"{TIME_RESOLUTION_MIN}min" )
    target_series = target_series.astype( np.float32 )

    dayofweek_covariate = datetime_attribute_timeseries( target_series.time_index, attribute = "dayofweek", one_hot = False )
    month_covariate = datetime_attribute_timeseries( target_series.time_index, attribute = "month", one_hot = False )
    month_covariate = ( month_covariate / 12.0 ).astype( np.float32 )
    future_covariate = concatenate( [ dayofweek_covariate, month_covariate ], axis = 1 ).astype( np.float32 )

    hour_covariate = datetime_attribute_timeseries( target_series.time_index, attribute = "hour", one_hot = False )
    minute_covariate = datetime_attribute_timeseries( target_series.time_index, attribute = "minute", one_hot = False )
    hour_covariate = ( hour_covariate / 23.0 ).astype( np.float32 )
    minute_covariate = ( minute_covariate / 59.0 ).astype( np.float32 )
    past_covariate = concatenate( [ hour_covariate, minute_covariate ], axis = 1 )

    scaler = Scaler()
    scaled_target_series = scaler.fit_transform( target_series )

    trainer_keyword_arguments = { "accelerator": "gpu", "devices": 1 } if torch.cuda.is_available() else { "accelerator": "cpu" }

    model = TFTModel(
        input_chunk_length = 288,
        output_chunk_length = 288,
        hidden_size = 128,
        lstm_layers = 2,
        num_attention_heads = 8,
        dropout = 0.1,
        batch_size = 32,
        n_epochs = 100,
        likelihood = QuantileRegression( quantiles = [ 0.1, 0.5, 0.9 ] ),
        random_state = 42,
        force_reset = True,
        save_checkpoints = False,
        pl_trainer_kwargs = trainer_keyword_arguments,
        )

    model.fit( series = scaled_target_series, past_covariates = past_covariate, future_covariates = future_covariate, verbose = True )

    os.makedirs( "models", exist_ok = True )
    model.save( f"models/{model_name}.pt" )
    
    with open( f"models/{model_name}_meta.pkl", "wb" ) as file_handle:
        pickle.dump( {
            "id_to_label": label_string_by_index_map,
            "labels": [ label_string_by_index_map[ label_index ] for label_index in range( len( label_string_by_index_map ) ) ],
            "valid_subjects": valid_subjects,
            "scaler": scaler,
            "time_res_min": TIME_RESOLUTION_MIN,
            "last_known_day": int( activity_data_frame[ "year_day" ].max() ),
            "has_past_cov": True,
            "training_year": actual_year,
        }, file_handle )

    print( f"✅ Model '{model_name}' trained on {len( activity_data_frame )} activities." )


def predict_future( model_name, subject_table_path, future_activity_path, day_count ):
    model = TFTModel.load( f"models/{model_name}.pt" )
    
    with open( f"models/{model_name}_meta.pkl", "rb" ) as file_handle:
        model_metadata = pickle.load( file_handle )

    label_string_by_index_map = model_metadata[ "id_to_label" ]
    label_string_array = model_metadata.get( "labels", [ label_string_by_index_map[ label_index ] for label_index in range( len( label_string_by_index_map ) ) ] )
    valid_subjects = model_metadata[ "valid_subjects" ]
    scaler = model_metadata[ "scaler" ]
    time_resolution = model_metadata[ "time_res_min" ]
    last_known_day_index = model_metadata[ "last_known_day" ]
    training_year = model_metadata.get( "training_year", 2023 )

    start_day_of_year_index = last_known_day_index + 1
    start_year = training_year

    maximum_days_in_training_year = days_in_year( training_year ) - 1

    if start_day_of_year_index > maximum_days_in_training_year:
        days_over_count = start_day_of_year_index - ( maximum_days_in_training_year + 1 )
        start_year = training_year + 1
        start_day_of_year_index = days_over_count

        maximum_days_in_next_year = days_in_year( start_year ) - 1
        maximum_prediction_day_index = start_day_of_year_index + int( day_count ) - 1
        
        if maximum_prediction_day_index > maximum_days_in_next_year:
            raise ValueError( f"Prediction goes to day {maximum_prediction_day_index}, but max allowed is {maximum_days_in_next_year} for year {start_year}." )
    else:
        maximum_prediction_day_index = last_known_day_index + int( day_count )
        
        if maximum_prediction_day_index > maximum_days_in_training_year:
            raise ValueError( f"Prediction goes to day {maximum_prediction_day_index}, but max allowed is {maximum_days_in_training_year} for year {training_year}." )

    start_date = datetime( start_year, 1, 1 ) + timedelta( days = start_day_of_year_index )
    start_date = start_date.replace( hour = 0, minute = 0, second = 0, microsecond = 0 )
    total_step_count = ( int( day_count ) * MINUTES_PER_DAY ) // time_resolution

    input_chunk_length = getattr( model, "input_chunk_length", 0 ) or 0
    output_chunk_length = getattr( model, "output_chunk_length", 0 ) or 0
    
    past_covariate_start_date = start_date - timedelta( minutes = input_chunk_length * time_resolution )
    past_covariate_period_count = input_chunk_length + total_step_count if input_chunk_length > 0 else total_step_count
    
    future_covariate_start_date = start_date
    
    if total_step_count > output_chunk_length and output_chunk_length > 0:
        future_covariate_period_count = total_step_count + output_chunk_length
    else:
        future_covariate_period_count = total_step_count

    future_timestamp_array = pd.date_range( start = start_date, periods = total_step_count, freq = f"{time_resolution}min" )
    past_covariate_timestamp_array = pd.date_range( start = past_covariate_start_date, periods = past_covariate_period_count, freq = f"{time_resolution}min" )
    future_covariate_timestamp_array = pd.date_range( start = future_covariate_start_date, periods = future_covariate_period_count, freq = f"{time_resolution}min" )

    dayofweek_covariate = datetime_attribute_timeseries( future_covariate_timestamp_array, attribute = "dayofweek", one_hot = False )
    month_covariate = datetime_attribute_timeseries( future_covariate_timestamp_array, attribute = "month", one_hot = False )
    month_covariate = ( month_covariate / 12.0 ).astype( np.float32 )
    future_covariate = concatenate( [ dayofweek_covariate, month_covariate ], axis = 1 ).astype( np.float32 )

    has_past_covariate = model_metadata.get( "has_past_cov", False )
    
    if has_past_covariate:
        hour_covariate = datetime_attribute_timeseries( past_covariate_timestamp_array, attribute = "hour", one_hot = False )
        minute_covariate = datetime_attribute_timeseries( past_covariate_timestamp_array, attribute = "minute", one_hot = False )
        hour_covariate = ( hour_covariate / 23.0 ).astype( np.float32 )
        minute_covariate = ( minute_covariate / 59.0 ).astype( np.float32 )
        past_covariate = concatenate( [ hour_covariate, minute_covariate ], axis = 1 )
    else:
        past_covariate = None

    if past_covariate is not None:
        scaled_prediction = model.predict( n = total_step_count, past_covariates = past_covariate, future_covariates = future_covariate, num_samples = 100 )
    else:
        scaled_prediction = model.predict( n = total_step_count, future_covariates = future_covariate, num_samples = 100 )
    median_prediction = scaled_prediction.quantile( 0.5 )
    raw_prediction = scaler.inverse_transform( median_prediction )

    probability_value_array = raw_prediction.values( copy = False )

    threshold = 0.3
    filtered_probability_value_array = probability_value_array.copy()
    filtered_probability_value_array[ filtered_probability_value_array < threshold ] = 0.0

    predicted_label_index_array = np.argmax( filtered_probability_value_array, axis = 1 ) if probability_value_array.ndim == 2 else np.zeros( ( total_step_count, ), dtype = int )

    maximum_probability_value_array = np.max( filtered_probability_value_array, axis = 1 ) if probability_value_array.ndim == 2 else np.zeros( ( total_step_count, ) )
    fallback_mask = maximum_probability_value_array < threshold
    
    if np.any( fallback_mask ):
        predicted_label_index_array[ fallback_mask ] = np.argmax( probability_value_array[ fallback_mask ], axis = 1 ) if probability_value_array.ndim == 2 else 0

    minimum_step_count = max( 1, ( 10 // time_resolution ) )
    label_run_array = []
    
    for label_index in predicted_label_index_array:
        if not label_run_array or label_run_array[ -1 ][ 0 ] != label_index:
            label_run_array.append( [ label_index, 1 ] )
        else:
            label_run_array[ -1 ][ 1 ] += 1
    
    run_index = 0
    
    while run_index < len( label_run_array ):
        label_index, run_length = label_run_array[ run_index ]
        
        if run_length < minimum_step_count:
            if run_index > 0:
                label_run_array[ run_index - 1 ][ 1 ] += run_length
                label_run_array.pop( run_index )
                continue
            elif run_index + 1 < len( label_run_array ):
                label_run_array[ run_index + 1 ][ 1 ] += run_length
                label_run_array.pop( run_index )
                continue
        run_index += 1

    smoothed_label_index_array = []
    
    for label_index, run_length in label_run_array:
        smoothed_label_index_array.extend( [ label_index ] * run_length )
    
    if len( smoothed_label_index_array ) < total_step_count:
        smoothed_label_index_array.extend( [ smoothed_label_index_array[ -1 ] ] * ( total_step_count - len( smoothed_label_index_array ) ) )
    smoothed_label_index_array = np.array( smoothed_label_index_array[ :total_step_count ], dtype = int )

    final_activity_row_array = []
    current_activity_block = None
    
    for step_index, timestamp in enumerate( future_timestamp_array ):
        label_index = smoothed_label_index_array[ step_index ]
        label_string = label_string_array[ label_index ]
        
        try:
            category_string, subject_string = label_string.split( "||", 1 )
        except ValueError:
            continue
        
        if ( category_string, subject_string ) not in valid_subjects:
            continue

        year_start_date = datetime( timestamp.year, 1, 1 )
        year_day_index = ( timestamp - year_start_date ).days
        week_day_index = timestamp.weekday()
        hour_value = timestamp.hour
        minute_value = timestamp.minute

        if current_activity_block and current_activity_block[ "category" ] == category_string and current_activity_block[ "subject" ] == subject_string:
            current_activity_block[ "duration" ] += time_resolution
        else:
            if current_activity_block:
                final_activity_row_array.append( current_activity_block )
            current_activity_block = {
                "year": timestamp.year,
                "year_day": year_day_index,
                "week_day": week_day_index,
                "hour": hour_value,
                "minute": minute_value,
                "duration": time_resolution,
                "category": category_string,
                "subject": subject_string,
            }

    if current_activity_block:
        final_activity_row_array.append( current_activity_block )

    output_activity_row_array = []
    
    if final_activity_row_array:
        predicted_activity_data_frame = pd.DataFrame( final_activity_row_array )
        predicted_activity_data_frame = predicted_activity_data_frame.sort_values( [ "year", "year_day", "hour", "minute" ] ).reset_index( drop = True )
        
        merged_activity_row_array = []
        previous_day_activity_row_array = None
        
        for ( year_value, day_index ), day_group in predicted_activity_data_frame.groupby( [ "year", "year_day" ] ):
            day_activity_row_array = day_group.to_dict( "records" )
            day_activity_row_array.sort( key = lambda activity_row: activity_row[ "hour" ] * 60 + activity_row[ "minute" ] )
            
            if previous_day_activity_row_array and day_activity_row_array:
                previous_last_activity_row = previous_day_activity_row_array[ -1 ]
                current_first_activity_row = day_activity_row_array[ 0 ]
                
                if ( previous_last_activity_row[ "category" ] == current_first_activity_row[ "category" ] and 
                     previous_last_activity_row[ "subject" ] == current_first_activity_row[ "subject" ] and
                     current_first_activity_row[ "hour" ] == 0 and current_first_activity_row[ "minute" ] == 0 ):
                    
                    previous_start_minute_index = previous_last_activity_row[ "hour" ] * 60 + previous_last_activity_row[ "minute" ]
                    previous_end_minute_index = previous_start_minute_index + previous_last_activity_row[ "duration" ]
                    
                    time_to_midnight_minutes = MINUTES_PER_DAY - previous_end_minute_index
                    
                    if time_to_midnight_minutes >= -60 and time_to_midnight_minutes <= 240:
                        total_duration_minutes = ( MINUTES_PER_DAY - previous_start_minute_index ) + current_first_activity_row[ "duration" ]
                        previous_last_activity_row[ "duration" ] = total_duration_minutes
                        day_activity_row_array = day_activity_row_array[ 1: ]
            
            merged_activity_row_array.extend( previous_day_activity_row_array if previous_day_activity_row_array else [] )
            previous_day_activity_row_array = day_activity_row_array
        
        if previous_day_activity_row_array:
            merged_activity_row_array.extend( previous_day_activity_row_array )
        
        for ( year_value, day_index ), day_group in pd.DataFrame( merged_activity_row_array ).groupby( [ "year", "year_day" ] ):
            day_activity_row_array = day_group.to_dict( "records" )
            valid_activity_row_array = enforce_no_overlap_and_bounds( day_activity_row_array )
            output_activity_row_array.extend( valid_activity_row_array )

    output_activity_data_frame = pd.DataFrame( output_activity_row_array )
    save_activity_table( output_activity_data_frame, future_activity_path )
    print( f"✅ Generated {len( output_activity_data_frame )} valid activities for {day_count} days." )


def main():
    if len( sys.argv ) < 2:
        print( "Usage: python next.py <train|predict> ..." )
        sys.exit( 1 )

    command = sys.argv[ 1 ]
    
    if command == "train":
        if len( sys.argv ) != 5:
            print( "Usage: python next.py train <model_name> <subject_table.csv> <activity_table.csv>" )
            sys.exit( 1 )
        _, _, model_name, subject_csv_path, activity_csv_path = sys.argv
        train_model( model_name, subject_csv_path, activity_csv_path )
    elif command == "predict":
        if len( sys.argv ) != 6:
            print( "Usage: python next.py predict <model_name> <subject_table.csv> <future_activity_table.csv> <day_count>" )
            sys.exit( 1 )
        _, _, model_name, subject_csv_path, future_csv_path, day_count = sys.argv
        predict_future( model_name, subject_csv_path, future_csv_path, day_count )
    else:
        print( "Unknown command. Use 'train' or 'predict'." )
        sys.exit( 1 )

# -- STATEMENTS

if torch.cuda.is_available():
    torch.set_float32_matmul_precision( 'medium' )

if __name__ == "__main__":
    main()
