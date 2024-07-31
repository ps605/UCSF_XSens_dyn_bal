%% This function processes accelerometer or force plate data to identify IC and TO events during running

%% Input variables
% sensor_location: string of name of sensor location
% original_data columns BACK: timestamp (ms), ML accelerations (g) (+ points right), AP accelerations (g) (+ points front), VT accelerations (g) (+ points up)
% original_data columns FOOT: timestamp (ms), ML accelerations (g) (+ points left), AP accelerations (g) (+ points back), VT accelerations (g) (+ points up)
% original_data columns FORCE: timestamp (ms), VT force (N) (+ points up)
% new_SR: new sampling rate (Hz)
% window_size: size of windows of data to process (s)
% delta_window: change in window start time (s)
% VISUALIZATION: boolean to see real-time display of data processing
% FINAL_PLOT: boolean to see final plot

%% Output variables
% resampled_data: original_data resampled at new_SR
% resultant: resultant of resampled_data
% output_rate: array with length equal to length of resampled data; each frame is 1/(time to process given frame)
% IC: frames where IC occur within resampled_data
% TO: frames where TO occur within resampled_data
% LEFT_STEP: boolean for each step (IC) where true means step is on left side


function [resampled_data, resultant, output_rate, IC, TO, LEFT_STEP] = getevents(sensor_location, original_data, new_SR, window_size, delta_window, VISUALIZATION, FINAL_PLOT)

%% Sensor-specific considerations
if strcmp(sensor_location, 'back')    
    visualization_plots = [1,2,3,5,6,7];  % identify plots to include in visualization: 1=ML, 2=AP, 3=VT, 4=Resultant, 5=Left Step, 6=IC, 7=TO    
    min_thresh = 0.25*new_SR;  % establish minimum threshold for back gait events at 1/4 of a second, which corresponds to 0.25*SR
    max_thresh = 0.5*new_SR;  % establish maximum threshold for back gait events at 1/2 of a second, which corresponds to 0.5*SR
    IC_axis = 3;  % identify axis to use to establish IC: 1=ML, 2=AP, 3=VT, 4=Resultant; assign negative value if minimum of axis should be used
    TO_axis = -3;  % identify axis to use to establish TO: 1=ML, 2=AP, 3=VT, 4=Resultant; assign negative value if minimum of axis should be used
    FIND_STEP_SIDE = true;  % step side needs to be determined
    find_events_function = @findeventsback;  % identify sensor-specific gait event detection function
elseif strcmp(sensor_location, 'Rfoot')
    visualization_plots = [1,2,3,4,6,7];  % identify plots to include in visualization: 1=ML, 2=AP, 3=VT, 4=Resultant, 5=Left Step, 6=IC, 7=TO
    min_thresh = 0.5*new_SR;  % establish minimum threshold for foot gait events at 1/2 of a second, which corresponds to 0.5*SR
    max_thresh = 1*new_SR;  % establish maximum threshold for foot gait events at 1 second, which corresponds to 1*SR
    IC_axis = 4;  % identify axis to use to establish IC: 1=ML, 2=AP, 3=VT, 4=Resultant; assign negative value if minimum of axis should be used
    TO_axis = -3;  % identify axis to use to establish TO: 1=ML, 2=AP, 3=VT, 4=Resultant; assign negative value if minimum of axis should be used
    FIND_STEP_SIDE = false;  % step side does not need to be determined
    find_events_function = @findeventsfoot;  % identify sensor-specific gait event detection function
elseif strcmp(sensor_location, 'force')
    visualization_plots = [3,6,7];  % identify plots to include in visualization: 1=ML, 2=AP, 3=VT, 4=Resultant, 5=Left Step, 6=IC, 7=TO
    min_thresh = 0.25*new_SR;  % establish minimum threshold for force plate gait events at 1/4 of a second, which corresponds to 0.25*SR
    max_thresh = 0.5*new_SR;  % establish maximum threshold for force plate gait events at 1/2 of a second, which corresponds to 0.5*SR
    IC_axis = -3;  % identify axis to use to establish IC: 1=ML, 2=AP, 3=VT, 4=Resultant; assign negative value if minimum of axis should be used
    TO_axis = -3;  % identify axis to use to establish TO: 1=ML, 2=AP, 3=VT, 4=Resultant; assign negative value if minimum of axis should be used
    FIND_STEP_SIDE = false;  % step side does not need to be determined
    find_events_function = @findeventsforce;  % identify sensor-specific gait event detection function
    if size(original_data,2) < 4  % if all three axes not available
        original_data = [original_data(:,1), zeros(size(original_data,1), 4-size(original_data,2)), original_data(:,2)];  % add columns of zeros for missing axes
    end
end

%% Prep Visualization
if VISUALIZATION
    prepvisualization(new_SR, visualization_plots);  % set up figure properties and establish legend for visualization
end

%% Resample Data
resampled_data = original_data; %resampledata(original_data, new_SR);  % return data resampled at new sampling rate
resultant = [resampled_data(:,1), sqrt(resampled_data(:,2).^2 + resampled_data(:,3).^2 + resampled_data(:,4).^2)];  % get resultant

%% Analyze data by sliding windows
start_frame = 1;  % start at first frame
end_frame = round(start_frame + new_SR*window_size);  % determine last frame of window based on window_size
IC_all = [];  % initialize as empty
TO_all = [];  % initialize as empty
IC = [];  % initialize as empty
TO = [];  % initialize as empty
LEFT_STEP = [];  % initialize as empty
ge_idx = 1;  % initialize index for first gait event
IN_ALG = zeros(size(resampled_data,1),1);  % initialize all frames as not in algorithm
t_start = zeros(size(resampled_data,1),1);  % initialize start time for all frames as zero
t_elapsed = zeros(size(resampled_data,1),1);  % initialize elapsed time all frames as zero
while start_frame < size(resampled_data,1) && end_frame <= size(resampled_data,1)  % establish loop so that sliding windows increment until the end of the data
    if start_frame < size(resampled_data,1)-min_thresh  % if start_frame is more than thresh frames from end (window large enough to search for gait events)
        END_POINTS = false;  % set as false
        IN_ALG(start_frame:end_frame,1) = 1;  % identify frames that are currently being processed
        t_start(sum([IN_ALG(:,1)==1, t_start(:,1)==0], 2) == 2,1) = tic;  % start the timer for any new frames
        
        [IC_temp, TO_temp] = find_events_function(resampled_data(start_frame:end_frame,2:4), resultant(start_frame:end_frame,2), new_SR);  % find gait events for window of data
        if size(IC_temp,1) > 0  % if at least one IC was found in the window
            IC_all = [IC_all; start_frame + IC_temp - 1];  % add all IC to total collection, with frame number for full data (not window)
        end
        if size(TO_temp,1) > 0  % if at least one TO was found in the window
            TO_all = [TO_all; start_frame + TO_temp - 1];  % add all TO to total collection, with frame number for full data (not window)
        end
        if size(IC_all,1) >= 1
            IC_all(IC_all(:,1)<min_thresh,:) = [];
        end
        if size(TO_all,1) >= 1
            TO_all(TO_all(:,1)<min_thresh,:) = [];
        end
        
        IC = unique(IC_all);  % find unique IC events
        TO = unique(TO_all);  % find unique TO events
        
        
        IC_diff = diff(IC(:,1));  % find number of frames between consecutive IC
        TO_diff = diff(TO(:,1));  % find number of frames between consecutive TO
    else  % window not large enough to search for gait events
        END_POINTS = true;  % set as true
    end

    if size(IC_diff,1) > 1 && size(TO_diff,1) > 1  % if at least 3 IC and at least 3 TO
        while ((ge_idx < size(IC,1) && END_POINTS) || ((ge_idx < size(IC_diff,1)) && (ge_idx <= size(TO_diff,1)) && (IC(ge_idx+2,1) < start_frame)))  % while (either (it is the end points AND (ge_idx is less than or equal to either IC or TO)) OR (ge_idx is not last two events for IC AND ge_idx is not last event for TO AND third IC is before start_frame (ie. all potential events are not in algorithm)))

            [IC_diff, IC_all, IC, INCREMENT_IC] = evalconsecIC(IC_diff, IC_all, IC, TO, ge_idx, min_thresh, max_thresh, [resampled_data(:,2:4), resultant(:,2)], IC_axis);  % evaluate consecutive IC; if too close, remove and don't increment
            [TO_diff, TO_all, TO, INCREMENT_TO] = evalconsecTO(TO_diff, TO_all, TO, IC, ge_idx, min_thresh, new_SR, [resampled_data(:,2:4), resultant(:,2)], TO_axis);  % evaluate consecutive TO; if too close, remove and don't increment
            
            
            if INCREMENT_IC && INCREMENT_TO  % if IC and TO at ge_idx are ok
                
                if FIND_STEP_SIDE  % if step side needs to be determined
                    LEFT_STEP = determinestepside(LEFT_STEP, ge_idx, IC, TO, resampled_data, new_SR);  % determine step side for IC at ge_idx
                end
                
                if ge_idx == 1  % if ge_idx is first event
                    completed_first_frame = 1;  % beginning of completed window is first frame
                else  % not first event
                    completed_first_frame = completed_last_frame+1;  % beginning of completed window is frame after previous completed window
                end
                if ge_idx >= size(IC,1)
                    completed_last_frame = IC(ge_idx,1);
                else
                    completed_last_frame = min([IC(ge_idx+1,1)-1, start_frame-1]);  % end of completed window is the earliest of the frame before the next IC or the frame before the sliding window
                end
                tic;  % start a timer for the subsequent for loop
                for x = completed_first_frame:completed_last_frame  % for each frame in the completed window
                    t_elapsed(x,1) = toc(cast(t_start(x,1), 'uint64')) - toc;  % determine the time elapsed from that frame's start time, subtracting the time in the for loop (ie. all frames within completed window finish at the same time, though the start time may be different)
                end
                IN_ALG(completed_first_frame:completed_last_frame,1) = 0;  % indicate that frame is no longer in the algorithm
                
                if VISUALIZATION
                    plotvisualization(completed_first_frame, completed_last_frame, IC, TO, LEFT_STEP, resampled_data, resultant, new_SR, visualization_plots);  % add completed data points to visualization
                end
                
                ge_idx = ge_idx + 1;  % increment to next gait event
            end
        end
    end
    start_frame = round(start_frame + new_SR*delta_window);  % get new start_frame for sliding window based on delta_window
    if round(start_frame + new_SR*window_size) > size(resampled_data,1)  % if a full window would cause end_frame to go beyond size of data
        end_frame = size(resampled_data,1);  % set end_frame as the last frame
    else  % full window fits within data
        end_frame = round(start_frame + new_SR*window_size);  % set end_frame based on window_size
    end
end
TO(size(IC,1):end,:) = [];  % remove any TO after the last IC

%% Remaining data
if FIND_STEP_SIDE  % if step side needs to be determined
    if size(IC,1) > 0 && ge_idx <= size(IC,1)  % if there was at least one IC and not all IC have a side determination
        for ge_idx2 = ge_idx:size(IC,1)  % for any remaining gait events
            LEFT_STEP = determinestepside(LEFT_STEP, ge_idx2, IC, TO, resampled_data, new_SR);  % determine step side for IC at ge_idx2
        end
    end
end

if sum(t_start(:,1) == 0) > 0  % if at least one frame was not included in algorithm (ie. start time is still 0)
    IN_ALG(t_start(:,1) == 0) = 1;  % mark as in algorithm
    [~, loc] = max(t_start(:,1) == 0);  % find first frame where start time is still 0
    if loc > 1  % if first frame where start time is still 0 is not the first frame within the whole dataset
        t_start(t_start(:,1) == 0) = t_start(loc-1,1);  % set start time to t_start of last frame that was included in the algorithm (essentially including it with last sliding window)
    else
        t_start(t_start(:,1) == 0) = tic;  % start the timer for all frames where start time is still 0 (in this case, all frames)
    end
end

if ge_idx == 1  % if ge_idx is first event (this will only happen if window_size is greater than size of data, thus data not evaluated for gait events)
    completed_first_frame = 1;  % beginning of completed window is first frame
else
    completed_first_frame = completed_last_frame+1;  % beginning of completed window is frame after previous completed window
end
completed_last_frame = size(resampled_data,1);  % end of completed window is the last frame of the dataset
tic;  % start a timer for the subsequent for loop
for x = completed_first_frame:completed_last_frame  % for each frame in the completed window
    t_elapsed(x,1) = toc(cast(t_start(x,1), 'uint64')) - toc;  % determine the time elapsed from that frames start time, subtracting the time in the for loop (ie. all frames within completed window finish at the same time, though the start time may be different)
end
IN_ALG(completed_first_frame:completed_last_frame,1) = 0;  % indicate that frame is no longer in the algorithm
output_rate = 1./t_elapsed;  % calculate output rate for each frame as 1 frame divided by time elapsed (time in algorithm) for that frame

%% Plot
if VISUALIZATION
    plotvisualization(completed_first_frame, completed_last_frame, IC, TO, LEFT_STEP, resampled_data, resultant, new_SR, visualization_plots);  % add completed data points to visualization
end

if FINAL_PLOT
    prepvisualization(new_SR, visualization_plots);  % set up figure properties and establish legend for visualization
    plotvisualization(1, size(resampled_data,1), IC, TO, LEFT_STEP, resampled_data, resultant, new_SR, visualization_plots);  % add completed data points to visualization
end