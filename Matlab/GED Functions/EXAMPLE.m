%% This script sets the variables to be passed into the gait event detection algorithm

%% Set parameters
% new sampling rate in Hz
new_SR = 60;
% window size in s
window_size = 2;
% change in window start time in s
delta_window = 0.02;
% boolean to see real-time display of data processing
VISUALIZATION = true;
% boolean to see final plot
FINAL_PLOT = true;

%% Run Algorithms
% load preprocessed data (filtered, trimmed, aligned)
load('data.mat', 'data');

[new_back, new_back_res, output_rate_back, IC_back, TO_back, LEFT_STEP_back] = getevents('back', data.back, new_SR, window_size, delta_window, VISUALIZATION, FINAL_PLOT);
[new_foot, new_foot_res, output_rate_foot, IC_foot, TO_foot, ~] = getevents('Rfoot', rfoot, new_SR, window_size, delta_window, VISUALIZATION, FINAL_PLOT);
[new_force, new_force_res, output_rate_force, IC_force, TO_force, ~] = getevents('force', data.force, new_SR, window_size, delta_window, VISUALIZATION, FINAL_PLOT);