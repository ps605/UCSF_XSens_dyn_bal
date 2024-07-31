%% This function resamples the data

function resampled_data = resampledata(original_data, new_SR)

resampled_data(:,1) = original_data(1,1):1/new_SR*1000:original_data(end,1);  % get new timestamp at new sampling rate

for i = 2:size(original_data,2)
    resampled_data(:,i) = interp1(original_data(:,1),original_data(:,i),resampled_data(:,1));  % resample data to new sampling rate
end

SR = round(mean(1./diff((resampled_data(:,1)-resampled_data(1,1))/1000)));  % get sampling rate
if new_SR ~= SR  % if sampling rate of resampled data is not equal to the requested new sampling rate
    warning('Incorrect data resampling');  % throw a warning
end