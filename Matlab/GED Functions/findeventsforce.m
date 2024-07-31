%% This function identifies IC and TO events from a force plate signal (Z = VT (+up))


function [IC, TO] = findeventsforce(data, resultant, SR)


fp_threshold_IC = 10;
fp_threshold_TO = 25;


IC = [];
TO = [];


count_to = 0;  % initialize counter
count_IC = 0;  % initialize counter
[~, locs] = findpeaks(data(:,3),SR,'MinPeakDistance',0.25);  % find major peaks in force data
pos_peaks = ceil((locs(:,1)+0.0001)*SR);

for i = 1:size(pos_peaks,1)  % for each peak
    frame = pos_peaks(i,1);
    if i == 1
        lower_bound = 1;
    else
        lower_bound = pos_peaks(i-1,1);
    end
    while frame > lower_bound
        if data(frame,3) > fp_threshold_IC  % while force at frame is greater than threshold
            frame = frame - 1;  % decrease frame
        else
            break;
        end
    end
    if frame > lower_bound  % if frame corresponds to a force less than threshold (did not stop because reached lower bound)
        count_IC = count_IC + 1;  % increment counter
        IC(count_IC,1) = frame;  % record IC
    else
        count_IC = count_IC + 1;  % increment counter
        IC(count_IC,1) = round(mean([lower_bound, pos_peaks(i,1)]));  % record IC as halfway between low bound and peak
    end
    
    frame = pos_peaks(i,1);  % get frame number for peak location
    if i == size(pos_peaks,1)
        upper_bound = size(data,1);
    else
        upper_bound = pos_peaks(i+1,1);
    end
    while frame < upper_bound
        if data(frame,3) > fp_threshold_TO  % while force at frame is greater than threshold
            frame = frame + 1;  % increase frame
        else
            break;
        end
    end
    if frame < upper_bound  % if frame corresponds to a force less than threshold (did not stop because last frame)
        count_to = count_to + 1;  % increment counter
        TO(count_to,1) = frame;  % record TO
    else
        count_to = count_to + 1;  % increment counter
        TO(count_to,1) = round(mean([pos_peaks(i,1), upper_bound]));  % record TO as halfway between peak and high_bound
    end
end






