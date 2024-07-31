%% This function identifies IC and TO events from an accelerometer signal with placement on the back (X = ML (+right), Y = AP (+front), Z = VT (+up))

function [IC, TO] = findeventsback(data, resultant, SR)

IC = [];
TO = [];
all_steps_norm = [];

[~, locs] = findpeaks(data(:,3),SR,'MinPeakDistance',0.25);  % find major peaks in vertical data
pos_peaks = ceil((locs(:,1)+0.0001)*SR);

for i = 2:size(pos_peaks,1)  % for each positive peak starting with the second one
    
    [pk, idx] = findpeaks(-data(pos_peaks(i-1,1):pos_peaks(i,1),2));  % find negative peaks in AP signals within window of two positive peaks in VT signal
    idx = idx + pos_peaks(i-1,1) - 1;  % get index in full reference frame
    [idx, new_order] = sort(idx, 'descend');  % put in descending order so that peak closest to end of window is first (tie breaker will be position closest to end of window)
    pk = pk(new_order);
    
    
    if isempty(idx)
        [~, min_idx] = min(data(pos_peaks(i-1,1):pos_peaks(i,1),2));  % find minimum of AP data
        IC(i,1) = min_idx + pos_peaks(i-1,1) - 1;
        
        [~, max_idx] = max(data(pos_peaks(i-1,1):IC(i,1),2));  % find max of AP data
        max_ap_idx = max_idx(1,1) + pos_peaks(i-1,1) - 1;
        
        slope = diff(data(max_ap_idx:IC(i,1),2))/(1/SR);  % find slope between max of AP and IC
        if size(slope,1) < 3
            TO(i-1,1) = max_ap_idx;
        else
            [~, max_slope] = findpeaks(slope, 'SortStr', 'descend', 'NPeaks', 1);  % find highest peak in slope (closest to zero)
            if isempty(max_slope)  % if no peak in slope
                ignore_end = ceil(size(slope,1)/10);
                if size(slope,1) <= 2*ignore_end
                    [~, b] = max(slope(:,1));  % find max slope ignoring end points
                    max_slope = b(1,1);
                else
                    [~, b] = max(slope(ignore_end+1:size(slope,1)-ignore_end, 1));  % find max slope ignoring end points
                    max_slope = b(1,1)+ignore_end;
                end
            end
            TO(i-1,1) = max_slope + max_ap_idx - 1;
        end
    elseif size(idx,1) == 1
        IC(i,1) = idx(1,1);
        
        [~, max_idx] = max(data(pos_peaks(i-1,1):IC(i,1),2));  % find max of AP data between first positive peak in VT data and IC
        max_ap_idx = max_idx(1,1) + pos_peaks(i-1,1) - 1;  % get index in full reference frame
        
        slope = diff(data(max_ap_idx:IC(i,1),2))/(1/SR);  % find slope between max of AP and IC
        if size(slope,1) < 3  % if not at least three points
            TO(i-1,1) = max_ap_idx;  % set TO for previous IC at location of max of AP data between first positive peak in VT data and IC
        else
            [~, max_slope] = findpeaks(slope, 'SortStr', 'descend', 'NPeaks', 1);  % find highest peak in slope (closest to zero)
            if isempty(max_slope)  % if no peak in slope
                ignore_end = ceil(size(slope,1)/10);  % find number of frames equal to 10% of the size of the window between max of AP and IC; at least one frame
                if size(slope,1) <= 2*ignore_end
                    [~, b] = max(slope(:,1));  % find max slope ignoring end points
                    max_slope = b(1,1);
                else
                    [~, b] = max(slope(ignore_end+1:size(slope,1)-ignore_end, 1));  % find max slope ignoring end points
                    max_slope = b(1,1)+ignore_end;
                end
            end
            TO(i-1,1) = max_slope + max_ap_idx - 1;
        end
    else
        [~, pk_order] = sort(pk, 'descend');  % get order of peaks by height
        [~, idx_order] = sort(idx, 'descend');  % get order of peaks by position
        ref_rank = 1:size(pk,1);  % get a reference ranking based on the number of peaks
        for p = 1:size(pk,1)  % for each peak
            pk_rank(p,1) = ref_rank(pk_order == p);  % find rank based on height
            idx_rank(p,1) = ref_rank(idx_order == p);  % find rank based on position
        end
        mean_rank = mean([pk_rank, idx_rank],2);  % find mean rank based on height and position
        [~,low_rank] = min(mean_rank);  % find lowest rank
        IC(i,1) = idx(low_rank,1);  % IC is lowest ranked peak
        idx(1:low_rank,:) = [];  % remove IC peak and any peaks after from consideration
        mean_rank(1:low_rank,:) = [];  % remove IC peak and any peaks after from consideration
        
        if ~isempty(idx)
            prev_pk = 1;
            while prev_pk <= size(idx,1)
                if abs(idx(prev_pk,1) - IC(i-1,1))/SR < 0.1  % if a previous peak is less than 0.1s after the previous IC
                    idx(prev_pk,:) = [];  % remove from TO contention
                else
                    prev_pk = prev_pk + 1;
                end
            end
        end
        
        if isempty(idx)
            
            [~, max_idx] = max(data(pos_peaks(i-1,1):IC(i,1),2));  % find max of AP data
            max_ap_idx = max_idx(1,1) + pos_peaks(i-1,1) - 1;
            
            slope = diff(data(max_ap_idx:IC(i,1),2))/(1/SR);  % find slope between max of AP and IC
            if size(slope,1) < 3
                TO(i-1,1) = max_ap_idx;
            else
                [~, max_slope] = findpeaks(slope, 'SortStr', 'descend', 'NPeaks', 1);  % find highest peak in slope (closest to zero)
                if isempty(max_slope)  % if no peak in slope
                    ignore_end = ceil(size(slope,1)/10);
                    if size(slope,1) <= 2*ignore_end
                        [~, b] = max(slope(:,1));  % find max slope ignoring end points
                        max_slope = b(1,1);
                    else
                        [~, b] = max(slope(ignore_end+1:size(slope,1)-ignore_end, 1));  % find max slope ignoring end points
                        max_slope = b(1,1)+ignore_end;
                    end
                end
                TO(i-1,1) = max_slope + max_ap_idx - 1;
            end
        else
            TO(i-1,1) = max(idx);  % TO is next closest to end
        end
    end
    
    
    
end
if ~isempty(IC)
    IC(IC==0,:) = [];
end
if ~isempty(TO)
    TO(TO==0,:) = [];
end




