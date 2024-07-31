%% This function determines if a step was taken on the left or right side, based on the back accelerometer signal (X = ML (+right), Y = AP (+front), Z = VT (+up))


function LEFT_STEP = determinestepside(LEFT_STEP, ge_idx, IC, TO, resampled_data, new_SR)

if ge_idx < size(IC,1) && ge_idx <= size(TO,1)  % if ge_idx is not last event for IC
    pos_pk = [];  % initialize as empty
    neg_pk = [];  % initialize as empty
    if (TO(ge_idx) <= size(resampled_data,1)) && ((TO(ge_idx) - IC(ge_idx)) >= 3)  % if TO is not outside of data
        SIDE_INDENTIFIED = false;  % initialize as false
        [pos_pk, pos_loc] = findpeaks(resampled_data(IC(ge_idx):TO(ge_idx),2), 'SortStr', 'descend', 'NPeaks', 1);  % find magnitude and location of largest positive peak within step
        [neg_pk, neg_loc] = findpeaks(-resampled_data(IC(ge_idx):TO(ge_idx)-1,2), 'SortStr', 'descend', 'NPeaks', 1);  % find magnitude and location of largest negative peak within step
        while ~SIDE_INDENTIFIED  % while side has not been identified
            if isempty(pos_pk)  % if no positive peak
                LEFT_STEP(ge_idx,1) = 1;  % set as left step
                SIDE_INDENTIFIED = true;  % set as true
            elseif isempty(neg_pk)  % if no negative peak
                LEFT_STEP(ge_idx,1) = 0;  % set as right step
                SIDE_INDENTIFIED = true;  % set as true
            elseif neg_loc > pos_loc  % if negative peak is later than positive peak
                if abs(neg_loc - (TO(ge_idx)-IC(ge_idx)+1)) < abs(neg_loc-pos_loc)  % if negative peak is closer to TO than to positive peak (correct)
                    LEFT_STEP(ge_idx,1) = 0;  % set as right step
                    SIDE_INDENTIFIED = true;  % set as true
                else  % positive peak is likely the initial peak right before the major negative peak (incorrect)
                    shift_loc = pos_loc;  % retain location of incorrect positive peak
                    if shift_loc < 0.015*new_SR
                        [pos_pk, pos_loc] = findpeaks(resampled_data(IC(ge_idx)+shift_loc:TO(ge_idx),2), 'SortStr', 'descend', 'NPeaks', 1);  % find magnitude and location of largest positive peak after incorrect positive peak
                        if ~isempty(pos_pk)  % if a new positive peak was found
                            pos_loc = pos_loc + shift_loc;  % get location of new positive peak in same frame reference as negative peak; while loop will be repeated
                        end
                    else
                        LEFT_STEP(ge_idx,1) = 0;  % set as right step
                        SIDE_INDENTIFIED = true;  % set as true
                    end
                end
            else  % positive peak is later than negative peak
                if abs(pos_loc - (TO(ge_idx)-IC(ge_idx)+1)) < abs(pos_loc-neg_loc)  % if positive peak is closer to TO than to negative peak (correct)
                    LEFT_STEP(ge_idx,1) = 1;  % set as left step
                    SIDE_INDENTIFIED = true;  % set as true
                else  % negative peak is likely the initial peak right before the major positive peak (incorrect)
                    shift_loc = neg_loc;  % retain location of incorrect negative peak
                    if shift_loc < 0.015*new_SR
                        [neg_pk, neg_loc] = findpeaks(-resampled_data(IC(ge_idx)+shift_loc:TO(ge_idx),2), 'SortStr', 'descend', 'NPeaks', 1);  % find magnitude and location of largest negative peak after incorrect negative peak
                        if ~isempty(neg_pk)  % if a new negative peak was found
                            neg_loc = neg_loc + shift_loc;  % get location of new negative peak in same frame reference as positive peak; while loop will be repeated
                        end
                    else
                        LEFT_STEP(ge_idx,1) = 1;  % set as left step
                        SIDE_INDENTIFIED = true;  % set as true
                    end
                end
            end
        end
    end
elseif ge_idx == size(IC,1) % ge_idx is last event for IC
    if LEFT_STEP(ge_idx-1,1) == 0  % if previous step was right
        LEFT_STEP(ge_idx, 1) = 1;  % set as left step
    else
        LEFT_STEP(ge_idx, 1) = 0;  % set as right step
    end
end