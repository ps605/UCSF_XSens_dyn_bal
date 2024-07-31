%% This function evaluates consecutive potential TO and retains the viable TO

function [TO_diff, TO_all, TO, INCREMENT_TO] = evalconsecTO(TO_diff, TO_all, TO, IC, ge_idx, min_thresh, new_SR, all_signals, TO_axis)

if TO_axis < 0  % if negative
    check_data = -all_signals(:,abs(TO_axis));  % get negative of data at TO_axis
else
    check_data = all_signals(:,abs(TO_axis));  % get data at TO_axis
end

INCREMENT_TO = true;  % initialize increment as true

if (ge_idx < size(IC,1)) && (ge_idx <= size(TO_diff,1)) && ((TO(ge_idx,1) < IC(ge_idx+1,1)) && (TO(ge_idx+1,1) < IC(ge_idx+1,1)))  % if (ge_idx is not last event for IC AND if ge_idx is not last event for TO) AND (both TO are before next IC)
    if sum(IC(1:ge_idx+1,1) == TO(ge_idx,1)) > 0  % if first TO matches any verified IC
        TO_all(TO_all==TO(ge_idx,:),:) = [];  % remove first TO from TO_all
        TO(ge_idx,:) = [];  % remove first TO from TO
        TO_diff = diff(TO(:,1));  % recalculate TO_diff
        INCREMENT_TO = false;  % set increment to false
    elseif TO(ge_idx,1) < IC(ge_idx,1)  % if first TO is less than corresponding IC
        TO_all(TO_all==TO(ge_idx,:),:) = [];  % remove first TO from TO_all
        TO(ge_idx,:) = [];  % remove first TO from TO
        TO_diff = diff(TO(:,1));  % recalculate TO_diff
        INCREMENT_TO = false;  % set increment to false
    else
        if (TO_diff(ge_idx,1) < min_thresh) || ((TO(ge_idx,1) < IC(ge_idx+1,1)) && (TO(ge_idx+1,1) < IC(ge_idx+1,1)))  % if TO at ge_idx is too close to next TO OR (both TO are before next IC)
            if (TO(ge_idx,1) < IC(ge_idx+1,1)) && (TO(ge_idx+1,1) > IC(ge_idx+1,1))  % if first TO is before next IC AND next TO is after next IC
                INCREMENT_TO = true;  % initialize increment as true (ensure there is a TO for each IC)
            else  % both TO are after IC and before next IC
                if ((TO(ge_idx,1) - IC(ge_idx,1)) / new_SR < 0.25) && ((TO(ge_idx+1,1) - IC(ge_idx,1)) / new_SR <= 0.5)  % if first TO is less than 0.25 s after IC AND next TO is less than or equal to 0.5 s after IC
                    TO_all(TO_all==TO(ge_idx,:),:) = [];  % remove first TO from TO_all
                    TO(ge_idx,:) = [];  % remove first TO from TO
                elseif ((TO(ge_idx,1) - IC(ge_idx,1)) / new_SR >= 0.25) && ((TO(ge_idx+1,1) - IC(ge_idx,1)) / new_SR > 0.5)  % if first TO is greater than or equal to 0.25 s after IC AND next TO is greater than 0.5 s after IC
                    TO_all(TO_all==TO(ge_idx+1,:),:) = [];  % remove next TO from TO_all
                    TO(ge_idx+1,:) = [];  % remove next TO from TO
                else
                    [~, loc] = max([check_data(TO(ge_idx,1),1), check_data(TO(ge_idx+1,1),1)]);  % find signal (TO_axis) at each TO and determine which has greater value
                    %                 [~, loc] = max(sum([[TO(ge_idx,1) - IC(ge_idx,1), TO(ge_idx+1,1) - IC(ge_idx,1)] / new_SR > 0.25; [TO(ge_idx,1) - IC(ge_idx,1), TO(ge_idx+1,1) - IC(ge_idx,1)] / new_SR < 0.5], 1));  % determine if each TO satisfies the conditions of >0.25s after IC and <0.5s after IC; identify TO with most criteria satisfied, with tie going to first TO
                    if loc == 1  % if first TO identified
                        TO_all(TO_all==TO(ge_idx+1,:),:) = [];  % remove next TO from TO_all
                        TO(ge_idx+1,:) = [];  % remove next TO from TO
                    else  % if second TO identified
                        TO_all(TO_all==TO(ge_idx,:),:) = [];  % remove first TO from TO_all
                        TO(ge_idx,:) = [];  % remove first TO from TO
                    end
                end
                TO_diff = diff(TO(:,1));  % recalculate TO_diff
                INCREMENT_TO = false;  % set increment to false
            end
        end
    end
end