%% This function evaluates consecutive potential IC and retains the viable IC

function [IC_diff, IC_all, IC, INCREMENT_IC] = evalconsecIC(IC_diff, IC_all, IC, TO, ge_idx, min_thresh, max_thresh, all_signals, IC_axis)

if IC_axis < 0  % if negative
    check_data = -all_signals(:,abs(IC_axis));  % get negative of data at IC_axis
else
    check_data = all_signals(:,abs(IC_axis));  % get data at IC_axis
end


if ge_idx > size(IC,1)
    INCREMENT_IC = false;  % initialize increment as false
else
    INCREMENT_IC = true;  % initialize increment as true
end
while (ge_idx <= size(IC_diff,1)) && (IC_diff(ge_idx,1) < min_thresh)  % while ge_idx is not last event for IC, and if IC at ge_idx is too close to next IC
    if (ge_idx == 1) || ((IC(ge_idx+1,1) - IC(ge_idx-1,1)) < max_thresh)  % if first IC OR both IC (at ge_idx and at ge_idx+1) are within range of IC at ge_idx-1 OR IC at ge_idx+2 is not in the range of IC at ge_idx
        [~, loc] = max([check_data(IC(ge_idx,1),1), check_data(IC(ge_idx+1,1),1)]);  % find signal (IC_axis) at each IC and determine which has greater value
        if loc == 1  % if signal is greatest for first IC
            IC_all(IC_all==IC(ge_idx+1,:),:) = [];  % remove next IC from IC_all
            IC(ge_idx+1,:) = [];  % remove next IC from IC
        else  % if signal is greatest for second IC
            IC_all(IC_all==IC(ge_idx,:),:) = [];  % remove first IC from IC_all
            IC(ge_idx,:) = [];  % remove first IC from IC
        end
        IC_diff = diff(IC(:,1));  % recalculate IC_diff
        INCREMENT_IC = false;  % set increment to false
    elseif ((ge_idx+1 <= size(IC_diff,1)) && ((IC(ge_idx+2,1) - IC(ge_idx,1)) < max_thresh))
        IC_all(IC_all==IC(ge_idx+1,:),:) = [];  % remove next IC from IC_all
        IC(ge_idx+1,:) = [];  % remove next IC from IC
        IC_diff = diff(IC(:,1));  % recalculate IC_diff
        INCREMENT_IC = false;  % set increment to false
    else
        break;
    end
end

while (ge_idx+1 <= size(IC_diff,1)) && (IC_diff(ge_idx+1,1) < min_thresh)  % while ge_idx+1 is not last event for IC, and if IC at ge_idx+1 is too close to next IC
    if (IC(ge_idx+2,1) - IC(ge_idx,1)) < max_thresh  % if both IC (at ge_idx+1 and at ge_idx+2) are within range of IC at ge_idx
        [~, loc] = max([check_data(IC(ge_idx+1,1),1), check_data(IC(ge_idx+2,1),1)]);  % find signal (IC_axis) at each IC and determine which has greater value
        if loc == 1  % if signal is greatest for first IC
            IC_all(IC_all==IC(ge_idx+2,:),:) = [];  % remove next IC from IC_all
            IC(ge_idx+2,:) = [];  % remove next IC from IC
        else  % if signal is greatest for second IC
            IC_all(IC_all==IC(ge_idx+1,:),:) = [];  % remove first IC from IC_all
            IC(ge_idx+1,:) = [];  % remove first IC from IC
        end
        IC_diff = diff(IC(:,1));  % recalculate IC_diff
        INCREMENT_IC = false;  % set increment to false
    else
        break;
    end
end



while (ge_idx <= size(IC_diff,1)) && (ge_idx > size(TO,1))  % if next IC with no corresponding TO
    IC_all(IC_all==IC(ge_idx+1,:),:) = [];  % remove next IC from IC_all
    IC(ge_idx+1,:) = [];  % remove next IC from IC
    IC_diff = diff(IC(:,1));  % recalculate IC_diff
end