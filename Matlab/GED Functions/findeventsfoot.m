%% This function identifies IC and TO events from an accelerometer signal with placement on the foot (X = ML (+left), Y = AP (+back), Z = VT (+up))

function [IC, TO] = findeventsfoot(data, resultant, SR)

% Initialize variables
IC = [];
TO = [];
pk_loc = [];

[~, locs] = findpeaks(resultant(:,1),SR,'MinPeakDistance',0.5);  % find major peaks in resultant
IC = ceil((locs(:,1)+0.0001)*SR);  % get frames

for i = 1:size(IC,1)
    if i < size(IC,1)
        end_frame = round((IC(i+1,1)+IC(i,1))/2);  % approximate contralateral step occurs at the midpoint between consecutive steps on the same foot, thus TO cannot be after midpoint
    else
        end_frame = size(resultant,1);  % TO can be up to end of data window
    end
    start_frame = round(IC(i,1) + 0.1*SR);  % TO is at least 0.1s after IC
    
    if (start_frame <= size(resultant,1)) && (end_frame <= size(resultant,1)) && (start_frame < end_frame) && (start_frame > 0) && (end_frame > 0)
        
        section = data(start_frame:end_frame,3);  % define the search section within step
        if size(section,1) >= 3
            [~, pk_loc] = findpeaks(section, 'SortStr', 'descend', 'NPeaks',1);  % find greatest positive peak for vertical axis in section
            if isempty(pk_loc)  % if no peak in vertical axis within section
                [~, pk_loc] = max(section);  % find greatest value for vertical axis in section
            end
            if isempty(pk_loc)
                pk_loc = round(0.8 * (end_frame-start_frame));  % find point corresponding to 0.8 of the section
            end
            TO(i,1) = pk_loc + start_frame - 1;  % record toe-off
        else
            IC(i,:) = [];  % not enough room to search for TO, remove IC
        end
    elseif (start_frame < end_frame)
        pk_loc = round(0.8 * (end_frame-start_frame));  % find point corresponding to 0.8 of the section        
        TO(i,1) = pk_loc + start_frame - 1;  % record toe-off
    else
        IC(i,:) = [];  % not enough room to search for TO, remove IC
    end
end




