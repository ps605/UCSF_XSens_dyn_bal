%% This function plots the visualization

function plotvisualization(completed_first_frame, completed_last_frame, IC, TO, LEFT_STEP, resampled_data, resultant, new_SR, visualization_plots)

range_data = max(max([resampled_data(:,2:4), resultant(:,2)])) - min(min([resampled_data(:,2:4), resultant(:,2)]));
ymin = min(min([resampled_data(:,2:4), resultant(:,2)])) - 0.1*range_data;
ymax = max(max([resampled_data(:,2:4), resultant(:,2)])) + 0.1*range_data;

for i = completed_first_frame:completed_last_frame  % for each frame in completed window
    if sum(IC(:,1) == i) > 0  % if frame is a IC
        if sum(visualization_plots == 5) > 0  % if plot 5 is chosen
            if LEFT_STEP(IC(:,1) == i,1) == 1  % if IC corresponds to a left step
                [~, IC_idx] = max(IC(:,1) == i);  % find index of IC for frame i
                if IC_idx == size(IC,1)  % if last IC
                    fill([i, i, completed_last_frame, completed_last_frame, i], [ymin, ymax, ymax, ymin, ymin], [0.9, 0.9, 0.9], 'EdgeColor', 'none');  % draw a box for left step from IC to last frame of completed window
                else  % not last IC
                    fill([i, i, IC(IC_idx+1,1)-1, IC(IC_idx+1,1)-1, i], [ymin, ymax, ymax, ymin, ymin], [0.9, 0.9, 0.9], 'EdgeColor', 'none');  % draw a box for left step from IC to frame before next IC
                end
            end
        end
        if sum(visualization_plots == 6) > 0  % if plot 6 is chosen
            plot(i*ones(2,1), [ymin ymax], 'm', 'LineWidth', 2);  % draw a line to indicate IC
        end
    elseif sum(TO(:,1) == i) > 0  % if frame is a TO
        if sum(visualization_plots == 7) > 0  % if plot 7 is chosen
            plot(i*ones(2,1), [ymin ymax], 'c', 'LineWidth', 2);  % draw a line to indicate TO
        end
    end
end

if completed_first_frame == 1  % if completed window starts with first frame of dataset, plot from completed_first_frame to completed_last_frame
    if sum(visualization_plots == 1) > 0  % if plot 1 is chosen
        plot(completed_first_frame:completed_last_frame, resampled_data(completed_first_frame:completed_last_frame,2),'g', 'Color', [0, 128/255, 0], 'LineWidth', 1);  % plot ML data
    end
    if sum(visualization_plots == 2) > 0  % if plot 2 is chosen
        plot(completed_first_frame:completed_last_frame, resampled_data(completed_first_frame:completed_last_frame,3), 'k', 'LineWidth', 1);  % plot AP data
    end
    if sum(visualization_plots == 3) > 0  % if plot 3 is chosen
        plot(completed_first_frame:completed_last_frame, resampled_data(completed_first_frame:completed_last_frame,4),'b', 'LineWidth', 1);  % plot VT data
    end
    if sum(visualization_plots == 4) > 0  % if plot 4 is chosen
        plot(completed_first_frame:completed_last_frame, resultant(completed_first_frame:completed_last_frame,2),'r', 'LineWidth', 1);  % plot Resultant data
    end
else  % connect to previous completed window by plotting from frame before completed_first_frame to completed_last_frame
    if sum(visualization_plots == 1) > 0  % if plot 1 is chosen
        plot(completed_first_frame-1:completed_last_frame, resampled_data(completed_first_frame-1:completed_last_frame,2),'g', 'Color', [0, 128/255, 0], 'LineWidth', 1);  % plot ML data
    end
    if sum(visualization_plots == 2) > 0  % if plot 2 is chosen
        plot(completed_first_frame-1:completed_last_frame, resampled_data(completed_first_frame-1:completed_last_frame,3), 'k', 'LineWidth', 1);  % plot AP data
    end
    if sum(visualization_plots == 3) > 0  % if plot 3 is chosen
        plot(completed_first_frame-1:completed_last_frame, resampled_data(completed_first_frame-1:completed_last_frame,4),'b', 'LineWidth', 1);  % plot VT data
    end
    if sum(visualization_plots == 4) > 0  % if plot 4 is chosen
        plot(completed_first_frame-1:completed_last_frame, resultant(completed_first_frame-1:completed_last_frame,2),'r', 'LineWidth', 1);  % plot Resultant data
    end
end
axis([completed_first_frame-2*new_SR completed_last_frame+2*new_SR ymin ymax]);  % set x-axis range: 2 seconds before completed_first_frame to 2 seconds after completed_last_frame
drawnow;  % display plot in real time