%% This function prepares the figure for visualization

function prepvisualization(new_SR, visualization_plots)

figure;  % create new figure
hold on;  % add plots to figure
count = 0;
if sum(visualization_plots == 1) > 0  % if plot 1 is chosen
    h1 = plot(-4*new_SR, 0, 'g', 'Color', [0, 128/255, 0], 'LineWidth', 1);  % plot ML
    count = count + 1;
    leg{1,count} = 'ML';
end
if sum(visualization_plots == 2) > 0  % if plot 2 is chosen
    h2 = plot(-4*new_SR, 0, 'k', 'LineWidth', 1);  % plot AP
    count = count + 1;
    leg{1,count} = 'AP';
end
if sum(visualization_plots == 3) > 0  % if plot 3 is chosen
    h3 = plot(-4*new_SR, 0, 'b', 'LineWidth', 1);  % plot VT
    count = count + 1;
    leg{1,count} = 'VT';
end
if sum(visualization_plots == 4) > 0  % if plot 4 is chosen
    h4 = plot(-4*new_SR, 0, 'r', 'LineWidth', 1);  % plot Resultant
    count = count + 1;
    leg{1,count} = 'Resultant';
end
if sum(visualization_plots == 5) > 0  % if plot 5 is chosen
    h5 = fill([-4*new_SR, -4*new_SR, -4*new_SR, -4*new_SR, -4*new_SR], [0, 0, 0, 0, 0], [0.9, 0.9, 0.9]);  % plot Left Step
    count = count + 1;
    leg{1,count} = 'Left Step';
end
if sum(visualization_plots == 6) > 0  % if plot 6 is chosen
    h6 = plot(-4*new_SR, 0, 'm', 'LineWidth', 2);  % plot IC
    count = count + 1;
    leg{1,count} = 'IC';
end
if sum(visualization_plots == 7) > 0  % if plot 7 is chosen
    h7 = plot(-4*new_SR, 0, 'c', 'LineWidth', 2);  % plot TO
    count = count + 1;
    leg{1,count} = 'TO';
end
legend(leg, 'Location', 'NorthEast', 'AutoUpdate','off');  % Add legend, but don't update legend in future plots
xlabel('Frames');  % label x-axis
ylabel('Acceleration (g)');  % label y-axis
drawnow;  % display plot in real time