% Script to test read in landmarks in .c3d format from MVN data 

clear
clc

% import org.opensim.modeling.*
file_name = 'AT_BL_MVN.c3d';

% c3dFA = C3DFileAdapter();
% data = c3dFA.read(file_name);
% markers = c3dFA.getMarkersTable(data);

[markers,VideoFrameRate,AnalogSignals,AnalogFrameRate,Event,ParameterGroup,CameraInfo,ResidualError] = readc3d(file_name);

n_markers = size(markers, 2);
n_frames = size(markers, 1);

markers_mat = NaN(n_frames, n_markers);

for i_marker = 1:n_markers
   
    markers_mat(:, (i_marker-1)*3 + 1 : i_marker*3) = reshape(markers(:,i_marker,:),[n_frames,3]);
   
end

figure(1)
plot(markers_mat(:, 1:3:end))
title('X displacement')
xlabel('Frame (#)')
ylabel('Position (mm)')

figure(2)
plot(markers_mat(:, 2:3:end))
title('Y displacement')
xlabel('Frame (#)')
ylabel('Position (mm)')

figure(3)
plot(markers_mat(:, 3:3:end))
title('Z displacement')
xlabel('Frame (#)')
ylabel('Position (mm)')