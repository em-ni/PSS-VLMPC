% MATLAB Script: Plotting Volume Triplets and Tip-Base Differences
% Make sure 'data.csv' is in the current folder or adjust the filename path.
clear
clc
close all
%% Read the CSV file
% Since the file does not contain header names, we read it without variable names.
exp_name = "../data/04-16_and_04_17/";
data = load(exp_name + "maintable.mat");
data = data.dataset;

% The CSV columns are assumed as follows:
% Col1: Timestamp
% Col2: Volume1
% Col3: Volume2
% Col4: Volume3
% Col5: img_left image path
% Col6: img_right image path
% Col7: Tip X coordinate
% Col8: Tip Y coordinate    
% Col9: Tip Z coordinate
% Col10: Base X coordinate
% Col11: Base Y coordinate
% Col12: Base Z coordinate

%% Extract the Data
% Extract the three volume values
volumes = [data.volume_1, data.volume_2, data.volume_3];
pressures = [data.pressure_1, data.pressure_2, data.pressure_3];
% pressures(:,1) = pressures(:,1)-pressures(1,1);
% pressures(:,2) = pressures(:,2)-pressures(1,2);
% pressures(:,3) = pressures(:,3)-pressures(1,3);

figure();
hold on;
plot(pressures(:,1));
plot(pressures(:,2));
plot(pressures(:,3));
hold off;

% 
% Extract the tip coordinates 
tipCoords = [data.tip_x, data.tip_y, data.tip_z];

% Extract the base coordinates 
baseCoords = [data.base_x, data.base_y, data.base_z];

% Calculate the difference between tip and base coordinates
diffCoords = tipCoords - baseCoords;

N = size(volumes, 1);
colors = zeros(N, 3);
colors(:,1)=(volumes(:,1)-min(volumes(:,1)))/(max(volumes(:,1))-min(volumes(:,1)));
colors(:,2)=(volumes(:,2)-min(volumes(:,2)))/(max(volumes(:,2))-min(volumes(:,2)));
colors(:,3)=(volumes(:,3)-min(volumes(:,3)))/(max(volumes(:,3))-min(volumes(:,3)));

%% 3D Plot of Volume Triplets (Inputs) with Matching Colors
figure;
scatter3(volumes(:,1), volumes(:,2), volumes(:,3), 50, colors, 'filled');
grid on;
xlabel('Volume 1');
ylabel('Volume 2');
zlabel('Volume 3');
title('3D Plot of Volume Triplets (Inputs)');

%% 3D Plot of Tip-Base Coordinate Differences (Outputs) with Matching Colors
figure;
hold on
axis equal
scatter3(diffCoords(:,1), diffCoords(:,2), diffCoords(:,3), 50, colors, 'filled');
grid on;
xlabel('X Difference (Tip - Base)');
ylabel('Y Difference (Tip - Base)');
zlabel('Z Difference (Tip - Base)');
title('3D Plot of Tip-Base Coordinate Differences (Outputs)');
hold off;

%% 3D Plot of Pressure Triplets (Inputs) with Matching Colors
figure;
scatter3(pressures(:,1), pressures(:,2), pressures(:,3), 50, colors, 'filled');
grid on;
xlabel('Volume 1');
ylabel('Volume 2');
zlabel('Volume 3');
title('3D Plot of Volume Triplets (Inputs)');

[i,val] = find(diffCoords(:,3) == 76.120319166383920)

normVol_1 = (volumes(:,1)-volumes(1,1))/max(volumes(:,1));
normVol_2 = (volumes(:,2)-volumes(1,2))/max(volumes(:,2));
normVol_3 = (volumes(:,3)-volumes(1,3))/max(volumes(:,3));

figure;
hold on;
plot(normVol_1, tipCoords(:,3), 'r', 'LineWidth', 2, 'MarkerSize', 8);
plot(normVol_3, tipCoords(:,2), 'g', 'LineWidth', 2, 'MarkerSize', 8);
plot(normVol_2, tipCoords(:,1), 'b', 'LineWidth', 2, 'MarkerSize', 8);
hold off;


