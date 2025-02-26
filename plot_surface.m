% MATLAB Script: Plotting Volume Triplets and Tip-Base Differences
% Make sure 'data.csv' is in the current folder or adjust the filename path.

%% Read the CSV file
% Since the file does not contain header names, we read it without variable names.
filename = "C:\Users\dogro\Desktop\Emanuele\github\sorolearn\data\exp_2025-02-26_13-56-18\output_exp_2025-02-26_13-56-18.csv";
data = readmatrix(filename)

% The CSV columns are assumed as follows:
% Col1: Timestamp
% Col2: Volume1
% Col3: Volume2
% Col4: Volume3
% Col5: cam_1 image path
% Col6: cam_0 image path
% Col7: Tip X coordinate
% Col8: Tip Y coordinate
% Col9: Tip Z coordinate
% Col10: Base X coordinate
% Col11: Base Y coordinate
% Col12: Base Z coordinate

%% Extract the Data
% Extract the three volume values
volumes = [maintable.VarName2, maintable.VarName3, maintable.VarName4];

% Extract the tip coordinates (columns 7, 8, 9)
tipCoords = [maintable.VarName15, maintable.VarName16, maintable.VarName17];

% Extract the base coordinates (columns 10, 11, 12)
baseCoords = [maintable.VarName18, maintable.VarName19, maintable.VarName20];

% Calculate the difference between tip and base coordinates
diffCoords = tipCoords - baseCoords;

%% 3D Plot of Volume Triplets
figure;
scatter3(volumes(:,1), volumes(:,2), volumes(:,3));
grid on;
xlabel('Volume 1');
ylabel('Volume 2');
zlabel('Volume 3');
title('3D Plot of Volume Triplets');

%% 3D Plot of Tip-Base Coordinate Differences
figure;
plot3(diffCoords(:,1), diffCoords(:,2), diffCoords(:,3), 'r*-', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('X Difference (Tip - Base)');
ylabel('Y Difference (Tip - Base)');
zlabel('Z Difference (Tip - Base)');
title('3D Plot of Tip-Base Coordinate Differences');
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
scatter3(diffCoords(:,1), diffCoords(:,2), diffCoords(:,3), 50, colors, 'filled');
grid on;
xlabel('X Difference (Tip - Base)');
ylabel('Y Difference (Tip - Base)');
zlabel('Z Difference (Tip - Base)');
title('3D Plot of Tip-Base Coordinate Differences (Outputs)');

normVol_1 = (volumes(:,1)-volumes(1,1))/max(volumes(:,1));
normVol_2 = (volumes(:,2)-volumes(1,2))/max(volumes(:,2));
normVol_3 = (volumes(:,3)-volumes(1,3))/max(volumes(:,3));

figure;
hold on;
plot(normVol_1, tipCoords(:,3), 'r', 'LineWidth', 2, 'MarkerSize', 8);
plot(normVol_3, tipCoords(:,2), 'g', 'LineWidth', 2, 'MarkerSize', 8);
plot(normVol_2, tipCoords(:,1), 'b', 'LineWidth', 2, 'MarkerSize', 8);
hold off;
%% Option 1: Per-Instance Mahalanobis Distance Between Each Input and Its Output
% Compute the difference vector for each instance
d = volumes - diffCoords;  % Each row is a 3D difference

% Compute the covariance matrix from all difference vectors
S = cov(d);
invS = inv(S);

% Compute Mahalanobis distance for each instance
N = size(d, 1);
mahalDistances = zeros(N, 1);
for i = 1:N
    mahalDistances(i) = sqrt( d(i,:) * invS * d(i,:)' );
end

disp('Per-instance Mahalanobis distances:');
disp(mahalDistances);

