% Clear all variables, close all figures, and suppress warning for adding new worksheets in Excel
clear all;
close all;
clc;

% Folder path
folder_path = 'D:\\[Personal Folder]\\[Project File]\\data\\Sample_4';
output_path = 'D:\\[Personal Folder]\\[Project File]\\data\\Feature_Extraction\\Sample4';

% Create an empty table for storing flattened features
Feature_flatten_all = {};

% Loop through each Excel file
for k = 1:25
    % Generate the full path for the current file
    file_path = fullfile(folder_path, ['Sweat-sample-4-', num2str(k), '.xlsx']);
    
    % Import data
    data = xlsread(file_path);
    
    % Extract time and sensor data
    time = data(:, 1);
    sensor_data = data(:, 2:end);
    
    % Smooth the sensor data using the Savitzky-Golay filter before calculating the first derivative
    smoothed_data = sgolayfilt(sensor_data, 2, 41); % Savitzky-Golay filter with polynomial order 2 and window size 41

    % Select time ranges
    time_range_max = time >= 95 & time <= 140;
    time_range_min = time >= 190 & time <= 200;
    
    % Calculate the maximum and minimum values within the time range for each sensor column
    max_values = max(smoothed_data(time_range_max, :));
    min_values = min(smoothed_data(time_range_min, :));

    % Initialize minimum derivative and corresponding time for each sensor column
    min_derivative = zeros(size(smoothed_data, 2), 1);
    min_derivative_time = zeros(size(smoothed_data, 2), 1);
    min_derivative_threshold = zeros(size(smoothed_data, 2), 1);
    min_derivative_threshold_time = zeros(size(smoothed_data, 2), 1);
    
    % Plot sensor data and annotate maximum and minimum values
    figure;
    for i = 1:size(smoothed_data, 2)
        subplot(4, 2, i);
        plot(time, smoothed_data(:, i));
        title(['Sensor ', num2str(i)]);
        xlabel('Time');
        ylabel('Value');
        hold on;
        % Mark maximum and minimum values
        max_index = find(smoothed_data(:, i) == max_values(i) & time_range_max, 1);
        min_index = find(smoothed_data(:, i) == min_values(i) & time_range_min, 1);
    
        plot(time(max_index), max_values(i), 'ro', 'MarkerSize', 10);
        plot(time(min_index), min_values(i), 'bo', 'MarkerSize', 10);
        hold off;
    end

    % Compute the first derivative for each sensor column and find the minimum value within 100-150s
    for i = 1:size(smoothed_data, 2)
        % Calculate the first derivative
        derivative = gradient(smoothed_data(:, i), time);
        
        % Limit time range
        time_range = time >= 95 & time <= 150;
        time_range_data = derivative(time_range);
        
        % Find the minimum value and its corresponding time point
        [min_derivative(i), idx_min] = min(time_range_data);
        min_derivative_time(i) = time(find(time_range, 1) + idx_min - 1);
        
        % Calculate the 30% threshold of the minimum derivative value
        min_derivative_threshold(i) = min_derivative(i) * 0.5;
        
        % Find the first point where the derivative falls below the threshold
        idx = find(time_range_data < min_derivative_threshold(i), 1);
        if ~isempty(idx)
            min_derivative_threshold_time(i) = time(find(time_range, 1) + idx - 1);
        end
    end
    
    % Plot sensor data and mark points where the derivative falls below 30% threshold
    response_times = cell(size(smoothed_data, 2), 1);
    response_times_10 = cell(size(smoothed_data, 2), 1);
    response_times_20 = cell(size(smoothed_data, 2), 1);
    response_times_30 = cell(size(smoothed_data, 2), 1);
    response_times_40 = cell(size(smoothed_data, 2), 1);
    response_times_50 = cell(size(smoothed_data, 2), 1);
    response_times_60 = cell(size(smoothed_data, 2), 1);
    response_times_70 = cell(size(smoothed_data, 2), 1);
    response_times_80 = cell(size(smoothed_data, 2), 1);
    response_times_90 = cell(size(smoothed_data, 2), 1);
    response_times_100 = cell(size(smoothed_data, 2), 1);
    delta_response_times_10 = cell(size(smoothed_data, 2), 1);
    delta_response_times_20 = cell(size(smoothed_data, 2), 1);
    delta_response_times_30 = cell(size(smoothed_data, 2), 1);
    delta_response_times_40 = cell(size(smoothed_data, 2), 1);
    delta_response_times_50 = cell(size(smoothed_data, 2), 1);
    delta_response_times_60 = cell(size(smoothed_data, 2), 1);
    delta_response_times_70 = cell(size(smoothed_data, 2), 1);
    delta_response_times_80 = cell(size(smoothed_data, 2), 1);
    delta_response_times_90 = cell(size(smoothed_data, 2), 1);
    delta_response_times_100 = cell(size(smoothed_data, 2), 1);
    start_response_times = cell(size(smoothed_data, 2), 1);
    start_response_values = cell(size(smoothed_data, 2), 1);
    stop_response_times = cell(size(smoothed_data, 2), 1);
    stop_response_values = cell(size(smoothed_data, 2), 1);
    for i = 1:size(smoothed_data, 2)
        subplot(4, 2, i);
        hold on;
        if min_derivative_threshold_time(i) ~= 0
            start_response_time = min_derivative_threshold_time(i);
            start_response_value = smoothed_data(find(time == start_response_time), i);
            plot(start_response_time, start_response_value, 'go', 'MarkerSize', 10);
            
            stop_response_time = time(find(smoothed_data(:, i) == min_values(i), 1));
            stop_response_value = min_values(i);
            plot(stop_response_time, stop_response_value, 'bo', 'MarkerSize', 10);
            
            % Store the start response time and value
            start_response_times{i} = start_response_time;
            start_response_values{i} = start_response_value;
            
            % Store the stop response time and value
            stop_response_times{i} = stop_response_time;
            stop_response_values{i} = stop_response_value;
            
            % Calculate the time from the green to the blue point
            response_time = stop_response_time - start_response_time;
            response_times{i} = response_time;
            
            % Find the time differences and store in the response time table
            response_10 = start_response_value - 0.1 * (start_response_value - stop_response_value);
            response_times_10{i} = time(find(time > start_response_time & smoothed_data(:, i) < response_10, 1));
            
            response_20 = start_response_value - 0.2 * (start_response_value - stop_response_value);
            response_times_20{i} = time(find(time > start_response_time & smoothed_data(:, i) < response_20, 1));
            
            response_30 = start_response_value - 0.3 * (start_response_value - stop_response_value);
            response_times_30{i} = time(find(time > start_response_time & smoothed_data(:, i) < response_30, 1));
            
            response_40 = start_response_value - 0.4 * (start_response_value - stop_response_value);
            response_times_40{i} = time(find(time > start_response_time & smoothed_data(:, i) < response_40, 1));
            
            response_50 = start_response_value - 0.5 * (start_response_value - stop_response_value);
            response_times_50{i} = time(find(time > start_response_time & smoothed_data(:, i) < response_50, 1));

            response_60 = start_response_value - 0.6 * (start_response_value - stop_response_value);
            response_times_60{i} = time(find(time > start_response_time & smoothed_data(:, i) < response_60, 1));
            
            response_70 = start_response_value - 0.7 * (start_response_value - stop_response_value);
            response_times_70{i} = time(find(time > start_response_time & smoothed_data(:, i) < response_70, 1));
            
            response_80 = start_response_value - 0.8 * (start_response_value - stop_response_value);
            response_times_80{i} = time(find(time > start_response_time & smoothed_data(:, i) < response_80, 1));
            
            response_90 = start_response_value - 0.9 * (start_response_value - stop_response_value);
            response_times_90{i} = time(find(time > start_response_time & smoothed_data(:, i) < response_90, 1));
            
            response_100 = start_response_value - 0.99 * (start_response_value - stop_response_value);
            response_times_100{i} = time(find(time > start_response_time & smoothed_data(:, i) < response_100, 1));
            
            % Calculate delta response time
            delta_response_times_10{i} = response_times_10{i} - start_response_time;
            delta_response_times_20{i} = response_times_20{i} - start_response_time;
            delta_response_times_30{i} = response_times_30{i} - start_response_time;
            delta_response_times_40{i} = response_times_40{i} - start_response_time;
            delta_response_times_50{i} = response_times_50{i} - start_response_time;
            delta_response_times_60{i} = response_times_60{i} - start_response_time;
            delta_response_times_70{i} = response_times_70{i} - start_response_time;
            delta_response_times_80{i} = response_times_80{i} - start_response_time;
            delta_response_times_90{i} = response_times_90{i} - start_response_time;
            delta_response_times_100{i} = response_times_100{i} - start_response_time;
            
            % Mark green and blue points on the sensor data plot
            plot(start_response_time, start_response_value, 'go', 'MarkerSize', 10);
            plot(stop_response_time, stop_response_value, 'bo', 'MarkerSize', 10);
            
            % Mark 10%, 30%, 50%, 70%, and 90% response points on the sensor data plot
            plot(response_times_10{i}, response_10, 'rx', 'MarkerSize', 10);
            plot(response_times_20{i}, response_20, 'rx', 'MarkerSize', 10);
            plot(response_times_30{i}, response_30, 'rx', 'MarkerSize', 10);
            plot(response_times_40{i}, response_40, 'rx', 'MarkerSize', 10);
            plot(response_times_50{i}, response_50, 'rx', 'MarkerSize', 10);
            plot(response_times_60{i}, response_60, 'rx', 'MarkerSize', 10);
            plot(response_times_70{i}, response_70, 'rx', 'MarkerSize', 10);
            plot(response_times_80{i}, response_80, 'rx', 'MarkerSize', 10);
            plot(response_times_90{i}, response_90, 'rx', 'MarkerSize', 10);
            plot(response_times_100{i}, response_100, 'rx', 'MarkerSize', 10);
        end
        hold off;
    end
    
    % Create a table called "Feature" to store all feature values
    Feature = table(delta_response_times_10, delta_response_times_20, delta_response_times_30, delta_response_times_40, delta_response_times_50, delta_response_times_60, delta_response_times_70, delta_response_times_80, delta_response_times_90, delta_response_times_100, 'VariableNames', {'Delta_Response_time_10', 'Delta_Response_time_20', 'Delta_Response_time_30', 'Delta_Response_time_40', 'Delta_Response_time_50', 'Delta_Response_time_60', 'Delta_Response_time_70', 'Delta_Response_time_80', 'Delta_Response_time_90', 'Delta_Response_time_100'});
    
    % Create a new table to store all features as cell arrays
    Feature_table = table(Feature.Delta_Response_time_10, Feature.Delta_Response_time_20, Feature.Delta_Response_time_30, Feature.Delta_Response_time_40, Feature.Delta_Response_time_50, Feature.Delta_Response_time_60, Feature.Delta_Response_time_70, Feature.Delta_Response_time_80, Feature.Delta_Response_time_90, Feature.Delta_Response_time_100 , 'VariableNames', {'Delta_Response_time_10', 'Delta_Response_time_20', 'Delta_Response_time_30', 'Delta_Response_time_40', 'Delta_Response_time_50', 'Delta_Response_time_60', 'Delta_Response_time_70', 'Delta_Response_time_80', 'Delta_Response_time_90', 'Delta_Response_time_100'});
    Feature_cell = table2cell(Feature_table);

    % Generate output file name for the current input file
    [~, file_name, ~] = fileparts(file_path);
    excel_output_path = fullfile(output_path, ['Feature_', file_name, '.xlsx']);
   
    % Write the current flattened features to an Excel file
    xlswrite(excel_output_path, Feature_cell);
    
    % Flatten the features into a 1D cell array
    Feature_flatten = reshape(Feature_table{:,:}.', 1, []);

    % Add the current flattened features to Feature_flatten_all
    Feature_flatten_all = [Feature_flatten_all; Feature_flatten];

    saveas(gcf, fullfile(output_path, ['Feature_Extract_Sweat-4-' , file_name, '.png']));
end

% Specify the path to save the Excel file
excel_output_path = 'D:\\[Personal Folder]\\[Project File]\\data\\Feature_Extraction\\Sample4\\Feature_flatten_all_4.xlsx';

% Write Feature_flatten_all to an Excel file
xlswrite(excel_output_path, Feature_flatten_all);
