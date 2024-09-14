%% Clear environment variables
warning off              % Turn off warning messages
close all                % Close all open figures
clear                    % Clear workspace variables
clc                      % Clear command window

%% Import data
res = xlsread('D:\\[Personal Folder]\\[Project File]\\data.xlsx');
resb = res(1:600, :);    % Use the first 600 rows of data

%% Split the dataset into training and testing sets
temp = randperm(600);    % Randomly permute indices from 1 to 600

P_train = resb(temp(1:480), 1:80)';  % First 480 samples for training, features (1:80)
T_train = resb(temp(1:480), 81)';    % First 480 samples for training, labels (81st column)
M = size(P_train, 2);                % Number of training samples

P_test = res(temp(481:600), 1:80)';  % Remaining 120 samples for testing, features (1:80)
T_test = res(temp(481:600), 81)';    % Remaining 120 samples for testing, labels (81st column)
N = size(P_test, 2);                 % Number of testing samples

%% Data normalization
[P_train, ps_input] = mapminmax(P_train, 0,1);  % Normalize training data to [0,1] range
P_test = mapminmax('apply', P_test, ps_input);  % Apply the same normalization to test data

t_train = categorical(T_train)';  % Convert training labels to categorical
t_test = categorical(T_test)';    % Convert testing labels to categorical

%% Reshape data
p_train = double(reshape(P_train, 8, 10, 1, M));  % Reshape training data to 4D array [8, 10, 1, M]
p_test = double(reshape(P_test, 8, 10, 1, N));    % Reshape testing data to 4D array [8, 10, 1, N]

%% Define the network structure
layers = [
    imageInputLayer([8, 10, 1])            % Input layer: image size 8x10 with 1 channel
    convolution2dLayer([2, 2], 64)         % 2x2 convolution with 64 filters
    batchNormalizationLayer                % Batch normalization layer
    reluLayer                              % ReLU activation function
    maxPooling2dLayer([2, 2], "Stride", 1) % Max pooling with 2x2 filter and stride 1
    convolution2dLayer([2, 2], 128)        % 2x2 convolution with 128 filters
    batchNormalizationLayer                % Batch normalization layer
    reluLayer                              % ReLU activation function
    maxPooling2dLayer([2, 2], 'Stride', 1) % Max pooling with 2x2 filter and stride 1
    fullyConnectedLayer(6)                 % Fully connected layer with 6 output classes
    softmaxLayer                           % Softmax activation for classification
    classificationLayer];                  % Classification output layer

%% Training options
options = trainingOptions('adam', ...      % Use Adam optimizer
    'MaxEpochs', 500, ...                  % Maximum number of epochs
    'MiniBatchSize', 128, ...              % Mini-batch size
    'InitialLearnRate', 0.001, ...         % Initial learning rate
    'L2Regularization', 0.002, ...         % L2 regularization parameter
    'LearnRateSchedule', 'piecewise', ...  % Piecewise learning rate schedule
    'LearnRateDropFactor', 0.1, ...        % Drop learning rate by a factor of 0.1
    'LearnRateDropPeriod', 450, ...        % Drop learning rate after 450 epochs
    'Shuffle', 'every-epoch', ...          % Shuffle data at every epoch
    'ValidationData', {p_test, t_test}, ... % Use test set as validation data
    'ValidationFrequency', 1, ...          % Validate every iteration
    'ValidationPatience', Inf, ...         % No early stopping based on validation results
    'Plots', 'training-progress', ...      % Show training progress
    'Verbose', true, 'VerboseFrequency', 1);

%% Train the network
net = trainNetwork(p_train, t_train, layers, options);

%% Model prediction
t_sim1 = classify(net, p_train);  % Classify training set
t_sim2 = classify(net, p_test);   % Classify test set

%% Performance evaluation
T_sim1 = double(t_sim1)';  % Convert predicted labels for training set to double
T_sim2 = double(t_sim2)';  % Convert predicted labels for test set to double

error1 = sum((T_sim1 == T_train)) / M * 100;  % Calculate accuracy for training set
error2 = sum((T_sim2 == T_test)) / N * 100;   % Calculate accuracy for test set

%% Plot - Training set results
figure
plot(1:M, T_train, 'r-*', 1:M, T_sim1, 'b-o', 'LineWidth', 1)  % Plot true vs predicted values for training set
legend('True values', 'Predicted values')
xlabel('Prediction samples')
ylabel('Prediction results')
title({'Training set prediction results comparison'; ['Accuracy = ' num2str(error1) '%']})
xlim([1, M])
grid

%% Plot - Test set results
figure
plot(1:N, T_test, 'r-*', 1:N, T_sim2, 'b-o', 'LineWidth', 1)  % Plot true vs predicted values for test set
legend('True values', 'Predicted values')
xlabel('Prediction samples')
ylabel('Prediction results')
title({'Test set prediction results comparison'; ['Accuracy = ' num2str(error2) '%']})
xlim([1, N])
grid

%% Confusion matrix - Training set
figure
cm_train = confusionchart(T_train, T_sim1);  % Create confusion matrix for training set
cm_train.Title = 'Confusion Matrix - Training Set';
cm_train.ColumnSummary = 'column-normalized';  % Normalize by columns
cm_train.RowSummary = 'row-normalized';        % Normalize by rows

%% Confusion matrix - Test set
figure
cm_test = confusionchart(T_test, T_sim2);  % Create confusion matrix for test set
cm_test.Title = 'Confusion Matrix - Test Set';
cm_test.ColumnSummary = 'column-normalized';  % Normalize by columns
cm_test.RowSummary = 'row-normalized';        % Normalize by rows
