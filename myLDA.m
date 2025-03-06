% Function that computes the constant and linear coefficients of the
% LDA discriminant functions based on the data's means, covariances and prior probabilities
% Input: train data U and Y
% Output: coefficients beta0 and beta

function [beta0, beta] = myLDA(Utrain,Ytrain)

unique_labels = unique(Ytrain); % Identify the unique classes in the set
n_classes = size(unique_labels, 1); % Number of classes
n_features = size(Utrain, 2); % Number of features
n_samples = size(Utrain, 1); % Number of samples in the input

% Calculating prior probabilities of the classes
prior = zeros(n_classes, 1);
for i = 1:n_classes
    prior(i) = sum(Ytrain == i) / size(Ytrain,1);
end
    
% Calculate class means
mu = zeros(n_classes, n_features);
for i = 1:n_classes
    mu(i,:) = mean(Utrain(Ytrain==i, :));
end

% Calculate the shared covariance matrix
sigma = zeros(n_features);
for i = 1:n_classes
    class_data = Utrain(Ytrain==i, :);
    centered_data = class_data - mu(i,:);
    sigma = sigma + (centered_data' * centered_data);
end
sigma = sigma / (n_samples - n_classes);  % Pooled estimate
sigma_inv = inv(sigma);
    
% Compute discriminant function parameters
beta = zeros(n_features, n_classes);    
beta0 = zeros(n_classes, 1);           
    
for i = 1:n_classes
    beta(:,i) = sigma_inv * mu(i,:)';
    beta0(i) = -0.5 * (mu(i,:) * sigma_inv * mu(i,:)') + log(prior(i));
end

end