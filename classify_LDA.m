% Function that classifies the test input based on the posterior
% probabilities of different classes and predicts the class with the
% highest posterior value.
% Input: test input Utest, coefficients beta0 and beta of the linear
% decision boundaries
% Output: predicted classes for each sample and the posteriors for all the
% classes

function [predicted_class, posterior] = classify_LDA(Utest, beta0, beta)

n_test = size(Utest, 1);

% Compute discriminant functions zi(t)
z = Utest * beta + repmat(beta0', n_test, 1);

% Convert to probabilities
posterior = exp(z);
posterior = posterior ./ sum(posterior, 2);

% Finding the predicted class posterior of which corresponds to the maximum
% probability among all the classes
[~, predicted_class] = max(posterior, [], 2);

end