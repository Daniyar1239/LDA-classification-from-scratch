% Function that computes and visualizes the linear decision boundaries between the
% relative classes 
% Input: input features U, output classes Y, coefficients beta0 and beta

function visualize_LDA(U, Y, beta0, beta)
    
figure;
hold on;

% Plot the features
colors = ['r', 'g', 'b', 'y', 'm'];
for i = 1:5
    scatter(U(Y==i,1), U(Y==i,2), 50, colors(i), 'filled');
end
    
% Set the limits on visualization of inputs and outputs
u_min = min(U(:,1)) - 2;
u_max = max(U(:,1)) + 2;
y_min = min(U(:,2)) - 2;
y_max = max(U(:,2)) + 2;

u1_range = linspace(u_min, u_max, 100);

% Plot only boundaries between adjacent classes
adjacent_pairs = [1 2;   % Class 1 and 2
    2 3;    % Class 2 and 3
    3 4;    % Class 3 and 4
    4 5;    % Class 4 and 5
    5 1];   % Class 5 and 1

for k = 1:size(adjacent_pairs, 1)
    i = adjacent_pairs(k,1);
    j = adjacent_pairs(k,2);
    % Decision boundary between classes i and j is where their discriminants are equal
    % beta(:,i)'*u + beta0(i) = beta(:,j)'*u + beta0(j)
    % (beta(:,i) - beta(:,j))'*u = beta0(j) - beta0(i)

    diff_beta = beta(:,i) - beta(:,j);
    diff_beta0 = beta0(j) - beta0(i);

    if abs(diff_beta(2)) > 1e-10
        % For each u1, solve for u2
        u2_boundary = -(diff_beta(1)*u1_range + diff_beta0)/diff_beta(2);

        valid_idx = u2_boundary >= y_min & u2_boundary <= y_max;
        if any(valid_idx)
            plot(u1_range(valid_idx), u2_boundary(valid_idx), 'k--', 'LineWidth', 1.5);
        end
    end
end

% Set axis limits
xlim([u_min u_max]);
ylim([y_min y_max]);

title('LDA Decision Boundaries and Class Distribution');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5',...
    'Location','best');
grid on
hold off;
end