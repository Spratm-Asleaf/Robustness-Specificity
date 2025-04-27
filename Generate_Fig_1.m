%{
    Online supplementary materials of the paper titled:
    Learning Against Distributional Uncertainty: On the Trade-off Between Robustness and Specificity

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 June 2024
    @Home:     https://github.com/Spratm-Asleaf/Robustness-Specificity
%}

% This MATLAB script is to Generate Figure 1

clear;
clc;

%% Linear Regression
rng(44);       % To fully reproduce Figure 1 

dim = 1;       % dimension of the investigated problem
x_0 = 1;       % True Value
full_sample_len = 1000;
in  = randn(dim, full_sample_len);              % input data
out = x_0'*in + randn(1, full_sample_len);      % output data; out = x' * in + e
epsilon = 1;                             % radius of distributional ball

% True Cost
v = @(x) mean(abs(out - x'*in));

% Training Samples
% Note: Use the 20th sample (only one training sample) to estimate SAA cost; 
%       This is for good visual effect, i.e., to differentiate cost functions as much as possible
train_index = 20;   % You may also try 1:20, using first 20 samples to estimate SAA cost, for a different visual effect

% SAA Cost
vn  = @(x) mean(abs(out(train_index) - x'*in(train_index)));

% DRO Cost
vrn = @(x) vn(x) + epsilon * norm([-x; 1], 2);

% Dispaly Range of x
x = -4:0.01:6;
disp_len = length(x);
V    = zeros(disp_len, 1);
Vn   = zeros(disp_len, 1);
Vrn  = zeros(disp_len, 1);
Vbn  = zeros(disp_len, 1);
Beta = zeros(disp_len, 1);
for i = 1:disp_len
    V(i)   = v(x(i));       % True Cost Evaluation
    Vn(i)  = vn(x(i));      % SAA Cost Evaluation
    Vrn(i) = vrn(x(i));     % DRO Cost Evaluation

    % beta^*_x
    if Vn(i) >= V(i)
        Beta(i) = 0;
    else
        if Vrn(i) - V(i) > 1e-6
            Beta(i) = (V(i) - Vn(i))/(Vrn(i) - Vn(i));      % (13)
        else
            Beta(i) = 0;
        end
    end
end

% Value of Beta
beta = 0.7; 
beta = 0.3; 
beta = max(max(Beta), 0);     % (13)

% BDR Cost Evaluation
for i = 1:disp_len
    Vbn(i) = beta * Vrn(i) + (1 - beta) * Vn(i);
end

%% Plots
figure;
plot(x, V, 'k', x, Vn, 'm', x, Vrn, 'r', x, Vbn, 'b--', 'linewidth', 2);
xlabel('$x$', 'interpreter', 'latex');
ylabel('Cost Functions');
title(['$\beta$ = ' num2str(beta)], 'interpreter', 'latex');
set(gca, 'fontsize', 16);
axis([-4, 6, 0, 10]);
xline(-1.7, 'g--', 'linewidth', 2);
xline(1.7, 'g--', 'linewidth', 2);
legend({'True', 'SAA', 'DRO', 'BDR'}, 'location', 'best');


