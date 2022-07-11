function [value,error] = OptionEval_MCM_MVGBM
r = 0.05;
sigma1 = 0.2;
sigma2 = 0.3;
rho = -0.5;
S0 = [100; 120];
MT = 1;
L = [sigma1 0; sigma2*rho sigma2*sqrt(1-rho^2)];
X0 = log(S0);
b = [r-sigma1^2/2;r-sigma2^2/2];
T = [0,MT];
N = 10^4;
payoff = zeros(1,N);
for k =1:N
  X = multiBM_path_simulation(X0,b,L,T);
  S = exp(X(:,end));
  payoff(k) = max(max(S./S0)-1,0);
end
value = mean(payoff);
error = sqrt(var(payoff)/N);

function path = multiBM_path_simulation(x0,b,L,T)
% Simulate a path of the multivariate geometric Brownian motion
% x0 is the initial condition (it has to be a column vector)
% b is a drift coefficient vector
% L is a lower triangular matrix defined by L*L=S where S is the covariance matrix
% T is a time partition of the form [0 t_1 t_2 t_3 ... t_n]
    N = length(T);
    M = length(x0);
    path    = zeros(M,N); %this matrix stores the simulated points
    path(:,1) = x0; %this is our initial condition
 
    for i=2:1:N % this loop simulates the GBM process at time T(i)
      Z = randn(M,1);
      path(:,i) = path(:,i-1) + b*(T(i)-T(i-1)) + L*Z;
    end
end 

end %function
                                    