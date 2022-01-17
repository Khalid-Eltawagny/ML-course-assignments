function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set JJ = sum(H) ; to the cost and grad to the gradient.
%

# getting the cost 
H = X * theta ; 
H = H .- y ; 
H = H .^ 2 ;
J = sum(H) ;
J = (1/(2*m)) * J ; 
tmpTheta = theta(2:end,1) ; 
thetaSum = sum(tmpTheta .^ 2)  * (lambda/(2*m)) ;
J = J + thetaSum ; 

# getting gradients 

H = (X * theta ) .- y ; # 12 * 1 
H = H' ;
H = H * X ; # 1 * 12 X 12 * 2 = 1 * 2 
H = H' ;  # 2 * 1
H = (1/m) .* H ; 
tmpTheta = theta ; 
tmpTheta(1) = 0 ; 
tmpTheta = tmpTheta .* (lambda / m) ; 
grad = grad + (H + tmpTheta) ;  


% =========================================================================

grad = grad(:);

end
