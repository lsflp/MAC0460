% file:        perceptron.m
% description: Uses the PLA (perceptron learning algorithm) to generate a rule
%              (generally, a hyperplane) to classify an input set.
% input:       X, a matrix with m lines (m inputs) and n columns (n-dimensioned
%                 input
%              Y, a column vector, that is the expected output for the inputs
%              W, a column vector, with n+1 lines, that represents the inicial
%                 weigths 

function perceptron (X, Y, W)
    m = size(X)(1);
    n = size(X)(2);
      
    X = horzcat(ones([1, m])', X);
    
    while !classified(X, Y, W)
        for i = 1:m
            if (X(i, :)*W) * Y(i) < 0
                W = update(W, X(i, :), Y(i));
            endif
        endfor    
    endwhile
    
    W
    
endfunction


function result = classified(X, Y, W)
    Z = X*W;
    result = true;
    for i = 1:size(X)(1)
        if Z(i)*Y(i) < 0
            result = false;
        endif
        if result == false
            break;
        endif
    endfor        
endfunction


function W = update(W, x, y)
    W = W + y*x';
endfunction
