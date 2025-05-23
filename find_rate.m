function [R_bin, R_nobin] = find_rate(sig_X, mul, STEP, NUM_SILENT, distortion_ratio, ITER_MAX)
    sig_X = sig_X * mul;
    T = size(sig_X,1);
    e_T = ones(1,T); e = ones(T,1);

    %%%%%%% initialize sig_1 and sig_2 through a max problem
    cvx_expert true
    cvx_begin quiet sdp
        variable sig_1(1,1) symmetric
        variable sig_2(1,1) symmetric
        variable sig_3(1,1) symmetric
        variable sig_4(1,1) symmetric
        variable sig_5(1,1) symmetric
        variable sig_6(1,1) symmetric
        variable sig_7(1,1) symmetric
        variable sig_8(1,1) symmetric
        variable sig_9(1,1) symmetric
        variable sig_10(1,1) symmetric
        
        maximize(0.5*log_det(diag([sig_1 sig_2 sig_3 sig_4 sig_5 sig_6 sig_7 sig_8 sig_9 sig_10])))
    
        subject to
            0 <= diag([sig_1 sig_2 sig_3 sig_4 sig_5 sig_6 sig_7 sig_8 sig_9 sig_10]) <= sig_X;
    cvx_end
    
    % sig_1_int = sig_1; sig_2_int = sig_2; sig_3_int = sig_3; sig_4_int = sig_4; sig_5_int = sig_5; 
    % sig_6_int = sig_6; sig_7_int = sig_7; sig_8_int = sig_8; sig_9_int = sig_9; sig_10_int = sig_10;
    sig_int = diag([sig_1 sig_2 sig_3 sig_4 sig_5 sig_6 sig_7 sig_8 sig_9 sig_10]);

    % sig_Z could be rank deficient, so projection
    tmp_matrix = diag([inv(sig_1) inv(sig_2) inv(sig_3) inv(sig_4) inv(sig_5) inv(sig_6) inv(sig_7) inv(sig_8) inv(sig_9) inv(sig_10)]) - inv(sig_X);
    [Pi_int,lambda_int] = eig(tmp_matrix);
    for ii = 1:T
        if lambda_int(ii,ii) < 1e-4            %%%%% 1e-6
        % if lambda_int(ii,ii) > 1e5 
            lambda_int(ii,ii) = 0;
            Pi_int(:,ii) = 0;
        end
    end
    Pi_int(:,all(Pi_int == 0))=[];
    lambda_int(:,all(lambda_int == 0))=[];
    lambda_int( ~any(lambda_int,2), : ) = [];  %rows
    for ii = 1:size(lambda_int,1)
        lambda_int(ii,ii) = 1/lambda_int(ii,ii);
    end
    
    
    % interval = 10;   % how many points we are examining
    % results = zeros(6,interval);
    % rate_results = zeros(22,interval);
    % num_c = 1;
    
    max_d = e_T*sig_X*e;
    d = max_d*distortion_ratio;
    
    Pi = Pi_int;
    lambda = lambda_int;
    sig = sig_int;
    
    % minimizing
    cvx_expert true
    cvx_begin quiet sdp
        variable D(T,T) semidefinite
        variable G_1(1,1) diagonal
        variable G_2(1,1) diagonal
        variable G_3(1,1) diagonal
        variable G_4(1,1) diagonal
        variable G_5(1,1) diagonal
        variable G_6(1,1) diagonal
        variable G_7(1,1) diagonal
        variable G_8(1,1) diagonal
        variable G_9(1,1) diagonal
        variable G_10(1,1) diagonal
    
        minimize(0.5*( log(det(Pi.'*sig_X*Pi+lambda)) + log(det(sig)) - log_det(Pi.'*D*Pi+lambda) - log_det(diag([G_1 G_2 G_3 G_4 G_5 G_6 G_7 G_8 G_9 G_10])) ))
    
        subject to
            0 <= D <= sig_X;
            %trace(D) <= d
            e_T*D*e <= d;
            0 <= diag([G_1 G_2 G_3 G_4 G_5 G_6 G_7 G_8 G_9 G_10]);
            0 <= [Pi.'*D*Pi+lambda, Pi.'*D; D*Pi, D-diag([G_1 G_2 G_3 G_4 G_5 G_6 G_7 G_8 G_9 G_10])];
    cvx_end
    
    
    % min_val = cvx_optval;
    % Gamma = diag([G_1 G_2 G_3 G_4 G_5 G_6 G_7 G_8 G_9 G_10]);
    
    % % Compute bounds
    % D_star = inv(inv(Gamma) - Pi*inv(lambda)*Pi.');
    % noise_diags_inv = inv(D_star) - inv(sig_X);
    % for ii = 1:T
    %     if noise_diags_inv(ii,ii) < 1e-1
    %         noise_diags_inv(ii,ii) = Inf;
    %     end
    % end
    % noise_diags = diag(inv(noise_diags_inv));
    % %noise_diags = np.diag(np.linalg.inv( np.linalg.inv(D_star) - np.linalg.inv(sig_X)));
    % upper_bound = 0.5*( log(det(Pi.'*sig_X*Pi+lambda)) + log(det(sig)) - log_det(Pi.'*D_star*Pi+lambda) - log_det(Gamma) );
    % lower_bound = 0.5*( log(det(Pi.'*sig_X*Pi+lambda)) + log(det(sig)) - log_det(Pi.'*D*Pi+lambda) - log_det(Gamma) );
    % gap = upper_bound - lower_bound; 


    M_10 = diag([sig_1 sig_2 sig_3 sig_4 sig_5 sig_6 sig_7 sig_8 sig_9 sig_10]);
    M_9 = diag([sig_1 sig_2 sig_3 sig_4 sig_5 sig_6 sig_7 sig_8 sig_9 G_10]);
    M_8 = diag([sig_1 sig_2 sig_3 sig_4 sig_5 sig_6 sig_7 sig_8 G_9 G_10]);
    M_7 = diag([sig_1 sig_2 sig_3 sig_4 sig_5 sig_6 sig_7 G_8 G_9 G_10]);
    M_6 = diag([sig_1 sig_2 sig_3 sig_4 sig_5 sig_6 G_7 G_8 G_9 G_10]);
    M_5 = diag([sig_1 sig_2 sig_3 sig_4 sig_5 G_6 G_7 G_8 G_9 G_10]);
    M_4 = diag([sig_1 sig_2 sig_3 sig_4 G_5 G_6 G_7 G_8 G_9 G_10]);
    M_3 = diag([sig_1 sig_2 sig_3 G_4 G_5 G_6 G_7 G_8 G_9 G_10]);
    M_2 = diag([sig_1 sig_2 G_3 G_4 G_5 G_6 G_7 G_8 G_9 G_10]); 
    M_1 = diag([sig_1 G_2 G_3 G_4 G_5 G_6 G_7 G_8 G_9 G_10]); 
    M_0 = diag([G_1 G_2 G_3 G_4 G_5 G_6 G_7 G_8 G_9 G_10]); 

    % binning rate
    I_10_left = 0.5*(log(det(lambda* (inv(lambda - Pi.' * M_10 * Pi)) * lambda)) - log(det(lambda* (inv(lambda - Pi.' * M_9 * Pi)) * lambda)));
    I_10_right = 0.5*( log(det(sig_10)) - log(det(G_10) ) );
    I_10 = I_10_left + I_10_right;

    I_9_left = 0.5*(log(det(lambda* (inv(lambda - Pi.' * M_9 * Pi)) * lambda)) - log(det(lambda* (inv(lambda - Pi.' * M_8 * Pi)) * lambda)));
    I_9_right = 0.5*( log(det(sig_9)) - log(det(G_9) ) );
    I_9 = I_9_left + I_9_right;
    
    I_8_left = 0.5*(log(det(lambda* (inv(lambda - Pi.' * M_8 * Pi)) * lambda)) - log(det(lambda* (inv(lambda - Pi.' * M_7 * Pi)) * lambda)));
    I_8_right = 0.5*( log(det(sig_8)) - log(det(G_8) ) );
    I_8 = I_8_left + I_8_right;
    
    I_7_left = 0.5*(log(det(lambda* (inv(lambda - Pi.' * M_7 * Pi)) * lambda)) - log(det(lambda* (inv(lambda - Pi.' * M_6 * Pi)) * lambda)));
    I_7_right = 0.5*( log(det(sig_7)) - log(det(G_7) ) );
    I_7 = I_7_left + I_7_right;
    
    I_6_left = 0.5*(log(det(lambda* (inv(lambda - Pi.' * M_6 * Pi)) * lambda)) - log(det(lambda* (inv(lambda - Pi.' * M_5 * Pi)) * lambda)));
    I_6_right = 0.5*( log(det(sig_6)) - log(det(G_6) ) );
    I_6 = I_6_left + I_6_right;
    
    I_5_left = 0.5*(log(det(lambda* (inv(lambda - Pi.' * M_5 * Pi)) * lambda)) - log(det(lambda* (inv(lambda - Pi.' * M_4 * Pi)) * lambda)));
    I_5_right = 0.5*( log(det(sig_5)) - log(det(G_5) ) );
    I_5 = I_5_left + I_5_right;
    
    I_4_left = 0.5*(log(det(lambda* (inv(lambda - Pi.' * M_4 * Pi)) * lambda)) - log(det(lambda* (inv(lambda - Pi.' * M_3 * Pi)) * lambda)));
    I_4_right = 0.5*( log(det(sig_4)) - log(det(G_4) ) );
    I_4 = I_4_left + I_4_right;
    
    I_3_left = 0.5*(log(det(lambda* (inv(lambda - Pi.' * M_3 * Pi)) * lambda)) - log(det(lambda* (inv(lambda - Pi.' * M_2 * Pi)) * lambda)));
    I_3_right = 0.5*( log(det(sig_3)) - log(det(G_3) ) );
    I_3 = I_3_left + I_3_right;
    
    I_2_left = 0.5*(log(det(lambda* (inv(lambda - Pi.' * M_2 * Pi)) * lambda)) - log(det(lambda* (inv(lambda - Pi.' * M_1 * Pi)) * lambda)));
    I_2_right = 0.5*( log(det(sig_2)) - log(det(G_2) ) );
    I_2 = I_2_left + I_2_right;
    
    I_1_left = 0.5*(log(det(lambda* (inv(lambda - Pi.' * M_1 * Pi)) * lambda)) - log(det(lambda* (inv(lambda - Pi.' * M_0 * Pi)) * lambda)));
    I_1_right = 0.5*( log(det(sig_1)) - log(det(G_1) ) );
    I_1 = I_1_left + I_1_right;
    
    R_bin = I_1 + I_2 + I_3 + I_4 + I_5 + I_6 + I_7 + I_8 + I_9 + I_10;

    % calculate (without binning)
    sig_X1 = sig_X(1,1); sig_X2 = sig_X(2,2); sig_X3 = sig_X(3,3); sig_X4 = sig_X(4,4); sig_X5 = sig_X(5,5);
    sig_X6 = sig_X(6,6); sig_X7 = sig_X(7,7); sig_X8 = sig_X(8,8); sig_X9 = sig_X(9,9); sig_X10 = sig_X(10,10);
    R_1 = 0.5* log( det(sig_X1) * det(1/sig_X1 + 1/G_1 - 1/sig_1) );
    R_2 = 0.5* log( det(sig_X2) * det(1/sig_X2 + 1/G_2 - 1/sig_2) );
    R_3 = 0.5* log( det(sig_X3) * det(1/sig_X3 + 1/G_3 - 1/sig_3) );
    R_4 = 0.5* log( det(sig_X4) * det(1/sig_X4 + 1/G_4 - 1/sig_4) );
    R_5 = 0.5* log( det(sig_X5) * det(1/sig_X5 + 1/G_5 - 1/sig_5) );
    R_6 = 0.5* log( det(sig_X6) * det(1/sig_X6 + 1/G_6 - 1/sig_6) );
    R_7 = 0.5* log( det(sig_X7) * det(1/sig_X7 + 1/G_7 - 1/sig_7) );
    R_8 = 0.5* log( det(sig_X8) * det(1/sig_X8 + 1/G_8 - 1/sig_8) );
    R_9 = 0.5* log( det(sig_X9) * det(1/sig_X9 + 1/G_9 - 1/sig_9) );
    R_10 = 0.5* log( det(sig_X10) * det(1/sig_X10 + 1/G_10 - 1/sig_10) );
    R_nobin = R_1 + R_2 + R_3 + R_4 + R_5 + R_6 + R_7 + R_8 + R_9 + R_10;

end