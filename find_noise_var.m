function [noise_diags_bin, A_bin, upper_bound, d, noise_nobin_min, A_nobin_min, d_nobin_min, noise_cental, A_central] = find_noise_var(sig_X, mul, STEP, NUM_SILENT, distortion_ratio, ITER_MAX)
    warning('off', 'all')
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
    
    sig_1_int = sig_1; sig_2_int = sig_2; sig_3_int = sig_3; sig_4_int = sig_4; sig_5_int = sig_5; 
    sig_6_int = sig_6; sig_7_int = sig_7; sig_8_int = sig_8; sig_9_int = sig_9; sig_10_int = sig_10;
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
    
    
    interval = 10;   % how many points we are examining
    results = zeros(6,interval);
    rate_results = zeros(22,interval);
    num_c = 1;
    
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
    
    
    min_val = cvx_optval;
    Gamma = diag([G_1 G_2 G_3 G_4 G_5 G_6 G_7 G_8 G_9 G_10]);
    
    % binning
    D_star = inv(inv(Gamma) - Pi*inv(lambda)*Pi.');
    noise_diags_inv = inv(D_star) - inv(sig_X);
    for ii = 1:T
        if noise_diags_inv(ii,ii) < 1e-2
            noise_diags_inv(ii,ii) = Inf;
        end
    end
    noise_diags_bin = diag(inv(noise_diags_inv));
    %noise_diags = np.diag(np.linalg.inv( np.linalg.inv(D_star) - np.linalg.inv(sig_X)));
    upper_bound = 0.5*( log(det(Pi.'*sig_X*Pi+lambda)) + log(det(sig)) - log_det(Pi.'*D_star*Pi+lambda) - log_det(Gamma) );
    lower_bound = 0.5*( log(det(Pi.'*sig_X*Pi+lambda)) + log(det(sig)) - log_det(Pi.'*D*Pi+lambda) - log_det(Gamma) );
    gap = upper_bound - lower_bound; 
    A_bin = sig_X/(sig_X + diag(noise_diags_bin));


    % no binning
    avg_rate = upper_bound/10;
    d_nobin_min = 10000;
    noise_nobin_min = zeros(10,1);
    search_vec = linspace(1/4,3/4,200);
    for param = search_vec
        noise_nobin_search = zeros(10,1);
        rate_alloc_base = ones(10,1)*avg_rate*param;
        for ii = 1:T
            noise_nobin_search(ii) = sig_X(ii,ii)/(exp(2*(rate_alloc_base(ii)+ upper_bound*(1-param)*(sig_X(ii,ii)/sum(diag(sig_X))) ))-1);
        end
        % calculate the distortion
        Y_search = inv(sig_X)+inv(diag(noise_nobin_search));
        d_nobin_search = e_T*inv(Y_search)*e;
        % record min
        if d_nobin_search < d_nobin_min
            d_nobin_min = d_nobin_search;                   % output
            noise_nobin_min = noise_nobin_search;           % output
        end
    end

    A_nobin_min = sig_X/(sig_X + diag(noise_nobin_min));    % output


    % centralizing
    sig_X_avg = mean(diag(sig_X));
    noise_cental = sig_X_avg/(exp(2*upper_bound)-1);        % output
    A_central = sig_X_avg/(sig_X_avg+noise_cental);         % output




end