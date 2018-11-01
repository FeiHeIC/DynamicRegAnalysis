function [means, S1, derivs, varDeriv] = gpr_covSE(data, input, input_x, hyperparam)

% 13/11/11
%
% GPR using squared exponential covariance function (with Gaussian noise),
% for multiple species in system.  Returns two matrices, each with a column
% per species and row per timepoint (in input_x).  These give the mean
% vectors for the GPs fitted to the function f(x) and its derivative
% df(x)/dt.
%
% Args: data - matrix of output data (row per timepoint, column per
% species), input - data timepoint, input_x - prediction timepoints,
% hyperparam - optimised hyperparams for covariance function.

means = zeros(length(input_x),size(data,2));
derivs = zeros(length(input_x),size(data,2));

for i = 1:size(data,2)
    
    output = data(:,i);
    loghyper = hyperparam(:,i);

    loghyper1 = loghyper.cov(1);
    L2 = exp(2*loghyper1);  invL2 = (1/L2);
    
    meanfunc = [];                    % empty: don't use a mean function
    covfunc = @covSEiso;              % Squared Exponental covariance function
    likfunc = @likGauss;
    [mu,S1] = gp(hyperparam, @infGaussLik, meanfunc, covfunc, likfunc, input, output, input_x);
    
    %     %%%%%%%%%%%%% GRAPHS - GPR mean +- 2sd %%%%%%%%%%%%%%%%%%%%%%
    %     figure(i)
    %     subplot(1,2,1)
    %     f = [mu+2*sqrt(S1);flipdim(mu-2*sqrt(S1),1)];
    %     fill([input_x; flipdim(input_x,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
    %     hold on
    %     plot(input_x,mu,'k-','LineWidth',2);
    %     hold on
    %     plot(input, output, 'kx', 'MarkerSize', 15);
    %     %hold on
    %     %plot(true(:,1),true(:,i+1),'r--','LineWidth',2);
    %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    K = feval(@covSEiso, loghyper.cov, input);
    K = K + eye(size(K))/(1E1);
    
    Kox = feval(@covSEiso, loghyper.cov, input, input_x);
    Kxo = Kox';
    Kxx = feval(@covSEiso, loghyper.cov, input_x);
    
    % mean and variance expressions for function
    invKoutput = K\output;
    
    % calculate matrix of differences (tj-ti for Ldf_xo term)
    tempXo = repmat(input', length(input_x), 1);
    tempXx = repmat(input_x, 1, length(input));
    diffXxXo = tempXo - tempXx;
    
    % calculate matrix of differences (ti-tj for Lfd_ox term)
    diffXoXx = tempXo'- tempXx';
    
    % Ldf_xo and mean for function derivative
    Ldf_xo = invL2*Kxo.*diffXxXo;
    meanDeriv = Ldf_xo*invKoutput;
    
    % Lfd_ox term for function derivative variance
    Lfd_ox = invL2*Kox.*diffXoXx;
    
    % Mxx term for function derivative variance
    tempXxi = repmat(input_x, 1, length(input_x));
    tempXxj = repmat(input_x',length(input_x),1);
    diffXxXx = tempXxi - tempXxj;
    diff2XxXx = diffXxXx.^2;
    invL4 = invL2^2;
    Mxx = invL2.*Kxx - invL4.*diff2XxXx.*Kxx;
    
    % variance for function derivative
    varDeriv = diag(Mxx - Ldf_xo*(K\Lfd_ox));
    
    %     %%%%%%%%%%%%% GRAPHS - GPR deriv +- 2sd %%%%%%%%%%%%%%%%%%%%%
    %     figure(i)
    %     subplot(1,2,2)
    %     f = [meanDeriv+2*sqrt(varDeriv);flipdim(meanDeriv-2*sqrt(varDeriv),1)];
    %     fill([input_x; flipdim(input_x,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
    %     hold on
    %     plot(input_x,meanDeriv,'k-','LineWidth',2);
    %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %     % only need sampling if doing param estimation, not just fitting
    %
    %     if nSamples > 1
    %
    %         % calculate Ldf term for input_x vs input_x times
    %
    %         diffRXxXx = tempXxj-tempXxi;
    %         Ldf_xx = invL2*diffRXxXx.*Kxx;
    %
    %         % generate matrix of random samples
    %
    %         myrmat = randn(length(input_x), nSamples);
    %
    %         for j = 1:nSamples
    %
    %             myr = myrmat(:,j);
    %
    %             % sample function
    %             y = aaa + chol(bbb)'*myr;
    %
    %             % sample deriv
    %             Daaa = Ldf_xx*(Kxx\y);
    %
    %             means_matrix(j,:) = y;
    %             derivs_matrix(j,:) = Daaa;
    %
    %         end
    %
    %         % mean & deriv for GP based on expt data
    %         results{i,1} = mu';
    %         results{i,2} = meanDeriv';
    %
    %         % replicate data samples of mean and deriv from GP function
    %         results{i,3} = means_matrix;
    %         results{i,4} = derivs_matrix;
    %     end
    
    means(:,i) = mu;
    derivs(:,i) = meanDeriv;
    
end

end







