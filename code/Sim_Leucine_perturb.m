clear all;
clc

% generate simulated measurement data
X0 = [0.0160   0.2690  0.0055  0.0014  1.0886]';
Tspan = 0:20:450;
Options = [];

[T, X] = ode15s(@ODE_Leucine,Tspan,X0,Options);

Y = X(:,[2 5]); %I2 & P

% options = optimset('MaxFunEvals',20000,'MaxIter',8000);
% ub = [1 1 1 1 0.05 0.05 0.006 0.006 2 0.6 0.6];
% ub = [1 1 1 1 0.05 0.05 0.005 0.005 2 0.6 0.6];
% [theta2,fval,exitflag] = fminsearchbnd(@(theta) MultiOutputGP(theta, T, Y),theta0,lb,ub)

theta0 = [0.5 0.5 0.5 0.5 0.03 0.03 0.01 0.01 0 0.1 0.1];
lb = [0.003 0.003 0.08 0.08 0.01 0.01 0.003 0.003 -2 0.05 0.05];
ub = [1 1 1 1 0.05 0.05 0.01 0.01 5 0.6 0.6];

options = optimoptions('fmincon','Display','iter');
[theta2,fval,exitflag] = fmincon(@(theta) MultiOutputGP(theta, T, Y),theta0,[],[],[],[],lb,ub,[],options)

Ts = [0:2:450]';
[Means, Cov] = MultiOutputGP(theta2, T, Y, Ts);

n = length(Ts);
means = [Means(1:n,:) Means(n+1:2*n,:)];
%means = [Means(1:n,:) Means(n+1:2*n,:) Means(2*n+1:end,:)];

Cov1 = Cov(1:n,1:n);
Cov2 = Cov(n+1:2*n,n+1:2*n);
%Cov3 = Cov(2*n+1:end,2*n+1:end);

s1 = diag(Cov1); s2 = diag(Cov2); %s3 = diag(Cov3);

figure
subplot(1,2,1)
f = [means(:,1)+2*sqrt(s1);flipdim(means(:,1)-2*sqrt(s1),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
plot(Ts,means(:,1),'LineWidth',2)
hold on
plot(T,Y(:,1),'r*')
grid
title('I2 - \beta IPM')
axis([0 400 0.2 3])
%axis([0 400 0.2 10])
xlabel('Time (min)')

subplot(1,2,2)
f = [means(:,2)+2*sqrt(s2);flipdim(means(:,2)-2*sqrt(s2),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
plot(Ts,means(:,2),'LineWidth',2)
hold on
plot(T,Y(:,2),'r*')
grid
title('P - Leucine')
axis([0 400 1 3])
%axis([0 400 1 10])
xlabel('Time (min)')

% subplot(1,3,3)
% f = [means(:,3)+2*sqrt(s3);flipdim(means(:,3)-2*sqrt(s3),1)];
% fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
% hold on
% plot(Ts,means(:,3),'LineWidth',2)
% hold on
% plot(T,Y(:,3),'r*')
% grid
% title('P - Leucine')

%% calculate metabolite derivatives
Ys = means;
meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood
hyp = struct('mean', [], 'cov', [3 0], 'lik', -15);
s1 = zeros(length(Ts),size(Y,2));
derivs = zeros(length(Ts),size(Y,2));
varDeriv = zeros(length(Ts),size(Y,2));
for i=1:size(Ys,2)
    hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, Ts, Ys(:,i));
    [means2, s1(:,i), derivs(:,i), varDeriv(:,i)] = gpr_covSE(Ys(:,i), Ts, Ts, hyp);
    if i==2
        [means2, s1(:,i), derivs(:,i), varDeriv(:,i)] = gpr_covSE(Ys(:,i), Ts, Ts, hyp2);
    end
end

% figure
% f = [derivs+2*sqrt(varDeriv);flipdim(derivs-2*sqrt(varDeriv),1)];
% fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
% hold on
% plot(Ts,derivs,'LineWidth',2)
% legend('','','I2','P')
% title('GP - metabolite derivs')
% grid on

% calculate reaction rates (from metabolite derivatives)
d4 = 0.008;
d5 = 0.077;
v3_gp = derivs(:,2)+d5.*means(:,2);
v2_gp = sum(derivs(:,1:2),2)+d5.*means(:,2);

s3 = sum(varDeriv(:,2),2);
s2 = sum(varDeriv(:,1:2),2);

% plot reaction rates
figure
subplot(1,2,1)
f = [v2_gp+2*sqrt(s2);flipdim(v2_gp-2*sqrt(s2),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
plot(Ts,v2_gp,'LineWidth',2)
axis([0 400 0 0.4])
grid on
xlabel('Time (min)')
ylabel('Reaction rate (mM/min)')
subplot(1,2,2)
f = [v3_gp+2*sqrt(s3);flipdim(v3_gp-2*sqrt(s3),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
plot(Ts,v3_gp,'LineWidth',2)
axis([0 400 0 0.4])
grid on
xlabel('Time (min)')
ylabel('Reaction rate (mM/min)')

%% GP fit to the protein data
Yp = X(:,[3 4]);

hyp0 = [0 -1 -1];
lb = [0 -10 -10];

ub = [3.5 5 1]; % phi_ext=0.1,dt=20
%ub = [4 5 1]; % phi_ext=0.3,dt=20
options = optimoptions('fmincon','Display','iter','Algorithm','interior-point');
means_P = zeros(length(Ts),size(Yp,2));
s_P = zeros(length(Ts),size(Yp,2));
for i=1:size(Y,2)
    [hyp2,fval] = fmincon(@(hyp) gp1(hyp, @infGaussLik, meanfunc, covfunc, likfunc, T, Yp(:,i)),hyp0,[],[],[],[],lb,ub,[],options);
    hyp2 = struct('mean', [], 'cov', [hyp2(1) hyp2(2)], 'lik', hyp2(3));
    [means_P(:,i) s_P(:,i)] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, T, Yp(:,i), Ts);
end

figure
subplot(1,2,1)
f = [means_P(:,1)+2*sqrt(s_P(:,1));flipdim(means_P(:,1)-2*sqrt(s_P(:,1)),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
p1 = plot(Ts,means_P(:,1),'LineWidth',2)
hold on
%plot(T,Yp,'*')
%legend([p1],'E1-Leu1','E2-Leu2')
title('GP - Protein E1')
grid on
xlabel('Time (min)')
ylabel('Concentration (mM)')

subplot(1,2,2)
f = [means_P(:,2)+2*sqrt(s_P(:,2));flipdim(means_P(:,2)-2*sqrt(s_P(:,2)),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
p1 = plot(Ts,means_P(:,2),'LineWidth',2)
hold on
xlabel('Time (min)')
title('GP - Protein E2')
ylabel('Concentration (mM)')
grid on

%% Estimate relative changes in the protein and reaction rate using GPR
phoH_v2 = (log(means_P(:,1))-log(Yp(1,1)))./(log(v2_gp)-log(0.083));
phoH_v3 = (log(means_P(:,2))-log(Yp(1,2)))./(log(v3_gp)-log(0.083));

% reaction catalyzed by E1, mu1 (z_e), mu2 (z_v)
sig1 = sqrt(s_P(:,1))./Yp(1,1);
sig2 = sqrt(s2)./0.083;
mu1 = log(means_P(:,1))-log(Yp(1,1));
mu2 = log(v2_gp)-log(0.083);

%  % reaction catalyzed by E2
% sig1 = sqrt(s_P(:,2))./Yp(1,2);
% sig2 = sqrt(s3)./0.083;
% mu1 = log(means_P(:,2))-log(Yp(1,2));
% mu2 = log(v3_gp)-log(0.083);

% compute the covariance/correlation between mu1 and mu2
Y = [mu1 mu2];
x = Ts;
xstar = Ts;
theta0 = [0.5 0.5 0.5 0.5 10 11 11 9 1 0.03 0.03];
lb = [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 -5 0.01 0.01];
ub = [1 1 1 1 15 15 15 15 5 0.06 0.06]; % phi_ext=0.1
%ub = [1 1 1 1 15 15 17 17 5 0.06 0.06]; % phi_ext=0.1, E2
%ub = [1 1 1 1 15 15 18 18 5 0.06 0.06]; % phi_ext=0.2,0.3
%ub = [1 1 1 1 15 15 17 17 5 0.06 0.06]; % phi_ext=0.3, fminsearch
options = optimoptions('fmincon','Display','iter');
[theta2,fval,exitflag] = fmincon(@(theta) MultiOutputGP(theta, x, Y),theta0,[],[],[],[],lb,ub,[],options);
%[theta2,fval,exitflag] = fminsearchbnd(@(theta) MultiOutputGP(theta, x, Y),theta0,lb,ub)
[Means, Cov] = MultiOutputGP(theta2, x, Y, xstar);

n = size(Y,1);
Yhat = [Means(1:n,:) Means(n+1:end,:)];

CovXX = Cov(1:n,1:n);
CovXY = Cov(1:n,n+1:end);
CovYY = Cov(n+1:end,n+1:end);
covY1Y2 = diag(CovXY);
stdY1 = sqrt(diag(abs(CovXX)));
stdY2 = sqrt(diag(abs(CovYY)));

corrcoeff = covY1Y2./(stdY1.*stdY2);

n = length(xstar);
Yhat = [Means(1:n,:) Means(n+1:end,:)];
Yvar = diag(abs(Cov));
Yhvar = [Yvar(1:n,:) Yvar(n+1:end,:)];

mu1_hat = Yhat(:,1);
mu2_hat = Yhat(:,2);
sig1_hat = sqrt(Yhvar(:,1));
sig2_hat = sqrt(Yhvar(:,2));

phoH_GP = mu1_hat./mu2_hat;
phoH_GP(1)=phoH_GP(2)-0.002;
phoH_GP = smoothdata(phoH_GP,'gaussian',5);

% plot and compare the simulated regulation coefficient (and mu1 and mu2) with GPR (mean)
figure
subplot(1,3,1)
plot(Ts,phoH_v2,'LineWidth',2)
axis([0 300 -0.2 1])
hold on
plot(Ts,phoH_GP,'r-','LineWidth',2)
axis([0 300 -0.2 1])
xlabel('Time (min)')
title('\rho_h (v2)')
hold off
axis([0 400 0 1])

subplot(1,3,2)
plot(Ts,phoH_v3,'LineWidth',2)
axis([0 300 -0.2 1])
%  hold on
%  plot(Ts,phoH_GP,'r-','LineWidth',2)
% % axis([0 300 -0.2 1])
% hold off
xlabel('Time (min)')
axis([0 400 0 1])
title('\rho_h (v3)')

subplot(1,3,3)
f = [Yhat+2*sqrt(Yhvar);flipdim(Yhat-2*sqrt(Yhvar),1)];
fill([xstar; flipdim(xstar,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/18);
hold on
plot(xstar,mu1,'b+')
hold on
plot(xstar,mu2,'r+')
hold on
plot(xstar,Yhat(:,1),'b',xstar,Yhat(:,2),'r','LineWidth',2)
hold off
xlabel('Time (min)')
axis([0 400 -0.2 1])
title('z_e and z_v')

%% Calculte the time-dependent hierarchical regulation coefficient (with confidence intervals)
nS = 800;
delta = 2;
p = zeros(length(s2),nS);
pH_low = zeros(length(s2),1);
pH_up = zeros(length(s2),1);
pH_mean = zeros(length(s2),1);
for i = 1:length(s2)
    p(i,:) = ratio_of_2normal(linspace(phoH_GP(i)-delta,phoH_GP(i)+delta,nS),mu1_hat(i),sig1_hat(i),mu2_hat(i),sig2_hat(i),corrcoeff(i));
    cdfI = cumsum(p(i,:))./max(cumsum(p(i,:)));

    [xL,IndL] = min(abs(cdfI-0.25));
    [xU,IndU] = min(abs(cdfI-0.75));
    [xM,IndM] = min(abs(cdfI-0.5));
    
    stdSamp = linspace(phoH_GP(i)-delta,phoH_GP(i)+delta,nS); 
    pH_low(i,:) = stdSamp(IndL);
    pH_up(i,:) = stdSamp(IndU);
    pH_mean(i,:) = stdSamp(IndM);
end

pH_low1 = pH_low;
pH_up1 = pH_up;
pH_mean1 = pH_mean;
pH_low = smoothdata(pH_low1,'gaussian',5);
pH_up = smoothdata(pH_up1,'gaussian',5);
pH_mean = smoothdata(pH_mean1,'gaussian',5);

pM_low = 1-pH_up;
pM_up = 1-pH_low;

figure
f1 = [pH_up; flipdim(pH_low,1)];
fill([Ts; flipdim(Ts,1)], f1, [7 7 7]/8,'EdgeColor', [7 7 7]/8)
hold on
plot(Ts,phoH_GP,'k-','LineWidth',2)
hold on
axis([0 400 0 1])
xlabel('Time (min)')
ylabel('\rho_h(t)')
grid on
