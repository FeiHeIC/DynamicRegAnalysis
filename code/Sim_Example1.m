% simulation & regulation analysis of an unbranched metabolic pathway with
% negative feedback regulation

clear all;
clc

% set up initial conditions
X0 = [4.826 1.89 0.7 0.7 1.34]'; %steady state with S=0.1
Tspan = [0:0.5:20];
Options = [];

% pathway parameters
k1 = 100;
K1s = 0.05;
L = 0.8;
cs = 0.0001;
cp = 0.0001;
K1x3 = 1;
K1x1 = 1000;
n = 3;
Keq1 = 100;
k2 = 50;
K2x1 = 1;
K2x2 = 1;
Keq2 = 10;
k3 = 75;
K3x2 = 1;
K3x3 = 10;
Keq3 = 1;
V4 = 100;
K4x3 = 0.4;
K4P = 1;
Keq4 = 10;
Ktrx3 = 0.5;
ntr = 2;
ktrscdeg = 0.5;
ktrnlsyn = 1;
ktrnldeg = 0.5;
S = 1;
P = 1;

% simulation with S=1
[T, X] = ode15s(@ODE_RegPath,Tspan,X0,Options,S,0.5,k3);

% calculate pathway fluxes before perturbation
x1 = X0(1);
x2 = X0(2);
x3 = X0(3);
mR = X0(4);
e = X0(5);
S = 0.1;
v10 = (e.*k1.*S.*(1 - x1./(Keq1.*S)).*(((1 + S./K1s + x1./K1x1)./(1 + (cs.*S)./K1s + (cp*x1)./K1x1)).^n + (cs.*L.*(1 + S./K1s + x1./K1x1).*(1 + x3./K1x3).^n)./(1 + (cs.*S)./K1s + (cp.*x1)./K1x1)))./(K1s.*(1 + S./K1s + x1./K1x1).*(((1 + S./K1s + x1./K1x1)./(1 + (cs.*S)./K1s + (cp.*x1)./K1x1)).^n + L.*(1 + x3./K1x3).^n));
v20 = (e.*k2.*x1.*(1 - x2./(Keq2.*x1)))./(K2x1.*(1 + x1./K2x1 + x2./K2x2));
v30 = (e.*k3.*x2.*(1 - x3./(Keq3.*x2)))./(K3x2.*(1 + x2./K3x2 + x3./K3x3));
v40 = (V4.*(1 - P./(Keq4.*x3)).*x3)./(K4x3.*(1 + P./K4P + x3./K4x3));
% calculate pathway fluxes after perturbation
x1 = X(:,1);
x2 = X(:,2);
x3 = X(:,3);
mR = X(:,4);
e = X(:,5);
S = 1;
v1 = (e.*k1.*S.*(1 - x1./(Keq1.*S)).*(((1 + S./K1s + x1./K1x1)./(1 + (cs.*S)./K1s + (cp*x1)./K1x1)).^n + (cs.*L.*(1 + S./K1s + x1./K1x1).*(1 + x3./K1x3).^n)./(1 + (cs.*S)./K1s + (cp.*x1)./K1x1)))./(K1s.*(1 + S./K1s + x1./K1x1).*(((1 + S./K1s + x1./K1x1)./(1 + (cs.*S)./K1s + (cp.*x1)./K1x1)).^n + L.*(1 + x3./K1x3).^n));
v2 = (e.*k2.*x1.*(1 - x2./(Keq2.*x1)))./(K2x1.*(1 + x1./K2x1 + x2./K2x2));
v3 = (e.*k3.*x2.*(1 - x3./(Keq3.*x2)))./(K3x2.*(1 + x2./K3x2 + x3./K3x3));
v4 = (V4.*(1 - P./(Keq4.*x3)).*x3)./(K4x3.*(1 + P./K4P + x3./K4x3));
vTrnlsyn = ktrnlsyn.*mR;
vTrnldeg = e.*ktrnldeg;

v3(1) = v3(1)+0.1;
v2(1) = v2(1)+0.1;

% cacluate & plot the 'simulated' regulation coefficients
phoH = ((e-X0(5))./X0(5))./((v1-v10)./v10);
phoM = 1 - phoH;

figure
plot(Tspan,phoH,'r')
hold on
plot(Tspan,phoM)
hold off
legend('PhoH','PhoM')
grid on
xlabel('Time (min)')
ylabel('\rho_h(t) & \rho_m(t)')
title('Hierarchical and metabolic regulation coefficients based on simulation')

%% GPR fitting to the simulated measurement data
% generate measurement data
randn('seed',12356)
Y = X+0.05.*randn(size(X));

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

Ts = [0:0.2:20]';
means = zeros(length(Ts),size(X,2));
s1 = zeros(length(Ts),size(X,2));
derivs = zeros(length(Ts),size(X,2));
varDeriv = zeros(length(Ts),size(X,2));

% GP fitting to the data and compute GP derivatives
hyp0 = [-1 -1 -1];
options = optimoptions('fmincon','Display','iter','Algorithm','interior-point');
for i=1:size(X,2)
    if i==1
        lb = [1.7 -15 -15];
        ub = [2.5 9 5];
    elseif i==2
        lb = [-1 -10 -9];
        ub = [3 5 1];
    elseif i==5
        lb = [-1 -10 -9];
        ub = [2 5 1];
    else
        lb = [-1 -10 -9];
        ub = [1 2 1];
    end
    [hyp2,~] = fmincon(@(hyp) gp1(hyp, @infGaussLik, meanfunc, covfunc, likfunc, T, Y(:,i)),hyp0,[],[],[],[],lb,ub,[],options);
    hyp2 = struct('mean', [], 'cov', [hyp2(1) hyp2(2)], 'lik', hyp2(3));
    [means(:,i), s1(:,i), derivs(:,i), varDeriv(:,i)] = gpr_covSE(Y(:,i), T, Ts, hyp2);
end

% plot metabolite data with GPR fitting
figure
subplot(1,2,1)
f = [means(:,1)+2*sqrt(s1(:,1));flipdim(means(:,1)-2*sqrt(s1(:,1)),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
p1 = plot(Ts,means(:,1),'LineWidth',2);
hold on
plot(Tspan,Y(:,1),'b.','LineWidth',2)
legend([p1],'X1')
grid
xlabel('Time (min)')
axis([0 15 0 55])

subplot(1,2,2)
f = [means+2*sqrt(s1);flipdim(means-2*sqrt(s1),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
plot(Tspan,Y(:,2:5),'b.','LineWidth',2)
hold on
p1 = plot(Ts,means(:,2:5),'LineWidth',2);
hold on
grid
legend([p1],'X2','X3','mR','e')
axis([0 15 0 5])
xlabel('Time (min)')
grid on

%% caculate time-dependent reaction rates
[T, X1] = ode15s(@ODE_RegPath,Ts',X0,Options,S,0.5,k3);
x3 = X1(:,3);
v4 = (V4.*(1 - P./(Keq4.*x3)).*x3)./(K4x3.*(1 + P./K4P + x3./K4x3));

s2 = sum(varDeriv(:,[1:3]),2);
s3 = sum(varDeriv(:,[2:3]),2);
s4 = sum(varDeriv(:,3),2);

% calculate the time-dependent regulation coefficients w.r.t to each
% reaction: v1, v2, v3
phoH_GP = ((means(:,5)-X0(5))./X0(5))./(((v4+sum(derivs(:,1:3),2))-v10)./v10); %reaction 1
%phoH_GP = ((means(:,5)-X0(5))./X0(5))./(((v4+sum(derivs(:,2:3),2))-v20)./v20);
%%reaction 2
%phoH_GP = ((means(:,5)-X0(5))./X0(5))./(((v4+sum(derivs(:,3),2))-v30)./v30);
%%reaction 3
phoM_GP = 1 - phoH_GP;

mu1 = (means(:,5)-X0(5))./X0(5);
sig1 = sqrt(s1(:,5))./(X0(5));

 mu2 = (((v4+sum(derivs(:,1:3),2))-v10)./v10); %reaction 1
 sig2 = sqrt(s2)./(v10);

% mu2 = (((v4+sum(derivs(:,2:3),2))-v20)./v20); %reaction 2
% sig2 = sqrt(s3)./(v20);

%mu2 = (((v4+sum(derivs(:,3),2))-v30)./v30); %reaction 3
%sig2 = sqrt(s4)./(v30);

% calculate reaction rates (from metabolite derivatives)
v1_gp = (v4+sum(derivs(:,1:3),2));
v2_gp = (v4+sum(derivs(:,2:3),2));
v3_gp = (v4+sum(derivs(:,3),2));

% Plot time-dependent reaction rates
figure
subplot(1,3,1)
f1 = [v1_gp+2*sqrt(s2(:,1)); flipdim(v1_gp-2*sqrt(s2(:,1)),1)];
fill([T; flipdim(T,1)], f1, [7 7 7]/8,'EdgeColor', [7 7 7]/8);
hold on
plot(Tspan,v1,'r','LineWidth',2)
hold on
plot(Ts,v1_gp,'b','LineWidth',2)
legend('95% CI','v_1-simulation','v_1-GP estimate')
grid on
xlabel('Time (min)')
ylabel('Reaction rate (mM/min)')
axis([0 15 40 120])

subplot(1,3,2)
f1 = [v2_gp+2*sqrt(s3(:,1)); flipdim(v2_gp-2*sqrt(s3(:,1)),1)];
fill([T; flipdim(T,1)], f1, [7 7 7]/8,'EdgeColor', [7 7 7]/8);
hold on
plot(Tspan,v2,'r','LineWidth',2)
hold on
plot(Ts,v2_gp,'b','LineWidth',2)
legend('95% CI','v_2-simulation','v_2-GP estimate')
grid on
xlabel('Time (min)')
axis([0 15 40 120])

subplot(1,3,3)
f1 = [v3_gp+2*sqrt(s4(:,1)); flipdim(v3_gp-2*sqrt(s4(:,1)),1)];
fill([T; flipdim(T,1)], f1, [7 7 7]/8,'EdgeColor', [7 7 7]/8);
hold on
plot(Tspan,v3,'r','LineWidth',2)
hold on
plot(Ts,v3_gp,'b','LineWidth',2)
legend('95% CI','v_3-simulation','v_3-GP estimate')
grid on
xlabel('Time (min)')
axis([0 15 40 120])

%% Estimate relative changes in the protein and reaction rate using GPR
Y = [mu1 mu2];
x = Ts;
xstar = Ts;
theta0 = [0.5 0.5 0.5 0.5 9 9 9 9 0 0.1 0.1]';
lb = [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 -5 0.01 0.01];
ub = [1 1 1 1 15 15 15 15 5 0.8 0.8];

options = optimoptions('fmincon','Display','iter');

[theta2,fval,exitflag] = fminsearchbnd(@(theta) MultiOutputGP(theta, x, Y),theta0,lb,ub)

%[theta2,fval,exitflag] = fmincon(@(theta) MultiOutputGP(theta, x, Y),theta0,[],[],[],[],lb,ub,[],options)
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
Yvar = diag(Cov);
Yhvar = [Yvar(1:n,:) Yvar(n+1:end,:)];

figure
f = [Yhat+2*sqrt(Yhvar);flipdim(Yhat-2*sqrt(Yhvar),1)];
fill([xstar; flipdim(xstar,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/18);
hold on
plot(xstar,mu1,'b+')
hold on
plot(xstar,mu2,'r+')
hold on
plot(xstar,Yhat(:,1),'b',xstar,Yhat(:,2),'r','LineWidth',2)
legend('95% CI','95% CI','z_e','z_v')
xlabel('Time (min)')
title('z_e and z_v')

% figure
% plot(xstar,mu1./mu2,'k+')

mu1_hat = Yhat(:,1);
mu2_hat = Yhat(:,2);
sig1_hat = sqrt(Yhvar(:,1));
sig2_hat = sqrt(Yhvar(:,2));

%% Calculate the time-dependent regulation coefficients (with confidence intervals)

nS = 900;
delta = 4;
p = zeros(length(s2),nS);
for i = 1:length(s2)
    p(i,:) = ratio_of_2normal(linspace(phoH_GP(i)-delta,phoH_GP(i)+delta,nS),mu1(i),sig1(i),mu2(i),sig2(i),corrcoeff(i));
    
    cdfI = cumsum(p(i,:))./max(cumsum(p(i,:)));
    [xL,IndL] = min(abs(cdfI-0.25));
    [xU,IndU] = min(abs(cdfI-0.75));
    [xM,IndM] = min(abs(cdfI-0.5));
    
    stdSamp = linspace(phoH_GP(i)-delta,phoH_GP(i)+delta,nS); 
    pH_low(i,:) = stdSamp(IndL);
    pH_up(i,:) = stdSamp(IndU);
    pH_mean(i,:) = stdSamp(IndM);
end

% ns=20;
% figure
% %plot(linspace(phoH_GP(ns)-delta,phoH_GP(ns)+delta,500),p(ns,:))
% plot(linspace(phoH_GP(ns)-delta,phoH_GP(ns)+delta,nS),p(ns,:)./max(p(ns,:)))
% hold on
% %plot(phoH_GP(ns),0,'r.')
% hold on

pM_low = 1-pH_up;
pM_up = 1-pH_low;

figure
f1 = [pH_up; flipdim(pH_low,1)];
fill([Ts; flipdim(Ts,1)], f1, [7 7 7]/8,'EdgeColor', [7 7 7]/8)
hold on
plot(Tspan,phoH,'r.')
hold on
plot(Ts,phoH_GP,'r-','LineWidth',2)
hold on

f1 = [pM_up; flipdim(pM_low,1)];
fill([Ts; flipdim(Ts,1)], f1, [7 7 7]/8,'EdgeColor', [7 7 7]/8)
hold on
plot(Tspan,phoM,'b.')
hold on
plot(Ts,phoM_GP,'b-','LineWidth',2)
hold on
legend('95% CI','PhoH','PhoH-GP','95% CI','PhoM','PhoM-GP')
grid on
xlabel('Time (min)')
ylabel('\rho_h(t) & \rho_m(t)')
axis([0 15 -4 5])

