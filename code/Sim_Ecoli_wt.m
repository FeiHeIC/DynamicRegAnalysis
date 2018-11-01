clear all

load Wt_M %wild-type: metabolite data
load Glng_M %Glng deletion: metabolite data
load Wt_P %wild-type protein data
load Glng_P %Glng deletion: protein data

t = [-40 0 1 2 5 15 35];
t1 = [-22 -7 1 2 5 10 30];

% simulation of a regulatory metabolic pathway
T = [0 1 2 5 15]';
Y = Wt_M([5 6 8 9 10],:);

Y = Y./1000*0.00161; % convert the unit to mM

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

Ts = [0:0.2:16]';
means = zeros(length(Ts),size(Y,2));
s1 = zeros(length(Ts),size(Y,2));

% Multi-output GPR
%    [v1 v2 v3 w1 w2 w3 A1 A2 A3 B1 B2 B3 mu sigma1 sigma2 sigma3]
theta0 = [0.5 0.5 0.5 0.5 0.5 0.5 1 1 1 1 1 1 0 0.1 0.1 0.1]';

lb = [-1 -1 -1 -1 -1 -1 0.1 0.1 0.1 0.1 0.1 0.1 -2 0.001 0.001 0.001];
ub = [1 1 1 1 1 1 5 5 5 4 6 2 2 0.2 0.2 0.2];

%ub = [1 1 1 1 1 1 5 5 5 1.5 1.5 1.5 1 0.2 0.3 0.2]; %
%ub = [1 1 1 1 1 1 5 5 5 1.5 1.5 1.5 1 0.3 0.3 0.3];
%ub = [1 1 1 1 1 1 5 5 5 1.5 2.3 1.5 1 0.2 0.3 0.2]; %

options = optimoptions('fmincon','Display','iter');
[theta2,fval,exitflag] = fmincon(@(theta) MultiOutputGP3(theta, T, Y),theta0,[],[],[],[],lb,ub,[],options)
%options = optimset('MaxFunEvals',10000,'MaxIter',8000);
%[theta2,fval,exitflag] = fminsearchbnd(@(theta) MultiOutputGP3(theta, T, Y),theta0,lb,ub,options)

[Means, Cov] = MultiOutputGP3(theta2, T, Y, Ts);

n = length(Ts);
means = [Means(1:n,:) Means(n+1:2*n,:) Means(2*n+1:end,:)];

Cov1 = Cov(1:n,1:n);
Cov2 = Cov(n+1:2*n,n+1:2*n);
Cov3 = Cov(2*n+1:end,2*n+1:end);

s1 = diag(Cov1); s2 = diag(Cov2); s3 = diag(Cov3);

% plot metabolite data with GPR fitting
figure
subplot(1,3,1)
f = [means(:,1)+2*sqrt(s1);flipdim(means(:,1)-2*sqrt(s1),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
plot(Ts,means(:,1),'LineWidth',2)
hold on
plot(T,Y(:,1),'r*')
grid
xlabel('Time (min)')
ylabel('aKG (mM)')

subplot(1,3,2)
f = [means(:,2)+2*sqrt(s2);flipdim(means(:,2)-2*sqrt(s2),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
plot(Ts,means(:,2),'LineWidth',2)
hold on
plot(T,Y(:,2),'r*')
grid
xlabel('Time (min)')
ylabel('GLU (mM)')

subplot(1,3,3)
f = [means(:,3)+2*sqrt(s3);flipdim(means(:,3)-2*sqrt(s3),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
plot(Ts,means(:,3),'LineWidth',2)
hold on
plot(T,Y(:,3),'r*')
grid
xlabel('Time (min)')
ylabel('GLN (mM)')

%% calculate metabolite derivatives
derivs = zeros(length(Ts),size(Y,2));
varDeriv = zeros(length(Ts),size(Y,2));
hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
Ys = means;
for i=1:size(Ys,2)
    hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, Ts, Ys(:,i))
    [means2, s2(:,i), derivs(:,i), varDeriv(:,i)] = gpr_covSE(Ys(:,i), Ts, Ts, hyp2);
end

% plot metabolite derivatives
figure
f = [derivs+2*sqrt(varDeriv);flipdim(derivs-2*sqrt(varDeriv),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
plot(Ts,derivs,'LineWidth',2)
title('GP - metabolite derivs')
grid on
xlabel('Time (min)')

% calculate reaction rates (from metabolite derivatives)
v1_gp = sum(derivs(:,1:3),2);
v2_gp = sum(derivs(:,2:3),2);
v3_gp = sum(derivs(:,3),2);

s1 = sum(varDeriv(:,[1:3]),2);
s2 = sum(varDeriv(:,[2:3]),2);
s3 = sum(varDeriv(:,3),2);

%% GP fit to the protein data
GS0 = [21163 33785 36805 35893 39702 0 25208 22752 25943 27948 30929]';
GSA = [993;-900;998;299;913;0;12003;12135;13239;7992;2343];
Yp = Wt_P([5 7 8 9 10],:);

GS0 = GS0([5 7 8 9 10],:)./1000*0.00161;
GSA = GSA([5 7 8 9 10],:)./1000*0.00161;
Yp = Yp./1000*0.00161;

Yp = [Yp Yp(:,1)-GSA GS0./Yp(:,1)];

hyp0 = [-1 -1 -15];
lb = [-1 -10 -10];
ub = [1.4 5 1];
options = optimoptions('fmincon','Display','iter','Algorithm','interior-point');
means_P = zeros(length(Ts),size(Yp,2));
s_P = zeros(length(Ts),size(Yp,2));
for i=1:size(Yp,2)
    if i==4
       ub = [1.2 2 0.1];
    elseif i==5
       ub = [1.3 5 0.1];
    end
    [hyp2,fval] = fmincon(@(hyp) gp1(hyp, @infGaussLik, meanfunc, covfunc, likfunc, T, Yp(:,i)),hyp0,[],[],[],[],lb,ub,[],options);
    hyp2 = struct('mean', [], 'cov', [hyp2(1) hyp2(2)], 'lik', hyp2(3));
    [means_P(:,i),s_P(:,i)] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, T, Yp(:,i), Ts);
end

% Plot protein data with GPR fitting
figure
subplot(1,2,1)
f = [means_P(:,1:4)+2*sqrt(s_P(:,1:4));flipdim(means_P(:,1:4)-2*sqrt(s_P(:,1:4)),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
p1 = plot(Ts,means_P(:,1:4),'LineWidth',2)
hold on
plot(T,Yp(:,1:4),'*')
legend([p1],'GS','GOGAT','GDH','GS0')
grid on
xlabel('Time (min)')
ylabel('Concentration (mM)')

subplot(1,2,2)
f = [means_P(:,5)+2*sqrt(s_P(:,5));flipdim(means_P(:,5)-2*sqrt(s_P(:,5)),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
p2 = plot(Ts,means_P(:,5),'LineWidth',2)
hold on
plot(T,Yp(:,5),'*')
legend([p2],'GS0/GS')
grid on
xlabel('Time (min)')

%% Plot time-dependent reaction rates
figure
subplot(1,3,1)
f = [v1_gp+2*sqrt(s1);flipdim(v1_gp-2*sqrt(s1),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
plot(Ts,v1_gp,'LineWidth',2)
xlabel('Time (min)')
ylabel('Reaction rate (mM/min)')

subplot(1,3,2)
f = [v2_gp+2*sqrt(s2);flipdim(v2_gp-2*sqrt(s2),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
plot(Ts,v2_gp,'LineWidth',2)
xlabel('Time (min)')

subplot(1,3,3)
f = [v3_gp+2*sqrt(s3);flipdim(v3_gp-2*sqrt(s3),1)];
fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
plot(Ts,v3_gp,'LineWidth',2)
xlabel('Time (min)')

%% Calculate the hierachical regulation coefficients (mean)
v2_0 = 25;
v3_0 = 1.7;

phoH_GDH_v2 = (log(means_P(:,3))-log(Yp(1,3)))./(log((v2_gp))-log(v2_0));
phoH_GS_v3 = (log(abs(means_P(:,1)))-log(Yp(1,1)))./(log((v3_gp))-log(v3_0));
phoH_GS0_v3 = (log(abs(means_P(:,4)))-log(Yp(1,4)))./(log(v3_gp)-log(v3_0));
phoH_GSs_v3 = (log(abs(means_P(:,5)))-log(Yp(1,5)))./(log(v3_gp)-log(v3_0));
phoH_GOGAT_v2 = (log(means_P(:,2))-log(Yp(1,2)))./(log(v2_gp)-log(v2_0));
phoH_GOGAT_v3 = (log(means_P(:,2))-log(Yp(1,2)))./(log(v3_gp)-log(v3_0));

% phoH_GDH_v2 = ((means_P(:,3)-Yp(1,3))./Yp(1,3))./((v2_gp-v2_0)./v2_0);
% phoH_GS_v3 = ((means_P(:,1)-Yp(1,1))./Yp(1,1))./((v3_gp-v3_0)./v3_0);
% phoH_GS0_v3 = ((means_P(:,4)-Yp(1,4))./Yp(1,4))./((v3_gp-v3_0)./v3_0);
% phoH_GSs_v3 = ((means_P(:,5)-Yp(1,5))./Yp(1,5))./((v3_gp-v3_0)./v3_0);
% phoH_GOGAT_v2 = ((means_P(:,2)-Yp(1,2))./Yp(1,2))./((v2_gp-v2_0)./v2_0);
% phoH_GOGAT_v3 = ((means_P(:,2)-Yp(1,2))./Yp(1,2))./((v3_gp-v3_0)./v3_0);

phoH_GS_v3 = smoothdata(phoH_GS_v3,'gaussian',8);  %0.2
phoH_GOGAT_v3 = smoothdata(phoH_GOGAT_v3,'gaussian',8);  %0.2
phoH_GSs_v3 = smoothdata(phoH_GSs_v3,'gaussian',8);  %0.2
phoH_GS0_v3 = smoothdata(phoH_GS0_v3,'gaussian',8);  %0.2
phoH_GDH_v2 = smoothdata(phoH_GDH_v2,'gaussian',8);  %0.2
phoH_GOGAT_v2 = smoothdata(phoH_GOGAT_v2,'gaussian',8);  %0.2

% phoH_GS_v3 = smoothdata(phoH_GS_v3,'gaussian',12);  %0.2
% phoH_GOGAT_v3 = smoothdata(phoH_GOGAT_v3,'gaussian',12);  %0.2
% phoH_GSs_v3 = smoothdata(phoH_GSs_v3,'gaussian',12);  %0.2

%plot time-dependent hierachical regulation coefficients
figure
subplot(2,2,1)
plot(Ts,phoH_GDH_v2,'LineWidth',2)
axis([0 15 -1 1])
grid on
xlabel('Time (min)')
subplot(2,2,2)
plot(Ts,phoH_GS_v3,'LineWidth',2)
hold on
plot(Ts,phoH_GSs_v3,'LineWidth',2)
grid on
plot(Ts,phoH_GS0_v3,'LineWidth',2)
grid on
xlabel('Time (min)')
axis([0 15 -1 2])
subplot(2,2,3)
plot(Ts,phoH_GOGAT_v2,'LineWidth',2)
grid on
axis([0 15 -1 2])
xlabel('Time (min)')
subplot(2,2,4)
plot(Ts,phoH_GOGAT_v3,'LineWidth',2)
grid on
axis([0 15 -1 2])
xlabel('Time (min)')


mu_GDH = (means_P(:,3)-Yp(1,3))./Yp(1,3);
s_GDH = s_P(:,3)./Yp(1,3);
mu_GS = (means_P(:,1)-Yp(1,1))./Yp(1,1);
s_GS = s_P(:,1)./Yp(1,1);
mu_GOGAT = (means_P(:,2)-Yp(1,2))./Yp(1,2);
s_GOGAT = s_P(:,2)./Yp(1,2);
mu_v2 = (v2_gp-v2_0)./v2_0;
s_v2 = s2./v2_0;
mu_v3 = (v3_gp-v3_0)./v3_0;
s_v3 = s3./v3_0;
mu_GSs = (means_P(:,5)-Yp(1,5))./Yp(1,5);
s_GSs = s_P(:,5)./Yp(1,5);

% figure
% plot(Ts,mu_v3,'r-','LineWidth',2)
% hold on
% plot(Ts,mu_GSs,'b-','LineWidth',2)


% mu_GDH = log((means_P(:,3)))-log(Yp(1,3));
% %mu_GS = log((means_P(:,1)))-log(Yp(1,1));
% mu_GS = log((means_P(:,4)))-log(Yp(1,4));
% mu_GOGAT = log((means_P(:,2)))-log(Yp(1,2));
% mu_GSs = log(means_P(:,5))-log(Yp(1,5));
% %mu_v2 = real(log(v2_gp)-log(v2_0));
% %mu_v3 = real(log(v3_gp)-log(v3_0));

%mu_v2 = log(v2_gp)-log(v2_0);
%mu_v3 = log(v3_gp)-log(v3_0);

phoH_GS_v3 = real(mu_GS./mu_v3);
phoH_GSs_v3 = real(mu_GSs./mu_v3);

phoH_GS_v3 = smoothdata(phoH_GS_v3,'gaussian',8);  %0.2
phoH_GSs_v3 = smoothdata(phoH_GSs_v3,'gaussian',8);  %0.2

% figure
% plot(Ts,phoH_GS_v3,'LineWidth',2)
% hold on
% plot(Ts,phoH_GSs_v3,'LineWidth',2)


%% Calculate the hierachical regulation coefficients (confidence intervals)
figure
% compute the covariance/correlation between mu1 and mu2
for j = 1:4
    if j == 1
        mu1 = mu_GDH;
        mu2 = mu_v2;
        sig1 = s_GDH;
        sig2 = s_v2;
        phoH_GP = phoH_GDH_v2;
        delta = 1.4;
        nD = 800;
    elseif j == 2
        mu1 = mu_GS;
        mu2 = mu_v3;
        sig1 = s_GS;
        sig2 = s_v3;
        phoH_GP = phoH_GS_v3;
        delta = 1.4;
        nD = 800;
    elseif j == 3
        mu1 = mu_GOGAT;
        mu2 = mu_v2;
        sig1 = s_GOGAT;
        sig2 = s_v2;
        phoH_GP = phoH_GOGAT_v2;
        delta = 1.4;
        nD = 800;
    elseif j == 4
        mu1 = mu_GOGAT;
        mu2 = mu_v3;
        sig1 = s_GOGAT;
        sig2 = s_v3;
        phoH_GP = phoH_GOGAT_v3;
        delta = 1.4;
        nD = 800;
    elseif j == 5
        mu1 = mu_GSs;
        mu2 = mu_v3;
        sig1 = s_GSs;
        sig2 = s_v3;
        phoH_GP = phoH_GSs_v3;
        delta = 0.2;
        nD = 800;
    end
    
    Ym = [mu1 mu2];
    x = Ts;
    xstar = Ts;
    theta0 = [0.5 0.5 0.5 0.5 10 11 11 9 1 0.1 0.1]';
    lb = [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 -5 0.01 0.01];
    ub = [1 1 1 1 15 15 15 15 5 0.8 0.8];
    options = optimoptions('fmincon','Display','iter');
    [theta2,fval,exitflag] = fmincon(@(theta) MultiOutputGP(theta, x, Ym),theta0,[],[],[],[],lb,ub,[],options)
    [Means, Cov] = MultiOutputGP(theta2, x, Ym, xstar);
    
    n = size(Ym,1);
    
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
    
    f = [Yhat+2*sqrt(Yhvar);flipdim(Yhat-2*sqrt(Yhvar),1)];
    fill([xstar; flipdim(xstar,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/18);
    hold on
    plot(x,mu1,'b+')
    hold on
    plot(x,mu2,'r+')
    hold on
    plot(xstar,Yhat(:,1),'b',xstar,Yhat(:,2),'r','LineWidth',2)
    
    mu1_hat = Yhat(:,1);
    mu2_hat = Yhat(:,2);
    sig1_hat = sqrt(Yhvar(:,1));
    sig2_hat = sqrt(Yhvar(:,2));
    
    phoH_GP = mu1_hat./mu2_hat;
    phoH_GP(1)=phoH_GP(2)-0.005;
    
    phoH_GP = smoothdata(phoH_GP,'gaussian',5);  %0.2
    %
    nS = 500;
    %delta = 0.1;
    p = zeros(length(s2),nS);
    
    sig1(sig1<0.001)=0.001;
    sig2(sig2<0.001)=0.001;
    for i = 1:length(s2)
        p(i,:) = ratio_of_2normal(linspace(phoH_GP(i)-delta,phoH_GP(i)+delta,nS),mu1_hat(i),sig1_hat(i),mu2_hat(i),sig2_hat(i),corrcoeff(i));
        
        cdfI = cumsum(p(i,:))./max(cumsum(p(i,:)));
        [xL,IndL] = min(abs(cdfI-0.25));
        [xU,IndU] = min(abs(cdfI-0.75));
        [xM,IndM] = min(abs(cdfI-0.5));
        
        stdSamp = linspace(phoH_GP(i)-delta,phoH_GP(i)+delta,nS);  %need to modify the 0.1 according to different E or v std
        pH_low(i,:) = stdSamp(IndL);
        pH_up(i,:) = stdSamp(IndU);
        pH_mean(i,:) = stdSamp(IndM);
    end
    
    pH_low = smoothdata(pH_low,'gaussian',8);
    pH_up = smoothdata(pH_up,'gaussian',8);
    pH_mean = smoothdata(pH_mean,'gaussian',8);
    
    if j<5
        subplot(2,2,j)
        f = [pH_up; flipdim(pH_low,1)];
        fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
        hold on
        plot(Ts,pH_mean,'r-','LineWidth',2)
        hold on
        axis([0 15 -1 1])
        xlabel('Time (min)')
    else
        figure
        f = [pH_up; flipdim(pH_low,1)];
        fill([Ts; flipdim(Ts,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
        hold on
        plot(Ts,pH_mean,'r-','LineWidth',2)
        hold on
        axis([0 15 -1 1])
        xlabel('Time (min)')
    end
end

