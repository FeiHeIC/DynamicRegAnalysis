function dX = ODE_RegPath(t,X,S,mu,k3)
% Bruggeman's model - IET SysBiol. 2006

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
%k3 = 75;
K3x2 = 1;
K3x3 = 10;
Keq3 = 1;
%Keq3 = 1;
V4 = 100;
K4x3 = 0.4;
K4P = 1;
Keq4 = 10;
Ktrx3 = 0.5;
ntr = 2;
ktrscdeg = 0.5;
ktrnlsyn = 1;
r = 0.0;
%ktrnldeg = 0.5;
ktrnldeg = mu;
%S = 1;
P = 1;
% mRsource = 1;
% mRsink = 1;
% esource = 1;
% esink = 1;

x1 = X(1,:);
x2 = X(2,:);
x3 = X(3,:);
mR = X(4,:);
e = X(5,:);

% Metabolic reactions:
v1 = (e*k1*S*(1 - x1/(Keq1*S))*(((1 + S/K1s + x1/K1x1)/(1 + (cs*S)/K1s + (cp*x1)/K1x1))^n + (cs*L*(1 + S/K1s + x1/K1x1)*(1 + x3/K1x3)^n)/(1 + (cs*S)/K1s + (cp*x1)/K1x1)))/(K1s*(1 + S/K1s + x1/K1x1)*(((1 + S/K1s + x1/K1x1)/(1 + (cs*S)/K1s + (cp*x1)/K1x1))^n + L*(1 + x3/K1x3)^n));
%v1 = (V4*S*(1 - x1/(Keq1*S))*(((1 + S/K1s + x1/K1x1)/(1 + (cs*S)/K1s + (cp*x1)/K1x1))^n + (cs*L*(1 + S/K1s + x1/K1x1)*(1 + x3/K1x3)^n)/(1 + (cs*S)/K1s + (cp*x1)/K1x1)))/(K1s*(1 + S/K1s + x1/K1x1)*(((1 + S/K1s + x1/K1x1)/(1 + (cs*S)/K1s + (cp*x1)/K1x1))^n + L*(1 + x3/K1x3)^n));
v2 = (e*k2*x1*(1 - x2/(Keq2*x1)))/(K2x1*(1 + x1/K2x1 + x2/K2x2));
%v2 = (V4*x1*(1 - x2/(Keq2*x1)))/(K2x1*(1 + x1/K2x1 + x2/K2x2));
v3 = (e*k3*x2*(1 - x3/(Keq3*x2)))/(K3x2*(1 + x2/K3x2 + x3/K3x3));
%v3 = (V4*x2*(1 - x3/(Keq3*x2)))/(K3x2*(1 + x2/K3x2 + x3/K3x3));
v4 = (V4*(1 - P/(Keq4*x3))*x3)/(K4x3*(1 + P/K4P + x3/K4x3));
vTrscsyn = (1 + (x3/Ktrx3)^ntr)^(-1);
vTrscdeg = ktrscdeg*mR;
vTrnlsyn = ktrnlsyn*mR-r;
vTrnldeg = e*ktrnldeg;

% Metabolic model state dynamics:
dX(1,:) = v1 - v2;
dX(2,:) = v2 - v3;
dX(3,:) = v3 - v4;
dX(4,:) = vTrscsyn - vTrscdeg;
dX(5,:) = vTrnlsyn - vTrnldeg;

