function p = ratio_of_2normal(x,m1,s1,m2,s2,r)
% RATIO_OF_2NORMAL - pdf for the ratio of 2 correlated N(mu1,sigma1)/N(mu2,sigma2)
% 
% Calling:
% p = ratio_of_2normal(x,mu1,sigma1,mu2,sigma2,r)

theta2 = (-s2.^2*m1.*x+r.*s1.*s2.*(m2.*x+m1)-m2.*s1.^2).^2./...
    (2*s1.^2*s2.^2.*(1-r.^2).*(s2.^2.*x.^2-2.*r.*s1.*s2.*x+s1.^2));

%m1 = mu1; m2 = mu2; s1 = sig1; s2 = sig2; r = 0.2;
K2 = exp(-(s2.^2.*m1.^2-2*r*s1.*s2.*m1.*m2+m2.^2.*s1.^2)./(2*(1-r^2).*(s1.*s2)))./(2*pi.*s1.*s2.*sqrt(1-r.^2));

p = K2*2*(1-r.^2).*s1.^2.*s2.^2.*F1(1,0.5,theta2)./(s2^2.*x.^2-2*r*s1.^2.*s2.^2.*x+s1.^2);


function F = F1(a,c,b)
F = 0;
for k=0:50
F = F + (pochhammer(a,k)./pochhammer(c,k)).*(b.^k./factorial(k));
end
