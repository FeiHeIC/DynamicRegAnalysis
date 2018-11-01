function [out1, out2] = MultiOutputGP3(theta, x, Y, xstar)

v1 = theta(1);
v2 = theta(2);
v3 = theta(3);
w1 = theta(4);
w2 = theta(5);
w3 = theta(6);
A1 = theta(7);
A2 = theta(8);
A3 = theta(9);
B1 = theta(10);
B2 = theta(11);
B3 = theta(12);
mu = theta(13);
sigma1 = theta(14);
sigma2 = theta(15);
sigma3 = theta(16);

% v1 = 0.5;
% v2 = 0.5;
% w1 = 0.5;
% w2 = 0.5;
% A1 = theta(1);
% A2 = theta(2);
% B1 = theta(3);
% B2 = theta(4);
% mu = theta(5);
% sigma1 = theta(6);
% sigma2 = theta(6);
% %sigma1 = 0.01;
% %sigma2 = 0.011;
% %sigma1 = 0.1;
% %sigma2 = 0.11;

N1 = length(Y(:,1));
N2 = length(Y(:,2));

Sig12 = A1*A2/(A1+A2);
Sig13 = A1*A3/(A1+A3);
Sig23 = A2*A3/(A2+A3);

for i=1:N1
    for j =1:N1
        C11(i,j) = pi^(0.5)*v1^2/sqrt(abs(A1))*exp(-0.25*(x(i)-x(j))^2*A1)+...
            pi^(0.5)*w1^2/sqrt(abs(B1))*exp(-0.25*(x(i)-x(j))^2*B1);
        C12(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(x(i)-x(j)-mu)^2*Sig12);
        C13(i,j) = (2*pi)^(0.5)*v1*v3/sqrt(abs(A1+A3))*exp(-0.5*(x(i)-x(j)-mu)^2*Sig13);
        C21(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(x(i)-x(j)+mu)^2*Sig12);
        C22(i,j) = pi^(0.5)*v2^2/sqrt(abs(A2))*exp(-0.25*(x(i)-x(j))^2*A2)+...
            pi^(0.5)*w2^2/sqrt(abs(B2))*exp(-0.25*(x(i)-x(j))^2*B2);
        C23(i,j) = (2*pi)^(0.5)*v2*v3/sqrt(abs(A2+A3))*exp(-0.5*(x(i)-x(j)-mu)^2*Sig23);
        C31(i,j) = (2*pi)^(0.5)*v1*v3/sqrt(abs(A1+A3))*exp(-0.5*(x(i)-x(j)+mu)^2*Sig13);
        C32(i,j) = (2*pi)^(0.5)*v2*v3/sqrt(abs(A2+A3))*exp(-0.5*(x(i)-x(j)+mu)^2*Sig23);
        C33(i,j) = pi^(0.5)*v2^2/sqrt(abs(A3))*exp(-0.25*(x(i)-x(j))^2*A3)+...
            pi^(0.5)*w3^2/sqrt(abs(B3))*exp(-0.25*(x(i)-x(j))^2*B3);
    end
end

C11 = C11 + eye(size(C11,1)).*sigma1^2;
C22 = C22 + eye(size(C22,1)).*sigma2^2;
C33 = C33 + eye(size(C33,1)).*sigma3^2;

C = [C11 C12 C13; C21 C22 C23; C31 C32 C33];
%C = C + eye(size(C,1))/(1E3);
%C = C + eye(size(C,1)).*sigma1^2;

if nargin == 3 %
    out1 = 0.5*log(det(C))+0.5*[Y(:,1);Y(:,2);Y(:,3)]'*(C\[Y(:,1);Y(:,2);Y(:,3)])+(N1+N2+N2)/2*log(2*pi);
    
    %     L = chol(C)';                        % cholesky factorization of the covariance
    %     alpha = solve_chol(L',[Y(:,1);Y(:,2)]);
    %     out1 = 0.5*[Y(:,1);Y(:,2)]'*alpha + sum(log(diag(L))) + (N1+N2)/2*log(2*pi);
    
    if nargout == 2
        
        out2 = zeros(size(theta));       % set the size of the derivative vector
        L = chol(C)';                        % cholesky factorization of the covariance
        alpha = solve_chol(L',[Y(:,1);Y(:,2);Y(:,3)]);
        W = L'\(L\eye(size(C,1)))-alpha*alpha';                % precompute for convenience
        for l = 1:length(out2)
            xstar1 = l;
            Nt = length(xstar1);
            for i=1:Nt
                for j =1:N1
                    K11(i,j) = pi^(0.5)*v1^2/sqrt(abs(A1))*exp(-0.25*(xstar1(i)-x(j))^2*A1)+...
                        pi^(0.5)*w1^2/sqrt(abs(B1))*exp(-0.25*(xstar1(i)-x(j))^2*B1);
                    
                    K12(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(xstar1(i)-x(j)-mu)^2*Sig);
                    
                    K21(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(xstar1(i)-x(j)+mu)^2*Sig);
                    
                    K22(i,j) = pi^(0.5)*v2^2/sqrt(abs(A2))*exp(-0.25*(xstar1(i)-x(j))^2*A2)+...
                        pi^(0.5)*w2^2/sqrt(abs(B2))*exp(-0.25*(xstar1(i)-x(j))^2*B2);
                    K33(i,j) = pi^(0.5)*v2^2/sqrt(abs(A3))*exp(-0.25*(xstar1(i)-x(j))^2*A3)+...
                        pi^(0.5)*w3^2/sqrt(abs(B3))*exp(-0.25*(xstar1(i)-x(j))^2*B3);
                end
            end
            K = [K11 K22 K33];
            out2(l) = sum(sum(W.*K))/2;
        end
    end
else
    Nt = length(xstar);
    for i=1:Nt
        for j =1:N1
            K11(i,j) = pi^(0.5)*v1^2/sqrt(abs(A1))*exp(-0.25*(xstar(i)-x(j))^2*A1)+...
                pi^(0.5)*w1^2/sqrt(abs(B1))*exp(-0.25*(xstar(i)-x(j))^2*B1)+sigma1^2;           
            K12(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(xstar(i)-x(j)-mu)^2*Sig12);
            K13(i,j) = (2*pi)^(0.5)*v1*v3/sqrt(abs(A1+A3))*exp(-0.5*(xstar(i)-x(j)-mu)^2*Sig13);            
            K21(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(xstar(i)-x(j)+mu)^2*Sig12);
            K22(i,j) = pi^(0.5)*v2^2/sqrt(abs(A2))*exp(-0.25*(xstar(i)-x(j))^2*A2)+...
                pi^(0.5)*w2^2/sqrt(abs(B2))*exp(-0.25*(xstar(i)-x(j))^2*B2)+sigma2^2;
            K23(i,j) = (2*pi)^(0.5)*v2*v3/sqrt(abs(A2+A3))*exp(-0.5*(xstar(i)-x(j)-mu)^2*Sig23);
            K31(i,j) = (2*pi)^(0.5)*v1*v3/sqrt(abs(A1+A3))*exp(-0.5*(xstar(i)-x(j)+mu)^2*Sig13);
            K32(i,j) = (2*pi)^(0.5)*v2*v3/sqrt(abs(A2+A3))*exp(-0.5*(xstar(i)-x(j)+mu)^2*Sig23);
            K33(i,j) = pi^(0.5)*v2^2/sqrt(abs(A3))*exp(-0.25*(xstar(i)-x(j))^2*A3)+...
                pi^(0.5)*w3^2/sqrt(abs(B3))*exp(-0.25*(xstar(i)-x(j))^2*B3)+sigma3^2;
        end
    end
    K = [K11 K12 K13;K21 K22 K23; K31 K32 K33];
    
    for i=1:Nt
        for j =1:Nt
            Kxx11(i,j) = pi^(0.5)*v1^2/sqrt(abs(A1))*exp(-0.25*(xstar(i)-xstar(j))^2*A1)+...
                pi^(0.5)*w1^2/sqrt(abs(B1))*exp(-0.25*(xstar(i)-xstar(j))^2*B1)+sigma1^2;
            Kxx12(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(xstar(i)-xstar(j)-mu)^2*Sig12);
            Kxx13(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(xstar(i)-xstar(j)-mu)^2*Sig13);
            Kxx21(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(xstar(i)-xstar(j)+mu)^2*Sig12);
            Kxx22(i,j) = pi^(0.5)*v2^2/sqrt(abs(A2))*exp(-0.25*(xstar(i)-xstar(j))^2*A2)+...
                pi^(0.5)*w2^2/sqrt(abs(B2))*exp(-0.25*(xstar(i)-xstar(j))^2*B2)+sigma2^2;
            Kxx23(i,j) = (2*pi)^(0.5)*v2*v3/sqrt(abs(A2+A3))*exp(-0.5*(xstar(i)-xstar(j)-mu)^2*Sig23);
            Kxx31(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A3))*exp(-0.5*(xstar(i)-xstar(j)+mu)^2*Sig13);
            Kxx32(i,j) = (2*pi)^(0.5)*v2*v3/sqrt(abs(A2+A3))*exp(-0.5*(xstar(i)-xstar(j)+mu)^2*Sig23);
            Kxx33(i,j) = pi^(0.5)*v2^2/sqrt(abs(A3))*exp(-0.25*(xstar(i)-xstar(j))^2*A3)+...
                pi^(0.5)*w3^2/sqrt(abs(B3))*exp(-0.25*(xstar(i)-xstar(j))^2*B3)+sigma3^2;
        end
    end
    Kxx = [Kxx11 Kxx12 Kxx13;Kxx21 Kxx22 Kxx23;Kxx31 Kxx32 Kxx33];
    
    out1 = K*(C\[Y(:,1);Y(:,2);Y(:,3)]);
    if nargout == 2
        out2 = Kxx - K*(C\K');
    end
    
end
end


