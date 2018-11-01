function [out1, out2] = MultiOutputGP(theta, x, Y, xstar)

v1 = theta(1);
v2 = theta(2);
w1 = theta(3);
w2 = theta(4);
A1 = theta(5);
A2 = theta(6);
B1 = theta(7);
B2 = theta(8);
mu = theta(9);
sigma1 = theta(10);
sigma2 = theta(11);

N1 = length(Y(:,1));
N2 = length(Y(:,2));

Sig = A1*A2/(A1+A2);

for i=1:N1
    for j =1:N1
        C11(i,j) = pi^(0.5)*v1^2/sqrt(abs(A1))*exp(-0.25*(x(i)-x(j))^2*A1)+...
            pi^(0.5)*w1^2/sqrt(abs(B1))*exp(-0.25*(x(i)-x(j))^2*B1);
        
        C12(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(x(i)-x(j)-mu)^2*Sig);
        
        C21(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(x(i)-x(j)+mu)^2*Sig);
        
        C22(i,j) = pi^(0.5)*v2^2/sqrt(abs(A2))*exp(-0.25*(x(i)-x(j))^2*A2)+...
            pi^(0.5)*w2^2/sqrt(abs(B2))*exp(-0.25*(x(i)-x(j))^2*B2);
    end
end

C11 = C11 + eye(size(C11,1)).*sigma1^2;
C22 = C22 + eye(size(C22,1)).*sigma2^2;
C = [C11 C12;C21 C22];

if nargin == 3 %
    out1 = 0.5*log(det(C))+0.5*[Y(:,1);Y(:,2)]'*(C\[Y(:,1);Y(:,2)])+(N1+N2)/2*log(2*pi);
    
    %     L = chol(C)';                        % cholesky factorization of the covariance
    %     alpha = solve_chol(L',[Y(:,1);Y(:,2)]);
    %     out1 = 0.5*[Y(:,1);Y(:,2)]'*alpha + sum(log(diag(L))) + (N1+N2)/2*log(2*pi);
    
    if nargout == 2
        
        out2 = zeros(size(theta));       % set the size of the derivative vector
        L = chol(C)';                        % cholesky factorization of the covariance
        alpha = solve_chol(L',[Y(:,1);Y(:,2)]);
        W = L'\(L\eye(size(C,1)))-alpha*alpha';                % precompute for convenience
        for l = 1:length(out2)
            xstar1 = l;
            Nt = length(xstar1);
            for i=1:Nt
                for j =1:N1
                    K11(i,j) = pi^(0.5)*v1^2/sqrt(abs(A1))*exp(-0.25*(xstar1(i)-x(j))^2*A1)+...
                        pi^(0.5)*w1^2/sqrt(abs(B1))*exp(-0.25*(xstar1(i)-x(j))^2*B1)+sigma1^2;
                    
                    K12(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(xstar1(i)-x(j)-mu)^2*Sig);
                    
                    K21(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(xstar1(i)-x(j)+mu)^2*Sig);
                    
                    K22(i,j) = pi^(0.5)*v2^2/sqrt(abs(A2))*exp(-0.25*(xstar1(i)-x(j))^2*A2)+...
                        pi^(0.5)*w2^2/sqrt(abs(B2))*exp(-0.25*(xstar1(i)-x(j))^2*B2)+sigma1^2;
                end
            end
            %K = [K11 K12;K21 K22];
            K = [K11 K22];
            out2(l) = sum(sum(W.*K))/2;
        end
    end
    %out2
    %if nargin == 4 %
else
    Nt = length(xstar);
    for i=1:Nt
        for j =1:N1
            K11(i,j) = pi^(0.5)*v1^2/sqrt(abs(A1))*exp(-0.25*(xstar(i)-x(j))^2*A1)+...
                pi^(0.5)*w1^2/sqrt(abs(B1))*exp(-0.25*(xstar(i)-x(j))^2*B1)+sigma1^2;
            
            K12(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(xstar(i)-x(j)-mu)^2*Sig);
            
            K21(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(xstar(i)-x(j)+mu)^2*Sig);
            
            K22(i,j) = pi^(0.5)*v2^2/sqrt(abs(A2))*exp(-0.25*(xstar(i)-x(j))^2*A2)+...
                pi^(0.5)*w2^2/sqrt(abs(B2))*exp(-0.25*(xstar(i)-x(j))^2*B2)+sigma1^2;
        end
    end
    K = [K11 K12;K21 K22];
    
    for i=1:Nt
        for j =1:Nt
            Kxx11(i,j) = pi^(0.5)*v1^2/sqrt(abs(A1))*exp(-0.25*(xstar(i)-xstar(j))^2*A1)+...
                pi^(0.5)*w1^2/sqrt(abs(B1))*exp(-0.25*(xstar(i)-xstar(j))^2*B1)+sigma1^2;
            
            Kxx12(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(xstar(i)-xstar(j)-mu)^2*Sig);
            
            Kxx21(i,j) = (2*pi)^(0.5)*v1*v2/sqrt(abs(A1+A2))*exp(-0.5*(xstar(i)-xstar(j)+mu)^2*Sig);
            
            Kxx22(i,j) = pi^(0.5)*v2^2/sqrt(abs(A2))*exp(-0.25*(xstar(i)-xstar(j))^2*A2)+...
                pi^(0.5)*w2^2/sqrt(abs(B2))*exp(-0.25*(xstar(i)-xstar(j))^2*B2)+sigma1^2;
        end
    end
    Kxx = [Kxx11 Kxx12;Kxx21 Kxx22];
    
    out1 = K*(C\[Y(:,1);Y(:,2)]);
    if nargout == 2
        out2 = Kxx - K*(C\K');
    end
    
end
end


