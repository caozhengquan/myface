function [mappedX, mapping] = TransferBayesian(X, labels)
% Joint Bayesian
% Chen D, Cao X, Wang L, et al. Bayesian face revisited: A joint formulation, ECCV 2012.
% 
% programmed by happynear
% https://github.com/happynear
    m = length(labels);
    n = size(X,2);
	
	% Make sure labels are nice
	[classes, bar, labels] = unique(labels);
    nc = length(classes);
	
	% Intialize Sw
	Sw = zeros(size(X, 2), size(X, 2));
    
   cur = {};
    withinCount = 0;
    numberBuff = zeros(1000,1);
%     numberInvert = zeros(1000,1);
    maxNumberInOneClass = [];
    for i=1:nc
        % Get all instances with class i
        cur{i} = X(labels == i,:);
        if size(cur{i},1)>1
            withinCount = withinCount + size(cur{i},1);
        end;
        if numberBuff(size(cur{i},1)) ==0
            numberBuff(size(cur{i},1)) = 1;
            maxNumberInOneClass = [maxNumberInOneClass;size(cur{i},1)];
        end;
    end;
    disp([nc withinCount]);
    fprintf('prepare done, maxNumberInOneClass=%d.\n',length(maxNumberInOneClass));
    tic;
    u = zeros(n,nc);
%     ep = zeros(n,withinCount);
%     nowp = 1;
	% Sum over classes
% 	for i=1:nc
% 		% Update within-class scatter
%         u(:,i) = mean(cur{i},1)';
%         
%         if size(cur{i},1)>1
%             ep(:,nowp:nowp+ size(cur{i}, 1)-1) = bsxfun(@minus,cur{i}',u(:,i));
%             nowp = nowp + size(cur{i}, 1);
% %             C = cov(cur{i});
% %             p = size(cur{i}, 1) / withinCount;
% %             Sw = Sw + (p * C);
%         end;
% 	end;
%         Su = cov(u');
%         Sw = cov(ep');

%     Su = u*u'/nc;
%     Sw = ep*ep'/withinCount;
%     Su = cov(rand(n,n));
%     Sw = cov(rand(5*n,n));
%     fprintf('LDA matrix done.\n');
%     toc;
%     F = inv(Sw);
%     mapping.Su = Su;
%     mapping.Sw = Sw;
%     mapping.G = -1 .* (2 * Su + Sw) \ Su / Sw;
%     mapping.A = inv(Su + Sw) - (F + mapping.G);
%     mappedX = X;
% end
    
    oldTw = Sw;
%     Gs = cell(maxNumberInOneClass,1);
    TuFG = cell(1000,1);
    TwG = cell(1000,1);
load('sourcemapping.mat');
Su=mapping.Su;
Sw=mapping.Sw;
lamda=0.8;
    Tu=Su;
    Tw=Sw;
    for l=1:500
%         tic;
        F = inv(Tw);
        ep =zeros(n,m);
        nowp = 1;
        for g = 1:1000%1000
            if numberBuff(g)==1
                G = -1 .* (g .* Tu + Tw) \ Tu / Tw;
                TuFG{g} = Tu * (F + g.*G);
                TwG{g} = Tw*G;
            end;
        end;
        for i=1:nc
            nnc = size(cur{i}, 1);
%             G = Gs{nnc};
            u(:,i) = sum(TuFG{nnc} * cur{i}',2);
            ep(:,nowp:nowp+ size(cur{i}, 1)-1) = bsxfun(@plus,cur{i}',sum(TwG{nnc}*cur{i}',2));
            nowp = nowp+ nnc;
        end;
        Tu = (1-lamda)*Su+lamda*cov(u');
        Tw = (1-lamda)*Sw+lamda*cov(ep');
%     Su = u*u'/nc;
%     Sw = ep*ep'/withinCount;
        fprintf('%d %f\n',l,norm(Tw - oldTw)/norm(Tw));
%         toc;
        if norm(Tw - oldTw)/norm(Tw)<1e-6
            break;
        end;
        oldTw = Tw;
    end;
    F = inv(Tw);
    mapping.G = -1 .* (2 * Tu + Tw) \ Tu / Tw;
    mapping.A = inv(Tu + Tw) - (F + mapping.G);
    mapping.Sw = Tw;
    mapping.Su = Tu;
%     mapping.U = chol(-G,'upper');
%     mapping.COEFF = COEFF;
%     mapping.y = mapping.U * X';
    mapping.c = zeros(m,1);
    for i = 1:m
        mapping.c(i) = X(i,:) * mapping.A * X(i,:)';
    end;
    mappedX = X;
   