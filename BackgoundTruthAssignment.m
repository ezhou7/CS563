clear;
addpath('./assignment/');
load('./3d_t_tracking.mat')
D = sTrue(:,1);
D = D{1,1}
X1 = D(:,1);
Y1 = D(:,2);
Z1 = D(:,3);
nF = 50;
assignment = cell(1,50);

for i = 1:1:50
   assignment{i} = {X1(i,1) Y1(i,1) Z1(i,1)};
end

for i = 2:1:20
    D = sTrue(:,i);
    D = D{1,1}
    X2 = D(:,1);
    Y2 = D(:,2);
    Z2 = D(:,3); 
    
	cell_previous = [X1,Y1,Z1];
    cell_now      = [X2,Y2,Z2];	

	% assign the detections to estimated track position
	% cost function 
	est_dist = pdist([cell_previous; cell_now]);
	est_dist = squareform(est_dist);
	est_dist = est_dist(1:nF,nF+1:end);

	% Hungarian algorithm
	[assign, cost] = assignmentoptimal(est_dist);
	assign = assign';
    
    k = 1;
    for F  = 1:size(assign,2) 
		if assign(F) > 0
            assignment{assign(F)}(end+1,:) = {X2(assign(F),1) Y2(assign(F),1) Z2(assign(F),1)};
        end
		k = k +1;
	end
 
    X1 = X2;
    Y1 = Y2;
    Z1 = Z2;
end
figure
groundTruth = cell(1,50);

for i = 1:1:50
    data = assignment(:,i);
    data = data{1,1};
    X = [data{:,1}];
    Y = [data{:,2}];
    Z = [data{:,3}];
    
    X_truth = [];
    Y_truth = [];
    Z_truth = [];
    for j = 1:1:length(X(1,:))
        if X(1,j) < 500
            X_truth = [X_truth,X(1,j)];
            Y_truth = [Y_truth,Y(1,j)];
            Z_truth = [Z_truth,Z(1,j)];
            groundTruth{i}(end+1,:) = {X(1,j) Y(1,j) Z(1,j)};
        end
    end


    maxx=max([data{:,1}]);
    minx=min([data{:,1}]);
    maxy=max([data{:,2}]);
    miny=min([data{:,2}]);
   
    
    scatter3(X_truth,Y_truth,Z_truth,'filled')
%     scatter3(X,Y,Z,'filled')
    view(35,30)
    axis equal

    axis([0 510 0 510 0 100]);

%     zlim([0, 100]);

%     axis([minx maxx miny maxy 0 100]);
    hold on
    pause
       
end
save("./backgroundTruth.mat",'groundTruth')
