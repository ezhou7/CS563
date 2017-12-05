 % background truth
clear;
load('./3d_t_tracking.mat')
addpath('./20_cells_txt/');

count = [];
% figure
estimation = cell(1,20);
for time = 1:1:20
    path = sprintf('%s%d%s', './20_cells_txt/3d_cells_', time,'.txt');

    estimation{1,time} = load(path);
end

% % The background truth
for i = 1:1:20
    P = estimation(:,i);
    P = P{1,1}
    XP = P(:,1);
    YP = P(:,2);
    ZP = P(:,3);
    
    D = sTrue(:,i);
    D = D{1,1}
    X = D(:,1);
    Y = D(:,2);
    Z = D(:,3);
    c = 0
    for i = i:1:length(Z(:,1))
        if X(i,1) <=500 && Z(i,1) > 0 && Z(i,1)<100
            c = c + 1;
        end
    end
    count = [count;c]
%     maxx=max(X);
%     minx=min(X);
%     maxy=max(Y);
%     miny=min(Y);
    figure
    scatter3(XP,YP,ZP,'r','filled')
    hold on 
    scatter3(X,Y,Z,'b','filled')
    colormap(jet); 
    % scatter3(x,y,z,'*')
    % scatter3(x,y,z,'MarkerFaceColor',[0 .75 .75])
    view(35,30)
    axis equal
    zlim([0, 100]);

%     axis([0 510 0 510 0 100]);
%     hold on
    pause
end
% 21+1 27 32+1 35 39+2 45 47 49+1 
