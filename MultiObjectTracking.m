%% Cell detection
clear;
addpath('./2D_images/');
addpath('./assignment/');

%get listing of frames

frame_list = dir('./2D_images/*jpg');

% % initialize gaussian filter-self define parameters
hsize = 10000;
sigma =  500;
h = fspecial('log',hsize,sigma);
% subplot(121);
% imagesc(h)
% subplot(122);
% mesh(h)
% colormap(jet)

% iteratively find cells and save the detected position
X = cell(1,length(frame_list));
Y = cell(1,length(frame_list));

for i = 1: length(frame_list)
	% reduce the dimension to greyscale
    img_real = (imread(frame_list(i).name));
    
	img_tmp = double(imread(frame_list(i).name));
    img_tmp = img_tmp(:,:,1);  
    
% 	do the blob filter
	blob_img = conv2(img_tmp,h,'same');
%	threshold to image to blobs
	idx = find(blob_img < 0.005);
	blob_img(idx) = nan;
    
	% find the blob peak indices for this frame
	[zmax,imax,zmin,imin] = extrema2(blob_img);
	[X{i},Y{i}] = ind2sub(size(blob_img),imax);
    subplot(2,1,1);
    imagesc(blob_img)
        axis off
    subplot(2,1,2);
	imshow(img_real)
	hold on 
	for j = 1: length(X{i})
		plot(Y{i}(j),X{i}(j),'r.','MarkerSize',15)
    end 
	axis off
	pause
	i
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Prediction and Estimation
% %% Kalman Filter
% %% define main variables
% dt = 1; % sampling rate
% start_frame = 1; % starting frame
% u = 0.005 % define acceleration magnitude
% 
% thn_x = .1; % [self-define] measurement noise in the x direction (how confident of the estimated result)
% thn_y = .1; % [self-define]measurement noise in the y direction
% Ez = [thn_x 0; 0 thn_y]
% 
% noise_magnitude = .1; % [self-define]how fast the objects changes positions
% Ex = [dt^4/4  0       dt^3/2  0;
%       0       dt^4/4  0       dt^3/2;
%       dt^3/2  0       dt^2    0;
%       0       dt^3/2  0       dt^2] * noise_magnitude^2
% P = Ex
% 
% %% define coefficient matrix
% A = [1 0 dt 0; 0 1 0 dt;0 0 1 0;0 0 0 1];
% B = [(dt^2/2);(dt^2/2);dt; dt]
% C = [1 0 0 0;0 1 0 0]
% 
% %%initialize result variables
% %% estimated position of the cell in the next frame
% Cell_location_measure = [];
% 
% %% initialize estimation variables for two dimensions
% Cell = [X(start_frame,1) Y(start_frame,2) zeros(length(X(start_frame)), 1) zeros(length(X(start_frame)), 1) % initial state
% 
% Cell_estimate = nan(4,20000);
% Cell_estimate(:,1:size(Cell,2)) = Cell;
% Cell_location_estimateY = nan(2000);
% Cell_location_estimateX = nan(2000);
% P_estimate = P;
% strk_traks = zeros(1,2000);
% nD = size(X(start_frame),1);
% nF = find(isnan(Cell_estimate(1,:)) == 1,1)-1;
% THRESHOLD = 50;
% 
% for t = start_frame:length(frame_list)-1
% 	% load the image
% 	img_tmp = double(imread(frame_list(t).name));
% 	img = img_tmp(:,:,1);
% 	Cell_location_measure = [X(t) Y(t)];
% 	% do the Kalman Filter
% 	nD = size(X(t),1);
% 	for F = 1:nF
% 		Cell_estimate(:,F) = A * Cell_estimate(:,F)+ B*u;
% 	end
%  	% predict next covariance
% 	P = A * P * A' + Ex;
% 	% Kalman Gain
% 	K = P*C'*inv(C*P*C'+Ez)
% 
% 	%% assign the detections to estimated track position
% 	% cost function 
% 	est_dist = pdost([Cell_estimate(1:2,1:nF)'; Cell_location_measure]);
% 	est_dist = squareform(est_dist);
% 	est_dist = est_dist(1:nF,nF+1:end)
% 
% 	% Hungarian algorithm
% 	[assign, cost] = ASSIGNMENTOPTIMAL(est_dist);
% 	assign = assign';
% 	% check1: is the detection far from the observation? If yes, reject it.
% 	rej = [];
% 	for F = 1:nF
% 		if assign(F) > 0
% 			rej(F) = est_dist(F , assign(F)) < THRESHOLD
% 		else
% 			rej(F) = 0;
% 		end
% 	end
% 	assign = assign.* rej
% 
% 	% apply the assignment to the update
% 	k = 1;
% 	for F  = 1:length(assign)
% 		if assign(F) > 0
% 			Cell_estimate(:,k) = Cell_estimate(:,k) + K*(Cell_location_measure(assign(F),:)'-C*Cell_estimate(:,k));
% 		end
% 		k = k +1
% 	end
% 	% update covariance estimation
% 	P = (eye(4)-K*C)*P
% 
% 	%% store data
% 	Cell_location_estimate(t,1:nF) = Cell_estimate(1,1:nF);
% 	Cell_location_estimate(t,1:nF) = Cell_estimate(2,1:nF);
% 
% 	% find the new detections and lost detections
% 	new_track = [];
% 	new_track = Cell_location_estimate;
% 	if ~isempty(new_track)
% 		Cell_estimate(:,nF+1:nF+size(new_track,2)) = [new_track; zeros(2,size(new_track,2))];
% 		nF = nF + size(new_track,2);
% 	end
% 
% 	% give a strike to any tracking that did not matched up to a detection
% 	no_track_list = find(assign == 0);
% 	if ~isempty(no_track_list)
% 		strk_traks(no_track_list) =  strk_traks(no_track_list)+1;
% 	end
% 
% 	% if a track has a strike greater than 6, delete  the tracking
% 	bad_tracks = find(strk_traks > 6)
% 	Cell_estimate(:,bad_tracks) = NaN; 
% 
% 	% clf
% 	img = imread(frame_list(t).name);
% 	imshow(img)
% 	hold on;
% 	plot(Y(t,:), X(t,:),'or');
% 	T = size(Cell_location_estimateX,2);
% 	Ms = [3,5]; % marker sizes
% 	color_list = ['r','b','g','c','m','y'];
% 	for Dc = 1:nF
% 		if ~isnan(Cell_loation_estimateX(t,Dc))
% 			Sz = mod(Dc,2)+1
% 			Cz = mod(Dc,6)+1
% 			if t<21
% 				st = t-1;
% 			else 
% 				st = 19;
% 			end
% 			tmX = Cell_location_estimateY(t-st:t,Dc);
% 			tmy = Cell_location_estimateY(t-st:t,Dc);
% 			plot(tmY,tmX,'.-','markersize?,Ms(Sz)','color',color_list(Cz),'linewidth',3)
% 			axis off
%         		end 
% 	end
% 	pause 
% 	t
% end
% 
% 
% 
% 
% 
% 
% 
