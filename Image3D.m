clear;
load mri
addpath('./3D_images/');
for time = 1 : 1 : 20
    if time < 10
        path = sprintf('%s%d%s', './3D_images/t=0', time,'_*.jpg');
    else
        path = sprintf('%s%d%s', './3D_images/t=', time,'_*.jpg');
    end
    frame_list1 = dir(path);

    for slice = 1 : length(frame_list1)
        thisSlice = double(imread(frame_list1(slice).name));
        if slice == 1
            image3D = thisSlice;
        else
            image3D = cat(3, image3D, thisSlice);
        end
    end
    
    figure
    contourslice(image3D,[],[],[1:1:100],8);
    view(3);
    axis tight
    image3Ds = smooth3(image3D);
    hiso = patch(isosurface(image3Ds,5),'FaceColor','black','EdgeColor','none');
    isonormals(image3Ds,hiso)
    hcap = patch(isocaps(image3D,5),'FaceColor','interp','EdgeColor','none');
    view(35,30) 
    axis equal
    pause
end
