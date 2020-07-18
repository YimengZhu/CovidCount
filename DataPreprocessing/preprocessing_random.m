clc;clear all;
load('/home/hongyi/py_ws/CovidCount/mall_dataset/mall_gt.mat'); %load the ground truth
img_path = '/home/hongyi/py_ws/CovidCount/mall_dataset/frames/';
output_img_path = '/home/hongyi/py_ws/CovidCount/mall_dataset/cropped_images/';
output_density_path = '/home/hongyi/py_ws/CovidCount/mall_dataset/density_maps/';
num_images = 2000;
count = zeros([8000,1]);
random_images = 8; 
img_index = 0;
count = zeros([2000*random_images,1]);

for i = 1:num_images

    if (mod(i,10)==0) 
        fprintf(1,'Processing %3d/%d files\n', i,num_images);
    end
    if (i<=9)
       input_img_name = strcat(img_path,'seq_00000',num2str(i),'.jpg');
    elseif (i>=10 && i<=99)
       input_img_name = strcat(img_path,'seq_0000',num2str(i),'.jpg');
    elseif (i>=100 && i<=999)
        input_img_name = strcat(img_path,'seq_000', num2str(i),'.jpg');
    elseif (i>=1000 && i<=2000)
        input_img_name = strcat(img_path,'seq_00',num2str(i),'.jpg');
    else
        print('idx out of range');continue;
    end
    
    img = imread(input_img_name);
    XY = frame{i}.loc;
    [w,h,c] = size(img);
    
    img_perspective = zeros(h,w);
    for y=1:240
        img_perspective(y,:) = 0.1243*y +24.49;
    end


    % crop the original image into 4 images, (xi0,yi0), (xi1,yi1) indicate the range of cropped images 
    % 1,2,3,4 sepearte the reigions in counter clockwise
    
    for j=1:random_images
        
        xc = floor(1/4*w + 1/2*rand*w);
        yc = floor(1/4*h + 1/2*rand*h);
        x1 = xc - 1/4*w + 1; x2 = xc + 1/4*w;
        y1 = yc - 1/4*h + 1; y2 = yc + 1/4*h;

        im_sampled = img(x1:x2, y1:y2, :);
        annPoints_sampled = XY(XY(:,1)>y1& ...
        XY(:,1)<y2 & ...
        XY(:,2)>x1 & ...
        XY(:,2)<x2,:);
        [count_sampled,dim] = size(annPoints_sampled);
        annPoints_sampled(:,1) = annPoints_sampled(:,1)-y1;
        annPoints_sampled(:,2) = annPoints_sampled(:,2)-x1;

        density_map = 255.0*get_density_map_gaussian(im_sampled, annPoints_sampled, img_perspective);
        
        img_index = img_index + 1;
        
        if(img_index<10)
            str_idx = strcat('0000', num2str(img_index));
        elseif(img_index>=10 && img_index<=99)
            str_idx = strcat('000', num2str(img_index));
        elseif(img_index>=100 && img_index<=999)
            str_idx = strcat('00', num2str(img_index));
        elseif(img_index>=1000 && img_index<=9999)
            str_idx = strcat('0', num2str(img_index));
        else
            str_idx = num2str(img_index);
        end
        
        
        imwrite(im_sampled, [output_img_path str_idx '.jpg']);
        
        imwrite(density_map, [output_density_path str_idx '.bmp']);
        count(img_index)=count_sampled;
        
        
    end

end


   
    
