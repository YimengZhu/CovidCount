clc;clear all;
load('/home/hongyi/py_ws/CovidCount/mall_dataset/mall_gt.mat'); %load the ground truth
img_path = '/home/hongyi/py_ws/CovidCount/mall_dataset/frames/';
output_path = '/home/hongyi/py_ws/CovidCount/mall_dataset/density_maps/';
output_img_path = '/home/hongyi/py_ws/CovidCount/mall_dataset/cropped_images/';
num_images = 2000;
N = 9 ; %randomly generate 9 patches from each original image
for idx = 1:num_images
    if (mod(idx,10)==0) 
        fprintf(1,'Processing %3d/%d files\n', idx,num_images);
    end
    if (idx<=9)
       input_img_name = strcat(img_path,'seq_00000',num2str(idx),'.jpg');
    elseif (idx>=10 && idx<=99)
       input_img_name = strcat(img_path,'seq_0000',num2str(idx),'.jpg');
    elseif (idx>=100 && idx<=999)
        input_img_name = strcat(img_path,'seq_000', num2str(idx),'.jpg');
    elseif (idx>=1000 && idx<=2000)
        input_img_name = strcat(img_path,'seq_00',num2str(idx),'.jpg');
    else
        print('idx out of range');continue;
    end
    input_image = imread(input_img_name);
    XY = frame{idx}.loc;
    [h,w,c] = size(input_image);
    if(c==3)
        input_image = rgb2gray(input_image);
    end
    img_perspective = zeros(h,w);
    for y=1:h
        img_perspective(y,:) = 0.1243*y +24.49;
    end
    density_map = get_density_map_gaussian(input_image, XY, img_perspective);
    
    % crop the original images and density maps into patches. 
    wn2 = w/8; hn2= h/8;
    wn2 = 8*floor(wn2/8);
    hn2 = 8*floor(hn2/8);
    % restrict the region of center of patches.
    a_w = wn2+1; b_w = w-wn2;
    a_h = hn2+1; b_h = h-hn2;
    
    for i = 1:N
        x = floor((b_w - a_w)*rand +a_w);
        y = floor((b_h - a_h)*rand +a_h);
        x1 = x-wn2; y1 = y-hn2;
        x2 = x+wn2-1; y2 = y+hn2-1;
       
        im_sampled = input_image(y1:y2, x1:x2);
        density_sampled = density_map(y1:y2, x1:x2);
        annPoints_sampled = XY(XY(:,1)>x1& ...
            XY(:,1)<x2 & ...
            XY(:,2)>y1 & ...
            XY(:,2)<y2,:);
        annPoints_sampled(:,1) = annPoints_sampled(:,1)-x1;
        annPoints_sampled(:,2) = annPoints_sampled(:,2)-y1;
        img_index = strcat(num2str(idx),'_',num2str(i));
        im_sampled = imresize(im_sampled,[72,72]);
        density_sampled = uint8(255 * imresize(density_sampled, [18,18], 'bilinear')); %normalize
        
        imwrite(im_sampled, [output_img_path num2str(img_index) '.jpg']);
        
        imwrite(density_sampled, [output_path num2str(img_index) '.bmp']);

        
    end
end
    
