%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File to create training and validation set       %
% for ShanghaiTech Dataset Part A and B. 10% of    %
% the training set is set aside for validation     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all;
seed = 95461354;
rng(seed)
N = 9;
dataset = 'A';
dataset_name = ['shanghaitech_part_' dataset '_patches_' num2str(N)];
path = ['/home/hongyi/py_ws/CovidCount/sht_dataset/ShanghaiTech/part_' dataset '/train_data/images/'];
output_path = '/home/hongyi/py_ws/CovidCount/sht_dataset/formatted_trainval/';
train_path_img = strcat(output_path, dataset_name,'/train/');
train_path_den = strcat(output_path, dataset_name,'/train_den/');
val_path_img = strcat(output_path, dataset_name,'/val/');
val_path_den = strcat(output_path, dataset_name,'/val_den/');
gt_path = ['/home/hongyi/py_ws/CovidCount/sht_dataset/ShanghaiTech/part_' dataset '/train_data/ground-truth/'];

mkdir(output_path)
mkdir(train_path_img);
mkdir(train_path_den);
mkdir(val_path_img);
mkdir(val_path_den);
img_index = 0;

if (dataset == 'A')
    num_images = 300;
else
    num_images = 400;
end
num_val = ceil(num_images*0.1);
indices = randperm(num_images);

for idx = 1:num_images
    i = indices(idx);
    if (mod(idx,10)==0)
        fprintf(1,'Processing %3d/%d files\n', idx, num_images);
    end
    load(strcat(gt_path, 'GT_IMG_',num2str(i),'.mat')) ;
    input_img_name = strcat(path,'IMG_',num2str(i),'.jpg');
    im = imread(input_img_name);
    [h, w, c] = size(im);
    if (c == 3)
        im = rgb2gray(im); %first different thing, use gray scale image for inference
    end
    
    %wn2 = w/8; hn2 = h/8;
    %wn2 =8 * floor(wn2/8);
    %hn2 =8 * floor(hn2/8);
    
    wn2 = 160; hn2 = 120;
    
    annPoints =  image_info{1}.location;
    if( w <= 2*wn2 )
        im = imresize(im,[ h,2*wn2+2]);
        annPoints(:,1) = annPoints(:,1)*2*wn2/w;
    end
    if( h <= 2*hn2)
        im = imresize(im,[2*hn2+2,w]);
        annPoints(:,2) = annPoints(:,2)*2*hn2/h;
    end
    [h, w, c] = size(im);
    %a_w = wn2+1; b_w = w - wn2;
    %a_h = hn2+1; b_h = h - hn2;
    
    a_w = 160 + 1; b_w = w - 160 -1;
    a_h = 120 + 1; b_h = h - 120 -1;
    
    im_density = get_density_map_gaussian(im,annPoints);
    for j = 1:N
        
        img_index = img_index + 1;
        
        x = floor((b_w - a_w) * rand + a_w);
        y = floor((b_h - a_h) * rand + a_h);
        x1 = x - wn2; y1 = y - hn2;
        x2 = x + wn2-1; y2 = y + hn2-1;
        
        
        im_sampled = im(y1:y2, x1:x2,:);
        im_density_sampled = im_density(y1:y2,x1:x2);
        
        annPoints_sampled = annPoints(annPoints(:,1)>x1 & ...
            annPoints(:,1) < x2 & ...
            annPoints(:,2) > y1 & ...
            annPoints(:,2) < y2,:);
        annPoints_sampled(:,1) = annPoints_sampled(:,1) - x1;
        annPoints_sampled(:,2) = annPoints_sampled(:,2) - y1;
        % img_idx = strcat(num2str(i), '_',num2str(j));      
        
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
        
        imwrite(im_sampled, [val_path_img str_idx '.jpg']);
        imwrite(im_density_sampled, [val_path_den str_idx '.bmp']);
        
    end
    
end

