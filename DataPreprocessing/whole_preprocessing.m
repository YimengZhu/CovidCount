clc;clear all;
load('/home/prak12-2/CovidCount/mall_dataset/mall_gt.mat'); %load the ground truth
img_path = '/home/prak12-2/CovidCount/mall_dataset/frames/';
output_path = '/home/prak12-2/CovidCount/mall_dataset/den_maps/';
num_images = 2000;

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
    match = [img_path, ".jpg"];
    output_name = erase(input_img_name, match);
    density_map = uint8(density_map * 255.0);
    imwrite(density_map, [output_path output_name '.bmp'])
    %csvwrite([output_path output_name '.csv'],density_map);
end
