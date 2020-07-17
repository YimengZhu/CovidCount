function im_density = get_density_map_gaussian(im,points,Mp) 

[hi,wi,c]=size(im);
im_density = zeros([hi,wi]); %create an empty map with size of input img
[h,w] = size(im_density);% h stands for height, w stands for width

if(size(points)==0)
    return;
end

if(length(points(:,1))==1) %if there contains only one annotated point in img, set it to 255
    x1 = max(1,min(w,round(points(1,1)))); %if the point is in the corner,set to (1,1)
    y1 = max(1,min(h,round(points(1,2))));
    im_density(y1,x1) = 255; 
    return;
end
for j = 1:length(points) 	
    f_sz = 15;
    m_x = max(1,min(w,round(points(j,1))));
    m_y = max(1,min(h,round(points(j,2))));
    sigma = Mp(m_y,m_x);
    H = fspecial('Gaussian',[f_sz, f_sz],sigma);  %fspecial ---- a gussian filter with window size f_sz and sigma
    x = min(w,max(1,abs(int32(floor(points(j,1)))))); 
    y = min(h,max(1,abs(int32(floor(points(j,2))))));
    if(x > w || y > h)
        continue; % the point is out of the image, skip over it and go to next loop
    end
    x1 = x - int32(floor(f_sz/2)); y1 = y - int32(floor(f_sz/2)); % left_down corner
    x2 = x + int32(floor(f_sz/2)); y2 = y + int32(floor(f_sz/2)); % right_up corner
    dfx1 = 0; dfy1 = 0; dfx2 = 0; dfy2 = 0;  % deal with the Situation that bounding-box at corner
    change_H = false;
    if(x1 < 1)
        dfx1 = abs(x1)+1;
        x1 = 1;
        change_H = true;
    end
    if(y1 < 1)
        dfy1 = abs(y1)+1;
        y1 = 1;
        change_H = true;
    end
    if(x2 > w)
        dfx2 = x2 - w;
        x2 = w;
        change_H = true;
    end
    if(y2 > h)
        dfy2 = y2 - h;
        y2 = h;
        change_H = true;
    end
    x1h = 1+dfx1; y1h = 1+dfy1; x2h = f_sz - dfx2; y2h = f_sz - dfy2; %new windows
    if (change_H == true)
        H =  fspecial('Gaussian',[double(y2h-y1h+1), double(x2h-x1h+1)],sigma); %[] describe the size of filter window
    end
    im_density(y1:y2,x1:x2) = im_density(y1:y2,x1:x2) +  H; 
     
end

end