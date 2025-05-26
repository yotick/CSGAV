clc; clear all; close all;
%% only for plaidas
currentPath = pwd;
sharepath = strcat(currentPath,'/shared');
addpath(genpath(sharepath));

global Qblocks_size   flag_cut_bounds dim_cut thvalues  ratio L sate;

sate = 'Houston';%Botswana,Houston,Pavia
method = 'CSGAV';
%CSGAV
curr_d = pwd;
initialize_sate_RS_MTF();

path_fused = strcat(currentPath,'\fused\',sate,'\',sate,'\',method);   % need to change
path_gt =  strcat(currentPath,'\fused\',sate,'\',sate,'gt','\',method);   % need to change

files_fused = dir(fullfile(path_fused,'*.mat'));
files_gt = dir(fullfile(path_gt,'*.mat'));     
count = length(files_fused);
count = 10;
num = count;
time = zeros(1,count); 
for  i= 1:count
    i
    %     %     count=count+1
    cd(curr_d);   
    path = fullfile(path_fused,files_fused(i).name);
    im = load(path);
    im_F = im.fused;
    path = fullfile(path_gt,files_gt(i).name);
    gt = load(path);
    im_gt = gt.fused;



    minValue = min(im_gt(:));
    maxValue = max(im_gt(:));

% 3. 归一化到0到255的范围
%     im_gt = 255 * (im_gt - minValue) / (maxValue - minValue);
    [Q_avg_Segm, SAM_Segm, ERGAS_Segm, SCC_GT_Segm, Q_Segm] = indexes_evaluation(im_F,im_gt, ...
   ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    
    [peaksnr] = psnr(im_F,im_gt);
    peaksnr
    data{i,1}=files_fused(i).name;
    data{i,2}=roundn(peaksnr,-4);
    data{i,3}=roundn(CC4(im_gt,im_F),-4);
    data{i,4}=roundn(UIQI4(im_gt,im_F),-4);
    data{i,5}=roundn(RASE(im_gt,im_F),-4);
%     data{i,6}=roundn(RMSE(im2double(im_gt),im2double(im_F)),-4);
    data{i,6}=roundn(RMSE(im_gt,im_F),-4);
    data{i,7}=roundn(SAM4(im_gt,im_F),-4);
    data{i,8}=roundn(ERGAS(im_gt,im_F),-4);
    
    data{i,9}=roundn(Q_avg_Segm,-4);
    data{i,10}=roundn(SAM_Segm,-4);
    data{i,11}=roundn(ERGAS_Segm,-4);
    data{i,12}=roundn(SCC_GT_Segm,-4);
    data{i,13}=roundn(Q_Segm,-4);
    data{i,14} = roundn(time(i),-2);
end
j = num +1;
data{j,1}='average';
data{j,2}=roundn(mean(cell2mat(data(1:j-1, 2))),-4);
data{j,3}=roundn(mean(cell2mat(data(1:j-1, 3))),-4);
data{j,4}=roundn(mean(cell2mat(data(1:j-1, 4))),-4);
data{j,5}=roundn(mean(cell2mat(data(1:j-1, 5))),-4);
data{j,6}=roundn(mean(cell2mat(data(1:j-1, 6))),-4);
data{j,7}=roundn(mean(cell2mat(data(1:j-1, 7))),-4);
data{j,8}=roundn(mean(cell2mat(data(1:j-1, 8))),-4);
data{j,9}=roundn(mean(cell2mat(data(1:j-1, 9))),-4);
data{j,10}=roundn(mean(cell2mat(data(1:j-1,10))),-4);
data{j,11}=roundn(mean(cell2mat(data(1:j-1,11))),-4);
data{j,12}=roundn(mean(cell2mat(data(1:j-1,12))),-4);
data{j,13}=roundn(mean(cell2mat(data(1:j-1,13))),-4);
data{j,14}=roundn(mean(cell2mat(data(1:j-1,14))),-2);

T = cell2table(data,'VariableNames',{'FILENAME' 'PSNR' 'CC' 'UIQI' 'RASE' 'RMSE' 'SAM' 'ERGAS'...
     'Q_UIQI' 'SAM2'  'ERGAS2' 'SCC' 'Q2n' 'time'});
cd(curr_d);
currentDateTime = clock;
hour = num2str(currentDateTime(4));
minute = num2str(currentDateTime(5));
path = strcat('final_result/',sate);
if ~exist(path, 'dir')
    % 如果文件夹不存在，则创建它
    mkdir(path);
end
writetable(T,strcat(path,'/',method,hour,minute,'.xlsx'),'WriteRowNames',true);



