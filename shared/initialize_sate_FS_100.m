function initialize_sate_FS_100()
%% Quality Index Blocks
global Qblocks_size  bicubic flag_cut_bounds dim_cut thvalues printEPS ratio L;
global  folder_pan_noR folder_rgb_noR folder_nir_noR ;
global  file_path_pan_noR file_path_rgb_noR file_path_nir_noR ;
global savepath_F savepath_up savepath_rgb savepath_up_rgb;
global data count time ;
global sensor sate file_path_gt256; 
% global folder_nir1_256 folder_nir2_256   folder_RedEdge256  folder_yellow256  folder_coastal256
global file_path_nir1_256 file_path_nir2_256 file_path_RedEdge256 file_path_coastal256 file_path_yellow256
switch sate
    case {'wv2','wv'}
        sensor = 'WV2';
    case 'wv3'
        sensor = 'WV3';
    case 'wv3-8'
        sensor = 'WV3';
    case 'wv3_8'
        sensor = 'WV3';
    case 'ik'
        sensor = 'IKONOS';
    case 'qb'
        sensor = 'QB';
    case 'geo'
        sensor = 'GeoEye1';
    otherwise
        sensor = 'none';
end

Qblocks_size = 32;
bicubic = 0;   %% Interpolator
flag_cut_bounds = 1; %% Cut Final Image
dim_cut = 11;
thvalues = 0;  %% Threshold values out of dynamic range
printEPS = 0;  %% Print Eps
ratio = 4;     %% Resize Factor
L = 11;        %% Radiometric Resolution