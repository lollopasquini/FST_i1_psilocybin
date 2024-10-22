%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% LEADING EIGENVECTOR DYNAMICS ANALYSIS (LEiDA)
%
% Calculates the static and dynamical measures for resting-state fMRI
% Insight-1 dataset with Schaefer 400
%
% Lorenzo March 2022, based on Jakub Vohryzek November 2021
%
%%
clear all; close all; clc;

addpath(genpath('/Users/lorenzopasquini/Desktop/lollo_backup_august_2020/A_Postdoc/01_Projects/15_Humboldt/INSIGHT_rest_clean/LEiDA/rephaseencodingscript'));
addpath(genpath('/Users/lorenzopasquini/Desktop/lollo_backup_august_2020/A_Postdoc/01_Projects/15_Humboldt/Hopf_consciousness-main'));

%% Decision Tree
analysisType = 'kmeans' % 'statistics' or 'kmeans'

%% data info

subjects={'01','02','03','04','05', '06', '07','08','09', '10', '11', '12', '13','14','15', '16', '18','21','23', '24', '25', '26', '30', '31', '32' ...
          '01','02','03','04','05', '06', '07','08','09', '10', '11', '12', '13','14','15', '16', '18','21','23', '24', '25', '26', '30', '31', '32' ...
          '01','02','03','04','05', '06', '07','08','09', '10', '11', '12', '13','14','15', '16', '18','21','23', '24', '25', '26', '30', '31', '32'};
numSbj = size(subjects,2);
atlasIdx = {'90'};
atlasPath = '/Users/lorenzopasquini/Desktop/lollo_backup_august_2020/A_Postdoc/01_Projects/15_Humboldt/';
cnd = {'pre','mid','post'};
numCnd = size(cnd,2);
grp = {'rsp','nrsp'};
numGrp = size(grp,2);

%% save file
saveFile = '/Users/lorenzopasquini/Desktop/lollo_backup_august_2020/A_Postdoc/01_Projects/15_Humboldt/INSIGHT_rest_clean/LEiDA/';

%% REMISSION

% version Robin response in 5 week (QIDS)
responders             = ones(25,1)';
noresponders           = zeros(25,1)';

respIdx                 = find(responders == 1);
norespIdx               = find(noresponders == 1);

respNum                 = sum(responders);
norespNum               = sum(noresponders);

%% FILTER SETTINGS

TR = 1.25;                 
fnq = 1/(2 * TR);               % Nyquist frequency
flp = 0.04; % 0.01;             % lowpass frequency of filter
fhi = 0.08; % 0.1;              % highpass
Wn = [flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
k = 2;                          % 2nd order butterworth filter
[bfilt,afilt] = butter(k,Wn);   % construct the filter

%% Initialising

keeprois = [7:10 13:16 23:32 41:42 71:78]; % AAL90 ROIs to keep
numAreas = size(keeprois,2);                    % number of brain regions
numTp    = 477;                    % number of timepoints = 477 points
excTp    = 3;                      % number of timepoints to exclude from before and after
numTp    = numTp-2*excTp;          % number of final timepoints = 471 points

%% AAL90 labels
load([atlasPath,'/AAL_labels.mat'])
% To reorder matrix plots
order = [1:2:numAreas numAreas:-2:2]; % AAL is interleaving right and left brain regions

%% %%%%%%%%%%%%%%%%% kmeans to be adjusted accordingly %%%%%%%%%%%%%%%%%%%%
maxK = 20;
rangeK = 1:maxK;
replicates = 20;
distance = 'sqeuclidean' % 'sqeuclidean', 'cosine';
workDate = '03_29_2022'
atlasused = '_aal28_test_RETEST'
mkdir([saveFile,'/Results/',workDate, atlasused])

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% figures %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result_save     = 1; % 0 if no saving results and 1 yes saving results
compute_kmeans  = 1; % 0 if no saving results and 1 yes computing kmeans
save_figure     = 1; % 0 if no saving results and 1 yes saving figurse

%%
% conditions

numTask = 3;                            % Restbefore, Restmid, Restafter

switch analysisType
    case 'kmeans'
%% Part 1. DATA
    load('/Users/lorenzopasquini/Desktop/lollo_backup_august_2020/A_Postdoc/01_Projects/15_Humboldt/INSIGHT_rest_clean/data_all_aal90.mat');
    tcs_all = tcs_all(:,keeprois,:);
    
    Leading_Eig_all.pre      = [];  Leading_Eig_all.mid      = []; Leading_Eig_all.post      = [];
    Leading_Eig_all_idx.pre  = []; Leading_Eig_all_idx.mid  = []; Leading_Eig_all_idx.post  = [];

    % extract the timeseries for each subject
    for c = 1 %:numCnd
            for subj = 1:size(subjects,2)
%                 prefix=['/Volumes/workHD/Datasets/PSILODEP/derivatives/sub-',subjects{subj},'/ses-',cnd{c}];
%                 suffix=['sub-',subjects{subj},'_ses-',cnd{c},'_task-rest_AAL90.mat'];
% 
%                 name = subjects{subj};
%                 disp(['sub-',subjects{subj},'_ses-',cnd{c},'_task-rest'])
%                 file_mat = [prefix '/' suffix];
                XBOLD = cell(2,2,size(subjects,2)); % for the optimisation part
                pre = 0; mid = 0; post = 0;

                BOLD = squeeze(tcs_all(subj,:,:)); %load(file_mat);
                XBOLD{2,2,size(subjects,2)} = BOLD;
                % BOLD signal to BOLD phase using the Hilbert transform
                [BOLD_processed, Phase_BOLD, staticFC] = BOLD2hilbert(BOLD, numAreas, bfilt, afilt, excTp);

                % Kuramoto Order Parameter
                [OP, Synchro, Metasta] = KuramotoOP(Phase_BOLD, numAreas);

                % Instantaneous FC (BOLD Phase Synchrony)
                [iFC, ~, Leading_Eig] = instFC(Phase_BOLD, numAreas, numTp,'default','halfSwitching');

                % FCD
                FCD = squareform(1-pdist(Leading_Eig','cosine'));
                % FCD 2
                Isubdiag = find(tril(ones(numAreas),-1));
                kk = 1;
                for t = 1:numTp-2
                    p1 = squeeze(mean(iFC(:,:,t:t+2),3));
                    p1_pattern = p1(Isubdiag);
                    for t2 = t+1:numTp-2
                        p2 = squeeze(mean(iFC(:,:,t2:t2+2),3));
                        p2_pattern = p2(Isubdiag);
                        FCD2(kk) = dot(p1_pattern,p2_pattern)/norm(p1_pattern)/norm(p2_pattern);
                        kk = kk+1;
                    end
                end

                %% SAVING INTO DATA STRUCTURE

                BOLD_processed_all.([cnd{c}])(subj,:,:) = BOLD_processed;  
                Phase_BOLD_all.([cnd{c}])(subj,:,:)     = Phase_BOLD;
                Synchrony.([cnd{c}])(1,subj,:)          = Synchro;
                Metastability.([cnd{c}])(1,subj,:)      = Metasta;
                staticFC_all.([cnd{c}])(subj,:,:)       = staticFC;
                GBC_all.([cnd{c}])(subj,:)              = sum(squeeze(staticFC));
                FCD_all.([cnd{c}])(subj,:,:)            = FCD;
                FCD2_all.([cnd{c}])(subj,:,:)           = FCD2;
                hist_FCD.([cnd{c}])(subj,:)             = nonzeros(triu(FCD,1));
                Leading_Eig_sbj.([cnd{c}])(subj,:,:)    = Leading_Eig;
                Leading_Eig_all.([cnd{c}])              = cat(2,Leading_Eig_all.([cnd{c}]),Leading_Eig);
                % idx
                Leading_Eig_sbj_idx.([cnd{c}])(subj,:)  = str2num(subjects{subj})*ones(1,numTp);
                Leading_Eig_all_idx.([cnd{c}])          = cat(2,Leading_Eig_all_idx.([cnd{c}]) ,str2num(subjects{subj})*ones(1,numTp));
             end
    end

    if result_save
        save(strcat(saveFile,'/Results/',workDate, atlasused, '/Processed_Data_',distance,'_',num2str(flp),'-',num2str(fhi),'Hz.mat'),...
            'BOLD_processed_all','Phase_BOLD_all','FCD_all','FCD2_all','hist_FCD','Leading_Eig_sbj','Leading_Eig_all',...
            'Leading_Eig_all_idx','staticFC_all','GBC_all','Synchrony','Metastability')
    end

%% %%%%%%%%%%%%%%%%%%% Part 2. CLUSTERS %%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%% KMEANS ALGORITHM %%%
    if compute_kmeans
        % Conditions/Sbjs: pre_run1, pre_run2, post_run1, post_run2
        Leading_Eig_kmeans = Leading_Eig_all.pre; %[Leading_Eig_all.pre_run1 Leading_Eig_all.pre_run2 Leading_Eig_all.post_run1 Leading_Eig_all.post_run2];
        % indexing the different conditions
        Leading_Eig_kmeans_idx_tmp = ones(1,size(Leading_Eig_all.pre,2)); %[ones(1,size(Leading_Eig_all.pre_run1,2)),2*ones(1,size(Leading_Eig_all.pre_run2,2)),...
                                %3*ones(1,size(Leading_Eig_all.post_run1,2)),4*ones(1,size(Leading_Eig_all.post_run2,2))];
        Leading_Eig_kmeans_idx.pre_run1    = find(Leading_Eig_kmeans_idx_tmp == 1);
        Leading_Eig_kmeans_idx.pre_run2    = find(Leading_Eig_kmeans_idx_tmp == 2);
        Leading_Eig_kmeans_idx.post_run1   = find(Leading_Eig_kmeans_idx_tmp == 3);
        Leading_Eig_kmeans_idx.post_run2   = find(Leading_Eig_kmeans_idx_tmp == 4);
        Leading_Eig_kmeans_idx.pre = Leading_Eig_kmeans_idx_tmp;


        Kmeans_results = cell(1,maxK);
        for k = 1:maxK
            clear idx
            %rangeK(k) = k;

            disp(['Calculating for ' num2str(k) 'clusters'])
            [IDX, C, SUMD, D]       = kmeans(Leading_Eig_kmeans',k,'Distance',distance,'Replicates',replicates,'Display','final'); %,'Options',opt);   

            [~, ind_sort]           = sort(hist(IDX,1:rangeK(k)),'descend');
            [~,idx_sort]            = sort(ind_sort,'ascend');
            Kmeans_results{k}.IDX   = idx_sort(IDX);   % Cluster time course - numeric collumn vectors
            Kmeans_results{k}.C     = C(ind_sort,:);       % Cluster centroids (FC patterns)
            Kmeans_results{k}.SUMD  = SUMD(ind_sort); % Within-cluster sums of point-to-centroid distances
            Kmeans_results{k}.D     = D(ind_sort);       % Distance from each point to every centroid   end

            %%
            for c = 1:numCnd
                for r = 1:numGrp
                    for subj = 1:size(subjects,2)
                        tp = Kmeans_results{k}.IDX(Leading_Eig_kmeans_idx.pre); %([cnd{c},'_run',num2str(r)]));
                        Cluster_label{k}.([cnd{c},'_run',num2str(r)])(subj,:) = tp(Leading_Eig_all_idx.pre); %([cnd{c},'_run',num2str(r)]) == str2num(subjects{subj}));
                    end
                end
            end
        end

        if result_save
            save(strcat(saveFile,'/Results/',workDate,atlasused,'/Kmeans_Data_',distance,'_',num2str(flp),'-',num2str(fhi),'Hz.mat'),...
                'Kmeans_results','distance')
        end
    end

    case 'statistics'
    %% Loading
    loadData = strcat(saveFile,'/Results/',workDate,atlasused);
    load(strcat(loadData,'/Kmeans_Data_',distance,'_',num2str(flp),'-',num2str(fhi),'Hz.mat'))
    %%
    for k = 1:maxK
        for c = 1:numCnd
            for r = 1:numGrp
                for subj = 1:size(subjects,2)
                    Ctime = Cluster_label{1,rangeK(k)}.([cnd{c},'_run',num2str(r)])(subj,:);

                    for cl = 1:rangeK(k)
                        % Probability
                        Ctime_sum(cl) = sum(Ctime == cl);
                        PO(cl) = mean(Ctime == cl) + eps; % adding eps to avoid dividing by 0 in the porbability transition matriix

                        % Mean Lifetime
                        Ctime_bin      = Ctime == cl;

                        % Detect switches in and out of this state
                        param_c1       = find(diff(Ctime_bin)==1);
                        param_c2       = find(diff(Ctime_bin)==-1);

                        % We discard the cases where state starts or ends ON
                        if length(param_c2)>length(param_c1)
                            param_c2(1)   = [];
                        elseif length(param_c1)>length(param_c2)
                            param_c1(end) = [];
                        elseif  ~isempty(param_c1) && ~isempty(param_c2) && param_c1(1)>param_c2(1)
                            param_c2(1)   = [];
                            param_c1(end) = [];
                        end
                        if ~isempty(param_c1) && ~isempty(param_c2)
                            C_durations   = param_c2-param_c1;
                        else
                            C_durations   = 0;
                        end
                        LT(c) = mean(C_durations)*TR; % lifetimes
                    end
                    % Probability of Transition
                    transferMatrix = zeros(k,k);
                    for tp = 2:size(Ctime,2)
                        transferMatrix(Ctime(tp-1),Ctime(tp)) = transferMatrix(Ctime(tp-1),Ctime(tp)) + 1;
                    end
                    PT    = transferMatrix./(size(Ctime,2)-1);                       % normalised by T-1
                    PTnorm = squeeze(PT)./squeeze(PO)';     % normalised by Probability of Ocupancy

                    Ctime_all{k}.([cnd{c},'_run',num2str(r)])(subj,:) = Ctime;
                    Ctime_sum_all{k}.([cnd{c},'_run',num2str(r)])(subj,:) = Ctime_sum;
                    PO_all{k}.([cnd{c},'_run',num2str(r)])(subj,:) = PO;
                    LT_all{k}.([cnd{c},'_run',num2str(r)])(subj,:) = LT;
                    PT_all{k}.([cnd{c},'_run',num2str(r)])(subj,:,:) = PT;
                    PTnorm_all{k}.([cnd{c},'_run',num2str(r)])(subj,:,:) = PTnorm;
                end
            end
        end
    end   

    if result_save
        save(strcat(saveFile,'/Results/',workDate,atlasused,'/Measures_Data_',distance,'_',num2str(flp),'-',num2str(fhi),'HZ.mat'),...
            'PO_all','LT_all','PT_all','PTnorm_all')
    end
end


for i =1:75
    for nk=1:3
        myFO(i,nk) = sum(Kmeans_results{3}.IDX((1+(471*(i-1))):471*i)==nk)/471;
        my_trans(i,nk) = sum(diff(Kmeans_results{3}.IDX((1+(471*(i-1))):471*i))~=0);
    end
end