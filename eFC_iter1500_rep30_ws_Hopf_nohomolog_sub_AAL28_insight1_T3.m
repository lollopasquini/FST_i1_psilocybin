%% Generating Jakub's measures for Hopf Model on insight1 dataset
% lorenzo pasquini, April 2022

clear all; 
close all; 
clc;

addpath(genpath('/Users/lorenzopasquini/Desktop/lollo_backup_august_2020/A_Postdoc/01_Projects/15_Humboldt/INSIGHT_rest_clean/LEiDA/rephaseencodingscript'));
addpath(genpath('/Users/lorenzopasquini/Desktop/lollo_backup_august_2020/A_Postdoc/01_Projects/15_Humboldt/Hopf_consciousness-main'));

%%  Read the empirical T2 data 
load('/Users/lorenzopasquini/Desktop/lollo_backup_august_2020/A_Postdoc/01_Projects/15_Humboldt/INSIGHT_rest_clean/data_all_aal90.mat');
tcs_t2 = tcs_all(51:75,:,:); % T2, post 1mg Psilocybin
keeprois = [7:10 13:16 23:32 41:42 71:78]; % AAL90 ROIs to keep
tcs_t2 = tcs_t2(:,keeprois,:); % Only FTS regions

% Variables
load('/Users/lorenzopasquini/Desktop/lollo_backup_august_2020/A_Postdoc/01_Projects/15_Humboldt/INSIGHT_rest_clean/hopf_scripts/relatentspace/DataSleepW_N3.mat', 'SC'); % your data here, you need ts + structural connectivity
SC = SC(keeprois, keeprois);
load('/Users/lorenzopasquini/Desktop/lollo_backup_august_2020/A_Postdoc/01_Projects/15_Humboldt/INSIGHT_rest_clean/hopf_scripts/hopf_freq_AAL28_insight1_T2.mat') % the output of COmpute_Hopf_freq_AAL90
C = SC/max(max(SC))*0.2;

N = length(keeprois);
NPARCELLS = N;
NSUB = size(tcs_t2,1); % sumber of subjects
NSUBSIM = NSUB; 
indexsub = 1:NSUB;
Isubdiag = find(tril(ones(N),-1));

keeprois = [7:10 13:16 23:32 41:42 71:78]; % AAL90 ROIs to keep
numAreas = size(keeprois,2); 

Tmax = size(squeeze(tcs_t2(1,:,:)),2); 
numTp = size(squeeze(tcs_t2(1,:,:)),2); % number of timepoints = 477 points
excTp = 3;                      % number of timepoints to exclude from before and after
numTp = numTp-2*excTp;          % number of final timepoints = 471 points

for nsub = indexsub
    tsdata(:,:,nsub) = squeeze(tcs_t2(nsub,:,1:Tmax));
    FCdata(nsub,:,:) = corrcoef(squeeze(tsdata(:,:,nsub))');
end

FC_emp = squeeze(mean(FCdata,1));

FCemp2 = FC_emp-FC_emp.*eye(N);
GBCemp = mean(FCemp2,2);

Isubdiag = find(tril(ones(N),-1));

TR = 1.25;
%%%%%%%%%%%%%% filter for simulated signa "MEGstyle"

flp = 0.04;           % lowpass frequency of filter
fhi = 0.08;           % highpass
delt = TR;            % sampling interval
k = 2;                  % 2nd order butterworth filter
fnq = 1/(2*delt);       % Nyquist frequency
Wn = [flp/fnq fhi/fnq]; % butterworth bandpass non-dimensional frequency
[bfilt,afilt] = butter(k,Wn);   % construct the filter

%%%%%%%%%%%%%%
%%

kk = 1;
for nsub = 1:NSUB
    nsub
        
    BOLD = (squeeze(tsdata(:,:,nsub)));
    
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
    Synchrony(nsub,:) = Synchro;
    Metastability(nsub,:) = Metasta;
    staticFC_all(nsub,:,:) = staticFC;
    GBC_all(nsub,:) = sum(squeeze(staticFC));
    FCD_all(nsub,:,:) = FCD;
    FCD2_all(nsub,:,:) = FCD2;
    hist_FCD(nsub,:)  = nonzeros(triu(FCD,1));
end

FCphasesemp =squeeze(mean(staticFC_all));
clear iFC
% FC_empfilt = squeeze(nanmean(FCdata_filt,1)); 
% FCemp2filt = FC_empfilt-FC_empfilt.*eye(N);
% GBCempfilt = mean(FCemp2filt,2);
% Metaemp_av = nanmean(Metaemp);

%% the model
% fixed parameter
TR = 1.25;
% %%%%%%%%%%%%%%

flp = 0.04;           % lowpass frequency of filter
fhi = 0.08;           % highpass
delt = TR;            % sampling interval
k = 2;                  % 2nd order butterworth filter
fnq = 1/(2*delt);       % Nyquist frequency
Wn = [flp/fnq fhi/fnq]; % butterworth bandpass non-dimensional frequency
[bfilt2,afilt2] = butter(k,Wn);   % construct the filter

fcsimul = zeros(NSUB,NPARCELLS,NPARCELLS);

Tmax = size(squeeze(tcs_t2(1,:,:)),2);
omega1 = repmat(2*pi*f_diff',1,2); omega1(:,1) = -omega1(:,1);
dt = 0.1*TR/2;
sig = 0.04;
dsig = sqrt(dt)*sig;

unil = 1;
for sim=1:30
    sim
for s = 1 % number of iteration for coupling G
    Gs(s) = 1*s
    
    G = Gs(s);
    
    wC = G*C;
    sumC = repmat(sum(wC,2),1,2); % for sum Cij*xj
    
    omega = omega1;
    
    %%%%%%%%%%%%%%%%%%
    Cnew=C;
    a = -0.02*ones(NPARCELLS,2);
    rng(unil);
    for iter=1:1500
        wC = G*Cnew;
        sumC = repmat(sum(wC,2),1,2); % for sum Cij*xj
        xs=zeros(Tmax,N);
        %number of iterations, 100 willk�hrlich, weil reicht in diesem Fall
        z = 0.1*ones(N,2); % --> x = z(:,1), y = z(:,2)
        nn=0;
        % discard first 3000 time steps
        for t=0:dt:3000
            suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
            zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
            z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma) + dsig*randn(N,2);
        end
        % actual modeling (x=BOLD signal (Interpretation), y some other oscillation)
        for t=0:dt:((Tmax-1)*TR)
            suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
            zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
            z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma) + dsig*randn(N,2);
            if abs(mod(t,TR))<0.01
                nn=nn+1;
                xs(nn,:)=z(:,1)';
            end
        end
        
        %%%%
        BOLD=xs';
        signal_filt=zeros(N,nn);
        Phase_BOLD=zeros(N,nn);
        for seed=1:N
            BOLD(seed,:)=demean(detrend(BOLD(seed,:)));
            signal_filt(seed,:) =filtfilt(bfilt2,afilt2,BOLD(seed,:));
            Phase_BOLD(seed,:) = angle(hilbert(signal_filt(seed,:)));
        end

        for t=1:nn
            for n=1:N
                for p=1:N
                    iFC(t,n,p)=cos(Phase_BOLD(n,t)-Phase_BOLD(p,t));
                end
            end
        end
        FCphases=squeeze(mean(iFC));

        %% update effective conn matrix Cnew
        for i=1:N
            for j=i+1:N
                if (C(i,j)>0)
                    Cnew(i,j)=Cnew(i,j)+0.01*(FCphasesemp(i,j)-FCphases(i,j));
                    if Cnew(i,j)<0
                        Cnew(i,j)=0;
                    end
                    Cnew(j,i)=Cnew(i,j);
                end
            end
        end
        
        Cnew=Cnew/max(max(Cnew))*0.2;
        
        D = abs(FCphasesemp-FCphases).^2;
        MSE(sim, iter) = sum(D(:))/numel(FCphases);
        %if MSE<0.01
        %    break;
        %end
        
        %%%%
        
    end
    
    Coptim{s}(sim,:,:)=Cnew;  %% effective Conn for G (we)
    
    %%%%%%%%%%%%%%%%%%
    
    kk = 1;
    for sub=1:NSUBSIM % Number of permutation, here equal to number of subjects
        sub
        rng(unil);
        %% Hopf Simulation
        a = -0.02*ones(NPARCELLS,2);
        
        xs = zeros(Tmax,NPARCELLS);
        %number of iterations, 100 willk�hrlich, weil reicht in diesem Fall
        z = 0.1*ones(NPARCELLS,2); % --> x = z(:,1), y = z(:,2)
        nn = 0;
        
        % discard first 2000 time steps
        for t = 0:dt:2000
            suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
            zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
            z = z + dt*(a.*z + zz.*omega - (z).*(z.*z+zz.*zz) + suma) + dsig*randn(NPARCELLS,2);
        end
        
        % actual modeling (x=BOLD signal (Interpretation), y some other oscillation)
        for t = 0:dt:((Tmax-1)*TR)
            suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
            zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
            z = z + dt*(a.*z + zz.*omega - (z).*(z.*z+zz.*zz) + suma) + dsig*randn(NPARCELLS,2);
            if abs(mod(t,TR))<0.01
                nn = nn+1;
                xs(nn,:) = z(:,1)';
            end
        end
        ts = xs';
        clear signal_filt
        
        BOLD_sim = ts;
        all_sim_ts(:,:,sub, s) = ts;
        
        % BOLD signal to BOLD phase using the Hilbert transform
        [BOLD_sim_processed, Phase_sim_BOLD, staticFC_sim] = BOLD2hilbert(BOLD_sim, numAreas, bfilt, afilt, excTp);
        
        % Kuramoto Order Parameter
        [OP_sim, Synchro_sim, Metasta_sim] = KuramotoOP(Phase_sim_BOLD, numAreas);
        
        % Instantaneous FC (BOLD Phase Synchrony)
        [iFC_sim, ~, Leading_Eig_sim] = instFC(Phase_sim_BOLD, numAreas, numTp,'default','halfSwitching');
        
        % FCD
        FCD_sim = squareform(1-pdist(Leading_Eig_sim','cosine'));
        % FCD 2
        Isubdiag = find(tril(ones(numAreas),-1));
        kk = 1;
        for t = 1:numTp-2
            p1_sim = squeeze(mean(iFC_sim(:,:,t:t+2),3));
            p1_pattern_sim = p1_sim(Isubdiag);
            for t2 = t+1:numTp-2
                p2_sim = squeeze(mean(iFC_sim(:,:,t2:t2+2),3));
                p2_pattern_sim = p2_sim(Isubdiag);
                FCD2_sim(kk) = dot(p1_pattern_sim, p2_pattern_sim)/norm(p1_pattern_sim)/norm(p2_pattern_sim);
                kk = kk+1;
            end
        end
        Synchrony_sim(sub,:) = Synchro_sim;
        Metastability_sim(sub,:) = Metasta_sim;
        staticFC_sim_all(sub,:,:) = staticFC_sim;
        GBC_sim_all(sub,:) = sum(squeeze(staticFC_sim));
        FCD_sim_all(sub,:,:) = FCD_sim;
        FCD2_sim_all(sub,:,:) = FCD2_sim;
        hist_FCD_sim(sub,:)  = nonzeros(triu(FCD_sim,1));
        
       unil = unil + 1; 
    end

    %% Compare metics 
    FC_simul = squeeze(mean(staticFC_sim_all,1));
    FC_empfilt = squeeze(mean(staticFC_all,1));
    cc = corrcoef(atanh(FC_empfilt(Isubdiag)),atanh(FC_simul(Isubdiag)));
    FCfitt(sim, s) = cc(2); %% FC fitting

    FCsim2 = FC_simul-FC_simul.*eye(N);
    GBCsim = nanmean(FCsim2,2);
    FC2 = FC_empfilt-FC_empfilt.*eye(N);

    GBCempfilt = nanmean(FC2,2);
    GBCfitt1(sim, s) = corr2(GBCempfilt,GBCsim);
    GBCfitt2(sim, s) = sqrt(nanmean((GBCempfilt-GBCsim).^2));
    
    [hh, pp, Synchfitt(sim, s)] = kstest2(Synchrony, Synchrony_sim);  %% Synchrony fitting
    [hh, pp, FCDfitt(sim, s)] = kstest2(reshape(hist_FCD, [], 1), reshape(hist_FCD_sim, [], 1));  %% FCD fitting
end
end
kk = 1

cd('/Users/lorenzopasquini/Desktop/lollo_backup_august_2020/A_Postdoc/01_Projects/15_Humboldt/INSIGHT_rest_clean/hopf_scripts/hopf_eFC/T3_30_rep_fixedG');
save(sprintf('contrano_optimize_fixedG_rep_30_insight1_t2_JakubMeasures_%03d.mat',kk),'GBCfitt1','GBCfitt2','FCfitt', 'FCDfitt','Synchfitt','all_sim_ts', 'Coptim', 'Gs', 'MSE');

% figure;
% subplot(2,2,1)
% plot(FCfitt(1:length(Gs)));
% xticks([1 10:10:length(Gs)]);
% xticklabels(Gs([1 10:10:length(Gs)]));
% ylabel('Corr');
% title('FCfitt');
% xlabel('G');
% yline(max(FCfitt(1:length(Gs))),'--red');
% xline(find(FCfitt(1:length(Gs))==max(FCfitt(1:length(Gs)))),'--red');
% %
% subplot(2,2,2)
% plot(GBCfitt1(1:length(Gs)));
% xticks([1 10:10:length(Gs)]);
% xticklabels(Gs([1 10:10:length(Gs)]));
% title('GBCfitt1');
% ylabel('fit');
% xlabel('G');
% yline(min(GBCfitt1(1:length(Gs))),'--red');
% xline(find(GBCfitt1(1:length(Gs))==min(GBCfitt1(1:length(Gs)))),'--red');
% %
% subplot(2,2,3)
% plot(FCDfitt(1:length(Gs)));
% xticks([1 10:10:length(Gs)]);
% xticklabels(Gs([1 10:10:length(Gs)]));
% title('FCDfitt');
% ylabel('fit');
% xlabel('G');
% yline(min(FCDfitt(1:length(Gs))),'--red');
% xline(find(FCDfitt(1:length(Gs))==min(FCDfitt(1:length(Gs)))),'--red');
% %
% subplot(2,2,4)
% plot(Synchfitt(1:length(Gs)));
% xticks([1 10:10:length(Gs)]);
% xticklabels(Gs([1 10:10:length(Gs)]));
% title('Synchrony');
% ylabel('fit');
% xlabel('G');
% yline(min(Synchfitt(1:length(Gs))),'--red');
% xline(find(Synchfitt(1:length(Gs))==min(Synchfitt(1:length(Gs)))),'--red');

% figure
% for np = 1:20
%     subplot(5,4,np)
%     imagesc(squeeze(Coptim(np,:,:)));
%     colorbar;
% end
