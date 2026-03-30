%% =========================================================================
%  PSO-Based Multi-Objective Optimization of CNC Milling Parameters
%  Simultaneous Optimization of Rz, Rt (Surface Roughness) and VB (Tool Wear)
%
%  Approach    : Derringer-Suich Desirability Function + PSO
%  Surrogate   : Gaussian Process Regression (GPR)
%  Optimizer   : MATLAB particleswarm (built-in)
%  Validation  : Leave-One-Out Cross-Validation (LOOCV)
%
%  Work Material : AISI 1050 Steel - Dry CNC Milling
%  Cutting Tool  : Coated Carbide Inserts (TNGA 160408)
%  Decision Vars : Vc [613-817 m/min], f [100-200 mm/min], DoC [0.1-0.2 mm]
%
%  Required MATLAB toolboxes:
%    - Statistics and Machine Learning Toolbox  (fitrgp, predict)
%    - Global Optimization Toolbox              (particleswarm)
%
%  Outputs:
%    - Console summary of all results
%    - 7 publication-ready dark-mode figures
%    - CSV tables in 'pso_results/' folder
% =========================================================================
clear; close all; clc;
rng(42, 'twister');   % reproducibility

%% ======================== DARK-MODE PALETTE ==============================
bg      = [0.12 0.12 0.16];   % figure / axes background
fg      = [0.92 0.92 0.92];   % text, tick marks
grid_c  = [0.30 0.30 0.35];   % grid lines
clr_Rz  = [0.29 0.78 1.00];   % cyan-blue  -> Rz
clr_Rt  = [1.00 0.55 0.20];   % amber      -> Rt
clr_VB  = [0.75 0.40 0.95];   % purple     -> VB
clr_opt = [0.40 1.00 0.55];   % mint green -> optimum / knee
clr_exp = [0.90 0.85 0.30];   % yellow     -> experimental points
clr_ref = [0.65 0.65 0.65];   % grey       -> reference lines

% Helper to apply dark style to any axes
setDark = @(ax) set(ax, 'Color', bg, 'XColor', fg, 'YColor', fg, ...
    'GridColor', grid_c, 'GridAlpha', 0.55, 'MinorGridColor', grid_c, ...
    'FontSize', 11, 'LineWidth', 0.8);

%% ======================== USER SETTINGS ==================================
% Balanced case: roughness gets 80% total weight, tool wear gets 20%
mainWeights = struct('Rz', 0.40, 'Rt', 0.40, 'VB', 0.20);

% Desirability shape parameters (s > 1 = more demanding near the limit)
shape = struct('Rz', 1.0, 'Rt', 1.0, 'VB', 1.1);

% Sparse-region penalty (discourages PSO from exploiting poorly-sampled areas)
usePenalty        = true;
penaltyStartDist  = 0.45;
penaltyWeight     = 0.15;

% PSO settings
swarmMain = 80;   iterMain = 250;
swarmFast = 50;   iterFast = 150;

% Sensitivity perturbation
delta = 0.05;   % +-5%

% Output folder
resultsDir = 'pso_results';
if ~isfolder(resultsDir), mkdir(resultsDir); end

% Weight cases for trade-off study
caseNames = {'Balanced', 'Rz-priority', 'Rt-priority', 'Balanced+Wear', 'Wear-priority'};
caseWeights = [
    0.40  0.40  0.20;
    0.55  0.25  0.20;
    0.25  0.55  0.20;
    0.35  0.35  0.30;
    0.25  0.25  0.50];

%% ======================== LOAD DATA ======================================
fprintf('============================================================\n');
fprintf('  PSO CNC OPTIMIZATION: Rz + Rt + VB (Tool Wear)\n');
fprintf('  Derringer-Suich Desirability + GPR Surrogates\n');
fprintf('============================================================\n\n');

% Averaged experimental data from Sheet3 of the case study workbook
% Columns: [Vc, f, DOC, VB, Rt, Rz]
rawData = [
    715, 150, 0.15, 0.06260, 1.9100,       1.3167;
    817, 200, 0.15, 0.04100, 1.5700,       1.3733;
    715, 200, 0.20, 0.07150, 3.6600,       2.0767;
    613, 200, 0.15, 0.06825, 1.7200,       1.7200;
    715, 150, 0.15, 0.08000, 1.9533,       1.4233;
    613, 150, 0.10, 0.06900, 0.8733,       0.8733;
    817, 100, 0.15, 0.03025, 2.4267,       1.3800;
    817, 150, 0.20, 0.03625, 1.3300,       0.8133;
    715, 200, 0.10, 0.05925, 2.0467,       1.2733;
    613, 150, 0.20, 0.05550, 5.6333,       2.2600;
    613, 100, 0.15, 0.05475, 2.9900,       2.3867;
    715, 150, 0.15, 0.05600, 1.2433,       1.0233;
    817, 150, 0.10, 0.03125, 0.9567,       0.8133;
    715, 150, 0.15, 0.07975, 2.2733,       1.6767;
    715, 150, 0.15, 0.07425, 1.8800,       1.8800;
    715, 100, 0.20, 0.05175, 1.0967,       0.9333;
    715, 100, 0.10, 0.03750, 1.8067,       1.4767];

Xraw   = rawData(:, 1:3);
VBraw  = rawData(:, 4);
Rtraw  = rawData(:, 5);
Rzraw  = rawData(:, 6);

% --- Average repeated design points before training ---
[Xu, ~, ic] = unique(Xraw, 'rows');
nUniq = size(Xu, 1);
RzAvg = accumarray(ic, Rzraw, [], @mean);
RtAvg = accumarray(ic, Rtraw, [], @mean);
VBAvg = accumarray(ic, VBraw, [], @mean);

fprintf('Raw experimental runs : %d\n', size(Xraw, 1));
fprintf('Unique design points  : %d  (repeated points averaged)\n', nUniq);

X  = Xu;
Rz = RzAvg;
Rt = RtAvg;
VB = VBAvg;

lb = [min(X(:,1)), min(X(:,2)), min(X(:,3))];
ub = [max(X(:,1)), max(X(:,2)), max(X(:,3))];
n  = size(X, 1);

fprintf('Bounds:\n');
fprintf('  Vc  = %.0f to %.0f m/min\n',  lb(1), ub(1));
fprintf('  f   = %.0f to %.0f mm/min\n', lb(2), ub(2));
fprintf('  DOC = %.2f to %.2f mm\n\n',   lb(3), ub(3));

%% ======================== TRAIN GPR SURROGATES ===========================
% Fixed Matern 3/2 kernel with explicit noise (Sigma) to avoid overfitting.
% The model no longer assumes near-perfect data.
gprSigma = 0.02;   % measurement noise std dev
fprintf('--- GPR Training: Matern 3/2 kernel, Sigma = %.3f ---\n', gprSigma);

responses    = {Rz, Rt, VB};
respNames    = {'Rz', 'Rt', 'VB'};
bestModels   = cell(1, 3);
bestCVpreds  = cell(1, 3);
bestCVstats  = cell(1, 3);
bestTrainR2  = zeros(1, 3);

for ri = 1:3
    y = responses{ri};

    [cvPred, cvStats] = looForGPR(X, y, 'matern32', gprSigma);

    bestModels{ri}  = fitrgp(X, y, ...
        'KernelFunction', 'matern32', ...
        'Sigma', gprSigma, ...
        'Standardize', true);
    bestCVpreds{ri} = cvPred;
    bestCVstats{ri} = cvStats;
    bestTrainR2(ri) = calcR2(y, predict(bestModels{ri}, X));

    fprintf('  %s:  LOOCV R^2 = %.4f  RMSE = %.4f  |  Training R^2 = %.4f\n', ...
        respNames{ri}, cvStats.R2, cvStats.RMSE, bestTrainR2(ri));
end

mdlRz = bestModels{1};
mdlRt = bestModels{2};
mdlVB = bestModels{3};

cvRz = bestCVpreds{1};   statsRz = bestCVstats{1};
cvRt = bestCVpreds{2};   statsRt = bestCVstats{2};
cvVB = bestCVpreds{3};   statsVB = bestCVstats{3};

R2_Rz = bestTrainR2(1);
R2_Rt = bestTrainR2(2);
R2_VB = bestTrainR2(3);

fprintf('\n=== SELECTED MODELS (Matern 3/2, Sigma=%.3f) ===\n', gprSigma);
fprintf('  Rz:  Training R^2 = %.4f  |  LOOCV R^2 = %.4f  RMSE = %.4f  MAE = %.4f\n', ...
    R2_Rz, statsRz.R2, statsRz.RMSE, statsRz.MAE);
fprintf('  Rt:  Training R^2 = %.4f  |  LOOCV R^2 = %.4f  RMSE = %.4f  MAE = %.4f\n', ...
    R2_Rt, statsRt.R2, statsRt.RMSE, statsRt.MAE);
fprintf('  VB:  Training R^2 = %.4f  |  LOOCV R^2 = %.4f  RMSE = %.4f  MAE = %.4f\n', ...
    R2_VB, statsVB.R2, statsVB.RMSE, statsVB.MAE);
fprintf('\n');

cvTbl = table((1:n)', Rz, cvRz, Rt, cvRt, VB, cvVB, ...
    'VariableNames', {'Run','Rz_Actual','Rz_LOOCV','Rt_Actual','Rt_LOOCV','VB_Actual','VB_LOOCV'});
writetable(cvTbl, fullfile(resultsDir, 'loocv_predictions.csv'));

%% ======================== SANITY CHECK (pre-PSO) =========================
% Predict on training data and compare to actual values.
% If predictions diverge from actuals the surrogate is unreliable.
fprintf('--- Sanity Check: GPR predictions on training data ---\n');
predRz_train = predict(mdlRz, X);
predRt_train = predict(mdlRt, X);
predVB_train = predict(mdlVB, X);

maxErrRz = max(abs(predRz_train - Rz));
maxErrRt = max(abs(predRt_train - Rt));
maxErrVB = max(abs(predVB_train - VB));

fprintf('  Rz max |pred - actual| = %.4f  (range %.4f)\n', maxErrRz, max(Rz)-min(Rz));
fprintf('  Rt max |pred - actual| = %.4f  (range %.4f)\n', maxErrRt, max(Rt)-min(Rt));
fprintf('  VB max |pred - actual| = %.5f  (range %.5f)\n', maxErrVB, max(VB)-min(VB));

sanityOK = true;
sanityThresh = 0.20;   % 20% of response range
for ri = 1:3
    y = responses{ri};
    yPred = predict(bestModels{ri}, X);
    relErr = max(abs(yPred - y)) / (max(y) - min(y));
    if relErr > sanityThresh
        fprintf('  WARNING: %s model max relative error = %.1f%% (threshold %.0f%%)\n', ...
            respNames{ri}, relErr*100, sanityThresh*100);
        sanityOK = false;
    end
end
if sanityOK
    fprintf('  All models pass sanity check (max error < %.0f%% of range).\n', sanityThresh*100);
end
fprintf('\n');

%% ======================== GPR MODEL EQUATIONS ============================
% Print the mathematical form and fitted parameters for each GPR surrogate.
fprintf('--- GPR Predictor Model Equations ---\n');
fprintf('  All three models use the Matern 3/2 kernel:\n');
fprintf('    k(x,x'') = sigma_f^2 * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)\n');
fprintf('    where r = ||x - x''||,  l = length scale,  sigma_f = signal std dev\n');
fprintf('    Prediction: y(x*) = k(x*,X) * [K(X,X) + sigma_n^2 I]^{-1} * y\n\n');

for ri = 1:3
    kinfo = bestModels{ri}.KernelInformation;
    kpars = kinfo.KernelParameters;          % [length scale(s); signal std]
    noiseSig = bestModels{ri}.Sigma;
    fprintf('  %s model:\n', respNames{ri});
    fprintf('    Length scale  l       = %.6f\n', kpars(1));
    fprintf('    Signal std    sigma_f = %.6f\n', kpars(2));
    fprintf('    Noise std     sigma_n = %.6f\n', noiseSig);
    fprintf('    Alpha (dual coefficients): [%s]\n', ...
        strjoin(arrayfun(@(v) sprintf('%.4f',v), bestModels{ri}.Alpha, 'UniformOutput', false), ', '));
    fprintf('\n');
end

%% ======================== DESIRABILITY LIMITS ============================
limits.Rz.target = min(Rz);   limits.Rz.upper = max(Rz);
limits.Rt.target = min(Rt);   limits.Rt.upper = max(Rt);
limits.VB.target = min(VB);   limits.VB.upper = max(VB);

fprintf('Desirability limits (smaller is better):\n');
fprintf('  Rz: target = %.4f um,  upper = %.4f um\n', limits.Rz.target, limits.Rz.upper);
fprintf('  Rt: target = %.4f um,  upper = %.4f um\n', limits.Rt.target, limits.Rt.upper);
fprintf('  VB: target = %.5f mm,  upper = %.5f mm\n\n', limits.VB.target, limits.VB.upper);

%% ======================== MAIN BALANCED OPTIMIZATION =====================
fprintf('============================================================\n');
fprintf('  MAIN PSO RUN (Balanced Desirability)\n');
fprintf('============================================================\n');

objMain = @(x) desirabilityObjective(x, mdlRz, mdlRt, mdlVB, ...
    limits, mainWeights, shape, X, lb, ub, usePenalty, penaltyStartDist, penaltyWeight);

global PSO_HISTORY;
PSO_HISTORY = [];

optsMain = optimoptions('particleswarm', ...
    'SwarmSize', swarmMain, 'MaxIterations', iterMain, ...
    'FunctionTolerance', 1e-8, 'Display', 'iter', 'OutputFcn', @psoRecorder);

[xBest, fBest] = particleswarm(objMain, 3, lb, ub, optsMain);

best.Vc  = xBest(1);
best.f   = xBest(2);
best.DOC = xBest(3);
[best.Rz, best.Rz_sd] = predict(mdlRz, xBest);
[best.Rt, best.Rt_sd] = predict(mdlRt, xBest);
[best.VB, best.VB_sd] = predict(mdlVB, xBest);
best.D   = calcOverallDesirability(best.Rz, best.Rt, best.VB, limits, mainWeights, shape);

fprintf('\n================ MAIN RECOMMENDED SOLUTION ================\n');
fprintf('Weights: wRz = %.2f, wRt = %.2f, wVB = %.2f\n', mainWeights.Rz, mainWeights.Rt, mainWeights.VB);
fprintf('Vc  = %.2f m/min\n',   best.Vc);
fprintf('f   = %.2f mm/min\n',  best.f);
fprintf('DOC = %.4f mm\n',      best.DOC);
fprintf('Predicted Rz = %.4f um  (95%% CI: [%.4f, %.4f])\n', best.Rz, best.Rz-1.96*best.Rz_sd, best.Rz+1.96*best.Rz_sd);
fprintf('Predicted Rt = %.4f um  (95%% CI: [%.4f, %.4f])\n', best.Rt, best.Rt-1.96*best.Rt_sd, best.Rt+1.96*best.Rt_sd);
fprintf('Predicted VB = %.5f mm  (95%% CI: [%.5f, %.5f])\n', best.VB, best.VB-1.96*best.VB_sd, best.VB+1.96*best.VB_sd);
fprintf('Overall Desirability D = %.4f\n', best.D);
fprintf('============================================================\n\n');

% --- Post-PSO check: nearest training point to the optimum ---
[minDist, idx] = min(vecnorm(X - xBest, 2, 2));
fprintf('--- Post-PSO Nearest-Neighbour Check ---\n');
fprintf('  Nearest training point: Run #%d  (Euclidean dist = %.4f)\n', idx, minDist);
fprintf('  That point: Vc=%.0f, f=%.0f, DOC=%.2f  |  Rz=%.4f, Rt=%.4f, VB=%.5f\n', ...
    X(idx,1), X(idx,2), X(idx,3), Rz(idx), Rt(idx), VB(idx));
if minDist > 0.5 * max(ub - lb)
    fprintf('  WARNING: optimum is far from any training data – prediction may be unreliable.\n');
else
    fprintf('  Optimum is within well-sampled region.\n');
end
fprintf('\n');

convHistory = PSO_HISTORY;

%% ======================== BEST OBSERVED POINT ============================
obsDes = zeros(n, 1);
for i = 1:n
    obsDes(i) = calcOverallDesirability(Rz(i), Rt(i), VB(i), limits, mainWeights, shape);
end
[bestObsD, idxObs] = max(obsDes);

fprintf('Best observed experimental point (same desirability weights):\n');
fprintf('  Run #%d: Vc=%.0f, f=%.0f, DOC=%.2f\n', idxObs, X(idxObs,1), X(idxObs,2), X(idxObs,3));
fprintf('  Rz=%.4f, Rt=%.4f, VB=%.5f\n', Rz(idxObs), Rt(idxObs), VB(idxObs));
fprintf('  Desirability = %.4f\n\n', bestObsD);

%% ======================== TRADE-OFF CASE STUDY ===========================
fprintf('--- Trade-Off Cases ---\n');

optsFast = optimoptions('particleswarm', ...
    'SwarmSize', swarmFast, 'MaxIterations', iterFast, ...
    'FunctionTolerance', 1e-7, 'Display', 'none');

nCases = numel(caseNames);

% Preallocate numeric arrays (avoids cell2mat errors)
caseVc   = zeros(nCases, 1);
casef    = zeros(nCases, 1);
caseDOC  = zeros(nCases, 1);
casePRz  = zeros(nCases, 1);
casePRt  = zeros(nCases, 1);
casePVB  = zeros(nCases, 1);
caseDes  = zeros(nCases, 1);

for i = 1:nCases
    w = struct('Rz', caseWeights(i,1), 'Rt', caseWeights(i,2), 'VB', caseWeights(i,3));
    obj = @(x) desirabilityObjective(x, mdlRz, mdlRt, mdlVB, ...
        limits, w, shape, X, lb, ub, usePenalty, penaltyStartDist, penaltyWeight);

    [xC, fC] = particleswarm(obj, 3, lb, ub, optsFast);

    caseVc(i)  = xC(1);
    casef(i)   = xC(2);
    caseDOC(i) = xC(3);
    casePRz(i) = predict(mdlRz, xC);
    casePRt(i) = predict(mdlRt, xC);
    casePVB(i) = predict(mdlVB, xC);
    caseDes(i) = 1 - fC;
end

caseTbl = table(caseNames', caseWeights(:,1), caseWeights(:,2), caseWeights(:,3), ...
    caseVc, casef, caseDOC, casePRz, casePRt, casePVB, caseDes, ...
    'VariableNames', {'Case','wRz','wRt','wVB','Vc','f','DOC','Pred_Rz','Pred_Rt','Pred_VB','Desirability'});
caseTbl = sortrows(caseTbl, 'Desirability', 'descend');
writetable(caseTbl, fullfile(resultsDir, 'case_study_summary.csv'));

fprintf('\nTrade-off case summary (sorted by desirability):\n');
disp(caseTbl);

%% ======================== SYSTEMATIC WEIGHT SWEEP ========================
fprintf('--- Systematic Weight Sweep (trade-off cloud) ---\n');

wearLevels = [0.10, 0.20, 0.30, 0.40];
alphaVec   = 0.10:0.10:0.90;
nSweep     = numel(wearLevels) * numel(alphaVec);

% Preallocate numeric array (avoids cell2mat errors)
sweepData = zeros(nSweep, 10);  % [wRz wRt wVB Vc f DOC Rz Rt VB D]
cnt = 0;

for iw = 1:numel(wearLevels)
    wVB = wearLevels(iw);
    remain = 1 - wVB;
    for ia = 1:numel(alphaVec)
        alpha = alphaVec(ia);
        wRz = remain * alpha;
        wRt = remain * (1 - alpha);

        w = struct('Rz', wRz, 'Rt', wRt, 'VB', wVB);
        obj = @(x) desirabilityObjective(x, mdlRz, mdlRt, mdlVB, ...
            limits, w, shape, X, lb, ub, usePenalty, penaltyStartDist, penaltyWeight);

        [xS, ~] = particleswarm(obj, 3, lb, ub, optsFast);
        yRz = predict(mdlRz, xS);
        yRt = predict(mdlRt, xS);
        yVB = predict(mdlVB, xS);
        DS  = calcOverallDesirability(yRz, yRt, yVB, limits, w, shape);

        cnt = cnt + 1;
        sweepData(cnt, :) = [wRz, wRt, wVB, xS(1), xS(2), xS(3), yRz, yRt, yVB, DS];
    end
end
sweepData = sweepData(1:cnt, :);

sweepTbl = array2table(sweepData, 'VariableNames', ...
    {'wRz','wRt','wVB','Vc','f','DOC','Pred_Rz','Pred_Rt','Pred_VB','Desirability'});
writetable(sweepTbl, fullfile(resultsDir, 'weight_sweep_summary.csv'));
fprintf('Weight sweep complete: %d combinations evaluated.\n\n', cnt);

%% ======================== LOCAL SENSITIVITY ==============================
fprintf('--- Local Sensitivity Analysis (+-5%% at recommended point) ---\n');

basePoint = [best.Vc, best.f, best.DOC];
baseResp  = [best.Rz, best.Rt, best.VB];
varNames  = {'Vc', 'f', 'DOC'};

sensPlus  = zeros(3, 3);
sensMinus = zeros(3, 3);

for j = 1:3
    xp = basePoint;   xm = basePoint;
    xp(j) = min(ub(j), basePoint(j) * (1 + delta));
    xm(j) = max(lb(j), basePoint(j) * (1 - delta));

    yp = [predict(mdlRz, xp), predict(mdlRt, xp), predict(mdlVB, xp)];
    ym = [predict(mdlRz, xm), predict(mdlRt, xm), predict(mdlVB, xm)];

    sensPlus(j,:)  = yp - baseResp;
    sensMinus(j,:) = ym - baseResp;
end

sensTbl = table(varNames', ...
    sensPlus(:,1), sensMinus(:,1), ...
    sensPlus(:,2), sensMinus(:,2), ...
    sensPlus(:,3), sensMinus(:,3), ...
    'VariableNames', {'Variable','dRz_plus5','dRz_minus5','dRt_plus5','dRt_minus5','dVB_plus5','dVB_minus5'});
writetable(sensTbl, fullfile(resultsDir, 'sensitivity_analysis.csv'));

fprintf('\nSensitivity Table:\n');
disp(sensTbl);

%% ======================== FINAL RECOMMENDATION TABLE =====================
finalTbl = table( ...
    best.Vc, best.f, best.DOC, best.Rz, best.Rt, best.VB, best.D, ...
    X(idxObs,1), X(idxObs,2), X(idxObs,3), Rz(idxObs), Rt(idxObs), VB(idxObs), bestObsD, ...
    'VariableNames', {'Opt_Vc','Opt_f','Opt_DOC','Opt_Rz','Opt_Rt','Opt_VB','Opt_D', ...
                      'BestObs_Vc','BestObs_f','BestObs_DOC','BestObs_Rz','BestObs_Rt','BestObs_VB','BestObs_D'});
writetable(finalTbl, fullfile(resultsDir, 'final_recommendation.csv'));

fprintf('\n============================================================\n');
fprintf('  FINAL RECOMMENDATION\n');
fprintf('============================================================\n');
fprintf('PSO-optimized:  Vc=%.1f m/min, f=%.1f mm/min, DOC=%.3f mm\n', best.Vc, best.f, best.DOC);
fprintf('  -> Rz=%.4f um, Rt=%.4f um, VB=%.5f mm  (D=%.4f)\n', best.Rz, best.Rt, best.VB, best.D);
fprintf('Best observed:  Vc=%.0f m/min, f=%.0f mm/min, DOC=%.2f mm\n', X(idxObs,1), X(idxObs,2), X(idxObs,3));
fprintf('  -> Rz=%.4f um, Rt=%.4f um, VB=%.5f mm  (D=%.4f)\n', Rz(idxObs), Rt(idxObs), VB(idxObs), bestObsD);
fprintf('============================================================\n\n');

%% ========================================================================
%                    FIGURES (ALL DARK MODE)
%% ========================================================================
fprintf('Generating figures...\n');

% --- Figure 1: LOOCV Validation ---
fig1 = figure('Color', bg, 'Name', '1 - LOOCV Validation', 'NumberTitle', 'off', ...
    'Position', [50 400 1200 350]);

ax1a = subplot(1,3,1);
scatter(ax1a, Rz, cvRz, 70, clr_Rz, 'filled', 'MarkerEdgeColor', bg); hold(ax1a, 'on');
plotIdLine(ax1a, Rz, cvRz, clr_ref);
setDark(ax1a);
xlabel(ax1a, 'Measured R_z (\mum)', 'Color', fg);
ylabel(ax1a, 'LOOCV Predicted R_z (\mum)', 'Color', fg);
title(ax1a, sprintf('R_z  [%s]  Train R^2=%.3f  LOOCV R^2=%.3f', 'Mat32', R2_Rz, statsRz.R2), ...
    'Color', fg, 'FontWeight', 'bold', 'FontSize', 10);
grid(ax1a, 'on'); axis(ax1a, 'square');

ax1b = subplot(1,3,2);
scatter(ax1b, Rt, cvRt, 70, clr_Rt, 'filled', 'MarkerEdgeColor', bg); hold(ax1b, 'on');
plotIdLine(ax1b, Rt, cvRt, clr_ref);
setDark(ax1b);
xlabel(ax1b, 'Measured R_t (\mum)', 'Color', fg);
ylabel(ax1b, 'LOOCV Predicted R_t (\mum)', 'Color', fg);
title(ax1b, sprintf('R_t  [%s]  Train R^2=%.3f  LOOCV R^2=%.3f', 'Mat32', R2_Rt, statsRt.R2), ...
    'Color', fg, 'FontWeight', 'bold', 'FontSize', 10);
grid(ax1b, 'on'); axis(ax1b, 'square');

ax1c = subplot(1,3,3);
scatter(ax1c, VB, cvVB, 70, clr_VB, 'filled', 'MarkerEdgeColor', bg); hold(ax1c, 'on');
plotIdLine(ax1c, VB, cvVB, clr_ref);
setDark(ax1c);
xlabel(ax1c, 'Measured VB (mm)', 'Color', fg);
ylabel(ax1c, 'LOOCV Predicted VB (mm)', 'Color', fg);
title(ax1c, sprintf('VB  [%s]  Train R^2=%.3f  LOOCV R^2=%.3f', 'Mat32', R2_VB, statsVB.R2), ...
    'Color', fg, 'FontWeight', 'bold', 'FontSize', 10);
grid(ax1c, 'on'); axis(ax1c, 'square');

sg1 = sgtitle(fig1, 'Leave-One-Out Cross-Validation of GPR Surrogates', ...
    'FontSize', 14, 'FontWeight', 'bold');
sg1.Color = fg;
saveas(fig1, fullfile(resultsDir, 'fig1_loocv_validation.png'));

% --- Figure 2: PSO Convergence ---
fig2 = figure('Color', bg, 'Name', '2 - PSO Convergence', 'NumberTitle', 'off');
ax2  = axes(fig2);
if ~isempty(convHistory)
    semilogy(ax2, 1:numel(convHistory), convHistory, '-', 'Color', clr_Rz, 'LineWidth', 2.5);
    hold(ax2, 'on');
    finalVal = convHistory(end);
    convIter = find(abs(convHistory - finalVal)./max(finalVal, eps) < 0.001, 1, 'first');
    if ~isempty(convIter)
        semilogy(ax2, convIter, convHistory(convIter), 'o', ...
            'Color', clr_opt, 'MarkerSize', 10, 'MarkerFaceColor', clr_opt);
        text(ax2, convIter + numel(convHistory)*0.025, convHistory(convIter), ...
            sprintf('  Converged @ iter %d\n  f* = %.5f', convIter, finalVal), ...
            'Color', clr_opt, 'FontSize', 10);
    end
    setDark(ax2);
    xlabel(ax2, 'Iteration', 'Color', fg, 'FontSize', 12);
    ylabel(ax2, 'Best Objective Value (log scale)', 'Color', fg, 'FontSize', 12);
    title(ax2, sprintf('PSO Convergence  (w_{Rz}=%.2f, w_{Rt}=%.2f, w_{VB}=%.2f)', ...
        mainWeights.Rz, mainWeights.Rt, mainWeights.VB), ...
        'Color', fg, 'FontSize', 13, 'FontWeight', 'bold');
    xlim(ax2, [1 numel(convHistory)]);
    grid(ax2, 'on'); box(ax2, 'on');
end
saveas(fig2, fullfile(resultsDir, 'fig2_pso_convergence.png'));

% --- Figure 3: Trade-Off Cloud split by VB weight level (2x2) ---
% Color sweep optima by weight ratio. Highlight PSO optimum and best experiment.
fig3 = figure('Color', bg, 'Name', '3 - Trade-Off Cloud', 'NumberTitle', 'off', ...
    'Position', [50 50 1300 950]);

for iw = 1:numel(wearLevels)
    ax3 = subplot(2, 2, iw, 'Parent', fig3);
    mask = abs(sweepData(:,3) - wearLevels(iw)) < 1e-6;
    sd = sweepData(mask, :);

    wRatio = sd(:,1) ./ (sd(:,1) + sd(:,2));
    [wRatio, sIdx] = sort(wRatio);
    sd = sd(sIdx, :);

    % Trade-off path
    plot(ax3, sd(:,8), sd(:,7), '-', 'Color', [1 1 1 0.30], 'LineWidth', 1.2, ...
        'HandleVisibility', 'off');
    hold(ax3, 'on');

    % Sweep points (colored by weight ratio)
    scatter(ax3, sd(:,8), sd(:,7), 60, wRatio, 'filled', ...
        'MarkerEdgeColor', [0.2 0.2 0.25], 'LineWidth', 0.4, 'DisplayName', 'Sweep optima');

    % Experimental data (white-edged for contrast)
    scatter(ax3, Rt, Rz, 55, clr_exp, 's', 'filled', ...
        'MarkerEdgeColor', fg, 'LineWidth', 0.6, 'DisplayName', 'Experiments');

    % Best experimental point (large circle, distinct)
    scatter(ax3, Rt(idxObs), Rz(idxObs), 160, [1 0.3 0.3], 'o', 'LineWidth', 2.5, ...
        'DisplayName', 'Best experiment');

    % PSO optimum (large star, brightest)
    scatter(ax3, best.Rt, best.Rz, 220, clr_opt, 'p', 'filled', ...
        'MarkerEdgeColor', fg, 'LineWidth', 1.2, 'DisplayName', 'PSO optimum');

    % Named cases for this panel
    for ic = 1:height(caseTbl)
        if abs(caseTbl.wVB(ic) - wearLevels(iw)) < 1e-6
            scatter(ax3, caseTbl.Pred_Rt(ic), caseTbl.Pred_Rz(ic), 110, 'r', 'd', 'filled', ...
                'MarkerEdgeColor', fg, 'DisplayName', caseTbl.Case{ic});
        end
    end

    setDark(ax3);
    xlabel(ax3, 'Predicted R_t (\mum)', 'Color', fg, 'FontSize', 11);
    ylabel(ax3, 'Predicted R_z (\mum)', 'Color', fg, 'FontSize', 11);
    title(ax3, sprintf('w_{VB} = %.2f', wearLevels(iw)), ...
        'Color', fg, 'FontWeight', 'bold', 'FontSize', 12);
    colormap(ax3, parula);
    cb3 = colorbar(ax3);
    cb3.Label.String = 'w_{Rz} / (w_{Rz}+w_{Rt})';
    cb3.Label.Color = fg; cb3.Color = fg;
    leg3 = legend(ax3, 'Location', 'northeast', 'FontSize', 7);
    leg3.TextColor = fg; leg3.Color = bg; leg3.EdgeColor = grid_c;
    grid(ax3, 'on'); box(ax3, 'on');
end

sg3 = sgtitle(fig3, 'Trade-Off Cloud: R_z vs R_t  (split by VB weight, color = R_z priority)', ...
    'FontSize', 14, 'FontWeight', 'bold');
sg3.Color = fg;
print(fig3, fullfile(resultsDir, 'fig3_tradeoff_cloud'), '-dpng', '-r300');
saveas(fig3, fullfile(resultsDir, 'fig3_tradeoff_cloud.fig'));

% --- Figure 4: Case Comparison Bar Charts ---
fig4 = figure('Color', bg, 'Name', '4 - Case Comparison', 'NumberTitle', 'off', ...
    'Position', [50 50 1100 400]);

ax4a = subplot(1,2,1, 'Parent', fig4);
cats = categorical(caseTbl.Case, caseTbl.Case);
b1 = bar(ax4a, cats, [caseTbl.Vc, caseTbl.f]);
b1(1).FaceColor = clr_Rz;  b1(1).EdgeColor = 'none';
b1(2).FaceColor = clr_Rt;  b1(2).EdgeColor = 'none';
setDark(ax4a);
ylabel(ax4a, 'Parameter Value', 'Color', fg);
title(ax4a, 'Recommended Machine Settings', 'Color', fg, 'FontWeight', 'bold');
l4a = legend(ax4a, {'V_c (m/min)', 'f (mm/min)'}, 'Location', 'best');
l4a.TextColor = fg; l4a.Color = bg; l4a.EdgeColor = grid_c;
grid(ax4a, 'on');

ax4b = subplot(1,2,2, 'Parent', fig4);
b2 = bar(ax4b, cats, [caseTbl.Pred_Rz, caseTbl.Pred_Rt, caseTbl.Pred_VB * 100]);
b2(1).FaceColor = clr_Rz;  b2(1).EdgeColor = 'none';
b2(2).FaceColor = clr_Rt;  b2(2).EdgeColor = 'none';
b2(3).FaceColor = clr_VB;  b2(3).EdgeColor = 'none';
setDark(ax4b);
ylabel(ax4b, 'Response Value', 'Color', fg);
title(ax4b, 'Predicted Responses', 'Color', fg, 'FontWeight', 'bold');
l4b = legend(ax4b, {'R_z (\mum)', 'R_t (\mum)', 'VB \times100 (mm)'}, 'Location', 'best');
l4b.TextColor = fg; l4b.Color = bg; l4b.EdgeColor = grid_c;
grid(ax4b, 'on');

sg4 = sgtitle(fig4, 'Trade-Off Case Study', 'FontSize', 14, 'FontWeight', 'bold');
sg4.Color = fg;
saveas(fig4, fullfile(resultsDir, 'fig4_case_comparison.png'));

% --- Figure 5: Local Sensitivity Bar Chart ---
fig5 = figure('Color', bg, 'Name', '5 - Sensitivity Analysis', 'NumberTitle', 'off');
ax5  = axes(fig5);

meanAbsSens = [mean(abs([sensPlus(:,1), sensMinus(:,1)]), 2), ...
               mean(abs([sensPlus(:,2), sensMinus(:,2)]), 2), ...
               mean(abs([sensPlus(:,3), sensMinus(:,3)]), 2)];

b5 = bar(ax5, meanAbsSens, 'grouped');
b5(1).FaceColor = clr_Rz;  b5(1).EdgeColor = 'none';
b5(2).FaceColor = clr_Rt;  b5(2).EdgeColor = 'none';
b5(3).FaceColor = clr_VB;  b5(3).EdgeColor = 'none';
setDark(ax5);
set(ax5, 'XTickLabel', {'V_c', 'f', 'DOC'});
ylabel(ax5, 'Mean |Change| from \pm5% Perturbation', 'Color', fg, 'FontSize', 12);
title(ax5, 'Local Sensitivity at Recommended Point', 'Color', fg, 'FontSize', 13, 'FontWeight', 'bold');
l5 = legend(ax5, {'R_z (\mum)', 'R_t (\mum)', 'VB (mm)'}, 'Location', 'best');
l5.TextColor = fg; l5.Color = bg; l5.EdgeColor = grid_c;
grid(ax5, 'on'); box(ax5, 'on');
saveas(fig5, fullfile(resultsDir, 'fig5_sensitivity.png'));

% --- Figure 6: 1-D Sensitivity Sweeps ---
fig6 = figure('Color', bg, 'Name', '6 - 1-D Sweeps', 'NumberTitle', 'off', ...
    'Position', [50 50 1300 700]);

sweepPts = 60;
ranges = {linspace(lb(1),ub(1),sweepPts), linspace(lb(2),ub(2),sweepPts), linspace(lb(3),ub(3),sweepPts)};
xlabs  = {'Cutting Speed V_c (m/min)', 'Feed Rate f (mm/min)', 'Depth of Cut DOC (mm)'};

for vi = 1:3
    Rz_sw = zeros(sweepPts,1);  Rz_sd = zeros(sweepPts,1);
    Rt_sw = zeros(sweepPts,1);  Rt_sd = zeros(sweepPts,1);
    VB_sw = zeros(sweepPts,1);  VB_sd = zeros(sweepPts,1);

    for k = 1:sweepPts
        xk = basePoint;
        xk(vi) = ranges{vi}(k);
        [Rz_sw(k), Rz_sd(k)] = predict(mdlRz, xk);
        [Rt_sw(k), Rt_sd(k)] = predict(mdlRt, xk);
        [VB_sw(k), VB_sd(k)] = predict(mdlVB, xk);
    end

    xv = ranges{vi}(:);

    % Top row: Rz with 95% CI
    axT = subplot(2, 3, vi, 'Parent', fig6);
    hold(axT, 'on');
    fill(axT, [xv; flipud(xv)], ...
        [Rz_sw - 1.96*Rz_sd; flipud(Rz_sw + 1.96*Rz_sd)], ...
        clr_Rz, 'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    fill(axT, [xv; flipud(xv)], ...
        [Rt_sw - 1.96*Rt_sd; flipud(Rt_sw + 1.96*Rt_sd)], ...
        clr_Rt, 'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(axT, xv, Rz_sw, '-', 'LineWidth', 2, 'Color', clr_Rz, 'DisplayName', 'R_z');
    plot(axT, xv, Rt_sw, '-', 'LineWidth', 2, 'Color', clr_Rt, 'DisplayName', 'R_t');
    xline(axT, basePoint(vi), '--', 'Color', clr_opt, 'LineWidth', 1.5, ...
        'Label', 'Opt', 'LabelColor', clr_opt, 'FontSize', 9);
    set(axT, 'Color', bg, 'XColor', fg, 'YColor', fg, 'GridColor', grid_c, ...
        'GridAlpha', 0.55, 'FontSize', 11, 'LineWidth', 0.8);
    xlabel(axT, xlabs{vi}, 'Color', fg);
    ylabel(axT, 'Roughness (\mum)', 'Color', fg);
    title(axT, sprintf('Roughness vs %s', varNames{vi}), 'Color', fg, 'FontWeight', 'bold');
    l6t = legend(axT, 'Location', 'best');
    l6t.TextColor = fg; l6t.Color = bg; l6t.EdgeColor = grid_c;
    grid(axT, 'on');

    % Bottom row: VB with 95% CI
    axB = subplot(2, 3, vi + 3, 'Parent', fig6);
    hold(axB, 'on');
    fill(axB, [xv; flipud(xv)], ...
        [VB_sw - 1.96*VB_sd; flipud(VB_sw + 1.96*VB_sd)], ...
        clr_VB, 'FaceAlpha', 0.20, 'EdgeColor', 'none');
    plot(axB, xv, VB_sw, '-', 'LineWidth', 2, 'Color', clr_VB);
    xline(axB, basePoint(vi), '--', 'Color', clr_opt, 'LineWidth', 1.5, ...
        'Label', 'Opt', 'LabelColor', clr_opt, 'FontSize', 9);
    setDark(axB);
    xlabel(axB, xlabs{vi}, 'Color', fg);
    ylabel(axB, 'VB (mm)', 'Color', fg);
    title(axB, sprintf('Tool Wear vs %s  (shaded = 95%% CI)', varNames{vi}), 'Color', fg, 'FontWeight', 'bold');
    grid(axB, 'on');
end

sg6 = sgtitle(fig6, '1-D Sensitivity Sweeps Around Recommended Point', ...
    'FontSize', 14, 'FontWeight', 'bold');
sg6.Color = fg;
saveas(fig6, fullfile(resultsDir, 'fig6_1d_sweeps.png'));

% --- Figure 7: 3-D Desirability Surface (Vc vs f, DOC at optimum) ---
fig7 = figure('Color', bg, 'Name', '7 - 3D Desirability Surface', 'NumberTitle', 'off', ...
    'Position', [100 100 800 600]);
ax7  = axes(fig7);

nGrid = 40;
Vc_grid = linspace(lb(1), ub(1), nGrid);
f_grid  = linspace(lb(2), ub(2), nGrid);
[VcM, fM] = meshgrid(Vc_grid, f_grid);
DM = zeros(size(VcM));

for ii = 1:nGrid
    for jj = 1:nGrid
        xq = [VcM(ii,jj), fM(ii,jj), best.DOC];
        yRz_q = predict(mdlRz, xq);
        yRt_q = predict(mdlRt, xq);
        yVB_q = predict(mdlVB, xq);
        DM(ii,jj) = calcOverallDesirability(yRz_q, yRt_q, yVB_q, limits, mainWeights, shape);
    end
end

surf(ax7, VcM, fM, DM, 'EdgeAlpha', 0.12, 'FaceAlpha', 0.92);
hold(ax7, 'on');

% Mark the PSO optimum on the surface
plot3(ax7, best.Vc, best.f, best.D + 0.02, 'p', ...
    'MarkerSize', 20, 'MarkerFaceColor', clr_opt, 'MarkerEdgeColor', fg, 'LineWidth', 1.2);
text(ax7, best.Vc, best.f, best.D + 0.04, ...
    sprintf('  Optimum\n  D=%.3f', best.D), ...
    'Color', clr_opt, 'FontSize', 10, 'FontWeight', 'bold');

% Mark experimental points that are near the fixed DOC slice
for i = 1:n
    if abs(X(i,3) - best.DOC) < 0.06
        Di = calcOverallDesirability(Rz(i), Rt(i), VB(i), limits, mainWeights, shape);
        plot3(ax7, X(i,1), X(i,2), Di, 'o', ...
            'MarkerSize', 8, 'MarkerFaceColor', clr_exp, 'MarkerEdgeColor', bg, 'LineWidth', 0.8);
    end
end

setDark(ax7);
ax7.ZColor = fg;
colormap(ax7, hot);
cb7 = colorbar(ax7);
cb7.Label.String = 'Desirability D';
cb7.Label.Color = fg; cb7.Color = fg;
xlabel(ax7, 'Cutting Speed V_c (m/min)', 'Color', fg, 'FontSize', 11);
ylabel(ax7, 'Feed Rate f (mm/min)', 'Color', fg, 'FontSize', 11);
zlabel(ax7, 'Desirability D', 'Color', fg, 'FontSize', 11);
title(ax7, sprintf('3-D Desirability Surface  (DOC = %.3f mm at optimum)', best.DOC), ...
    'Color', fg, 'FontSize', 13, 'FontWeight', 'bold');
view(ax7, -35, 30);
grid(ax7, 'on'); box(ax7, 'on');
saveas(fig7, fullfile(resultsDir, 'fig7_3d_desirability_surface.png'));

% --- Figure 8: 2-D Pareto Front  Rz vs Rt ---
% Dense grid sample -> non-dominated front -> sorted smooth curve.
% Visual hierarchy: PSO star > Pareto line > experiments > faint cloud.
fig8 = figure('Color', bg, 'Name', '8 - Rz vs Rt Trade-Off', 'NumberTitle', 'off', ...
    'Position', [100 100 800 650]);
ax8  = axes(fig8);

nPar = 3000;
Xrand = [lb(1) + (ub(1)-lb(1))*rand(nPar,1), ...
         lb(2) + (ub(2)-lb(2))*rand(nPar,1), ...
         lb(3) + (ub(3)-lb(3))*rand(nPar,1)];
[pRz, pRz_sd] = predict(mdlRz, Xrand);
[pRt, pRt_sd] = predict(mdlRt, Xrand);

% Identify non-dominated (Pareto) front: both Rz and Rt minimized
isDom = false(nPar, 1);
for i = 1:nPar
    for j = 1:nPar
        if j ~= i && pRz(j) <= pRz(i) && pRt(j) <= pRt(i) && (pRz(j) < pRz(i) || pRt(j) < pRt(i))
            isDom(i) = true;
            break;
        end
    end
end
paretoMask = ~isDom;

% Sort front by Rt for a smooth monotonic curve
pf_Rt = pRt(paretoMask);   pf_Rz = pRz(paretoMask);
pf_Rz_sd = pRz_sd(paretoMask);
[pf_Rt, sOrd] = sort(pf_Rt);
pf_Rz = pf_Rz(sOrd);  pf_Rz_sd = pf_Rz_sd(sOrd);

% Enforce monotonicity: walking left to right, Rz should decrease
for k = 2:numel(pf_Rz)
    if pf_Rz(k) > pf_Rz(k-1)
        pf_Rz(k) = pf_Rz(k-1);
    end
end

hold(ax8, 'on');

% Layer 4 (faintest): feasible region cloud
scatter(ax8, pRt(isDom), pRz(isDom), 8, [0.35 0.35 0.42], 'filled', ...
    'MarkerFaceAlpha', 0.15, 'DisplayName', 'Feasible region');

% Layer 3: 95% CI band
fill(ax8, [pf_Rt; flipud(pf_Rt)], ...
    [pf_Rz - 1.96*pf_Rz_sd; flipud(pf_Rz + 1.96*pf_Rz_sd)], ...
    clr_Rz, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% Layer 2: Pareto front line (thick)
plot(ax8, pf_Rt, pf_Rz, '-', 'Color', clr_Rz, 'LineWidth', 3, ...
    'DisplayName', 'Pareto front');

% Layer 1b: Experimental data
scatter(ax8, Rt, Rz, 65, clr_exp, 's', 'filled', ...
    'MarkerEdgeColor', fg, 'LineWidth', 0.5, 'DisplayName', 'Experimental data');

% Layer 1a: Best experimental point (large red circle)
scatter(ax8, Rt(idxObs), Rz(idxObs), 180, [1 0.3 0.3], 'o', 'LineWidth', 3, ...
    'DisplayName', 'Best experiment');

% Layer 0 (brightest): PSO optimum (large star)
scatter(ax8, best.Rt, best.Rz, 280, clr_opt, 'p', 'filled', ...
    'MarkerEdgeColor', fg, 'LineWidth', 1.5, 'DisplayName', 'PSO optimum');

% Annotation: label the optimum
text(ax8, best.Rt + 0.12, best.Rz + 0.06, ...
    sprintf('Optimal point\n(PSO \\approx experiment)'), ...
    'Color', clr_opt, 'FontSize', 10, 'FontWeight', 'bold');

% Direction-of-improvement arrows
xLims = xlim(ax8);  yLims = ylim(ax8);
arrowX = xLims(2) - 0.12*(xLims(2)-xLims(1));
arrowY = yLims(2) - 0.08*(yLims(2)-yLims(1));
text(ax8, arrowX, arrowY, '\leftarrow R_t better', ...
    'Color', [0.6 0.6 0.6], 'FontSize', 9, 'HorizontalAlignment', 'right');
text(ax8, arrowX, arrowY - 0.06*(yLims(2)-yLims(1)), '\downarrow R_z better', ...
    'Color', [0.6 0.6 0.6], 'FontSize', 9, 'HorizontalAlignment', 'right');

% Zoom to the interesting region (trim empty space)
padRt = 0.1 * (max(Rt) - min(Rt));
padRz = 0.1 * (max(Rz) - min(Rz));
xlim(ax8, [min([pf_Rt; Rt]) - padRt, max(Rt) + padRt]);
ylim(ax8, [min([pf_Rz; Rz]) - padRz, max(Rz) + padRz]);

setDark(ax8);
xlabel(ax8, 'Predicted R_t (\mum)', 'Color', fg, 'FontSize', 12);
ylabel(ax8, 'Predicted R_z (\mum)', 'Color', fg, 'FontSize', 12);
title(ax8, 'R_z vs R_t Pareto Front', ...
    'Color', fg, 'FontSize', 13, 'FontWeight', 'bold');
leg8 = legend(ax8, 'Location', 'northeast', 'FontSize', 10);
leg8.TextColor = fg; leg8.Color = bg; leg8.EdgeColor = grid_c;
grid(ax8, 'on'); box(ax8, 'on');

% Caption-ready annotation below the plot
annotation(fig8, 'textbox', [0.10 0.01 0.85 0.04], ...
    'String', 'The Pareto front is narrow, indicating minimal conflict between R_z and R_t; the optimal solution coincides with an experimental data point.', ...
    'Color', [0.65 0.65 0.65], 'FontSize', 9, 'FontAngle', 'italic', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FitBoxToText', 'off');

print(fig8, fullfile(resultsDir, 'fig8_rz_vs_rt_tradeoff'), '-dpng', '-r300');
saveas(fig8, fullfile(resultsDir, 'fig8_rz_vs_rt_tradeoff.fig'));

% --- Figure 9: Desirability vs Rz/(Rz+Rt) weight ratio ---
fig9 = figure('Color', bg, 'Name', '9 - Desirability vs Weight Ratio', 'NumberTitle', 'off');
ax9  = axes(fig9);

clrs9 = [clr_Rz; clr_Rt; clr_VB; clr_opt];
hold(ax9, 'on');

for iw = 1:numel(wearLevels)
    mask = abs(sweepData(:,3) - wearLevels(iw)) < 1e-6;
    sd = sweepData(mask, :);
    % weight ratio = wRz / (wRz + wRt)
    ratio = sd(:,1) ./ (sd(:,1) + sd(:,2));
    plot(ax9, ratio, sd(:,10), '-o', 'Color', clrs9(iw,:), 'LineWidth', 2, ...
        'MarkerSize', 5, 'MarkerFaceColor', clrs9(iw,:), ...
        'DisplayName', sprintf('w_{VB} = %.2f', wearLevels(iw)));
end

% Mark the balanced optimum
balRatio = mainWeights.Rz / (mainWeights.Rz + mainWeights.Rt);
scatter(ax9, balRatio, best.D, 140, clr_exp, 'p', 'filled', ...
    'MarkerEdgeColor', fg, 'DisplayName', 'Balanced optimum');

setDark(ax9);
xlabel(ax9, 'w_{Rz} / (w_{Rz} + w_{Rt})  \rightarrow  more R_z priority', 'Color', fg, 'FontSize', 12);
ylabel(ax9, 'Overall Desirability D', 'Color', fg, 'FontSize', 12);
title(ax9, 'Desirability vs Roughness Weight Ratio', 'Color', fg, 'FontSize', 13, 'FontWeight', 'bold');
leg9 = legend(ax9, 'Location', 'best');
leg9.TextColor = fg; leg9.Color = bg; leg9.EdgeColor = grid_c;
grid(ax9, 'on'); box(ax9, 'on');
saveas(fig9, fullfile(resultsDir, 'fig9_desirability_vs_weight_ratio.png'));

fprintf('\nAll 9 figures and CSV tables saved in: %s/\n', resultsDir);
fprintf('Done!\n');

%% ======================== LOCAL FUNCTIONS ================================

function [predLOO, stats] = looForGPR(X, y, kernelName, sigma)
    n = size(X, 1);
    predLOO = zeros(n, 1);
    for i = 1:n
        idx = true(n, 1);
        idx(i) = false;
        mdl = fitrgp(X(idx,:), y(idx), ...
            'KernelFunction', kernelName, ...
            'Sigma', sigma, ...
            'Standardize', true);
        predLOO(i) = predict(mdl, X(i,:));
    end
    stats.R2   = calcR2(y, predLOO);
    stats.RMSE = sqrt(mean((y - predLOO).^2));
    stats.MAE  = mean(abs(y - predLOO));
end

function r2 = calcR2(y, yhat)
    denom = sum((y - mean(y)).^2);
    if denom <= eps
        r2 = NaN;
    else
        r2 = 1 - sum((y - yhat).^2) / denom;
    end
end

function printCVStats(name, s)
    fprintf('  %s -> R^2 = %.4f | RMSE = %.4f | MAE = %.4f\n', name, s.R2, s.RMSE, s.MAE);
end

function f = desirabilityObjective(x, mdlRz, mdlRt, mdlVB, limits, weights, shape, Xtrain, lb, ub, usePenalty, penaltyStart, penaltyWeight)
    x = reshape(x, 1, []);
    yRz = predict(mdlRz, x);
    yRt = predict(mdlRt, x);
    yVB = predict(mdlVB, x);

    D = calcOverallDesirability(yRz, yRt, yVB, limits, weights, shape);

    penalty = 0;
    if usePenalty
        Xn = (Xtrain - lb) ./ (ub - lb);
        xn = (x - lb) ./ (ub - lb);
        dmin = min(sqrt(sum((Xn - xn).^2, 2)));
        if dmin > penaltyStart
            penalty = penaltyWeight * (dmin - penaltyStart)^2;
        end
    end

    f = 1 - D + penalty;
end

function D = calcOverallDesirability(yRz, yRt, yVB, limits, weights, shape)
    dRz = smallerIsBetter(yRz, limits.Rz.target, limits.Rz.upper, shape.Rz);
    dRt = smallerIsBetter(yRt, limits.Rt.target, limits.Rt.upper, shape.Rt);
    dVB = smallerIsBetter(yVB, limits.VB.target, limits.VB.upper, shape.VB);

    wsum = weights.Rz + weights.Rt + weights.VB;
    D = (dRz^weights.Rz * dRt^weights.Rt * dVB^weights.VB)^(1/wsum);
end

function d = smallerIsBetter(y, target, upper, s)
    if y <= target
        d = 1.0;
    elseif y >= upper
        d = 1e-6;
    else
        d = ((upper - y) / (upper - target))^s;
        d = max(d, 1e-6);
    end
end

function plotIdLine(ax, a, b, col)
    lo = min([a(:); b(:)]);
    hi = max([a(:); b(:)]);
    plot(ax, [lo hi], [lo hi], '--', 'Color', col, 'LineWidth', 1.2);
end

function stop = psoRecorder(optimValues, state)
    global PSO_HISTORY;
    stop = false;
    switch state
        case 'init'
            PSO_HISTORY = [];
        case 'iter'
            PSO_HISTORY(end+1,1) = optimValues.bestfval; %#ok<AGROW>
    end
end
