%% Semi-supvised learning

%  Instructions
%  ------------
% 
%

%% Initialisation
clear ; close all; clc

%% =========== Part 1: CHARGEMENT ET VISUALISATION DES DONNEES =============
%  Dans ce partie, on charge la donnée MNIST de http://yann.lecun.com/exdb/mnist/
%  gr?ce à les deux fonctions: loadMNISTImages et loadMNISTLabels 
%  pour simplifier et aplliquer dans notre algorithme, on note X et Y correspondant à image et label 
%  afin de les visualiser, on choisit 100 échantillons aléatoires
%


% Read training images

images = loadMNISTImages('train-images.idx3-ubyte');
X = images';
m = size(X, 1);

% Load training labels

fprintf('chargement ...\n')

labels = loadMNISTLabels('train-labels.idx1-ubyte');
Y = labels;


% On choisit N échantillon aléatoire
N = 100;
sel = randperm(size(X, 1));
sel = sel(1:N);
displayData(X(sel, :));

x = X(sel, :);
y = Y(sel,:);
fprintf('Program paused. Press enter to continue.\n');
pause;

%DETERMITER NOBRE DE VALEURS UNLABELED!
NL = 20;
for i  = 1:NL
    y(i,1) = -1;
end
%% ================ Part 2: CHARGEMENT DES PARAMETRES ET CALCUL DES POIDS ================

Wgauss = zeros(N);
sigma = 1;

for i = 1:N
    for j = 1:N
        Wgauss(i,j) = exp(sum((x(i,:)-x(j,:)).^2)/sigma^2);
    end
end

for i = 1:N
    Wgauss(i,i) = 1;
end


%% ================= Part 3: CALCUL ET TRI VECTEURS PROPRES ====================

D = zeros(N);

for i = 1:N
    D(i,i) = sum(Wgauss(i,:));
end


L =(D-Wgauss);

[V,Diag] = eig(L);

vp = eig(L);

vpsorted = sort(vp);


%% ================= Part 4: CONSTRUCTION DE LA SOLUTION====================
Ysol = V(:,1);
k = 3 ;

[idx,C] = kmeans(Ysol,k);



%% =================== Part 5: MINIMISE FONCTION DE COUT ======================
%  After you have completed the two functions computeCentroids and
%  findClosestCentroids, you have all the necessary pieces to run the
%  kMeans algorithm. In this part, you will run the K-Means algorithm on
%  the example dataset we have provided. 
%
% fprintf('\nRunning K-Means clustering on example dataset.\n\n');
% 
% % Load an example dataset
% load('ex7data2.mat');
% 
% % Settings for running K-Means
%  
% max_iters = 10;
% 
% % For consistency, here we set centroids to specific values
% % but in practice you want to generate them automatically, such as by
% % settings them to be random examples (as can be seen in
% % kMeansInitCentroids).
% initial_centroids = [3 3; 6 2; 8 5];
% 
% % Run K-Means algorithm. The 'true' at the end tells our function to plot
% % the progress of K-Means
% [centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);   %change in runkMeans
% fprintf('\nK-Means Done.\n\n');
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;
%% =================== Part 6: CALCUL DU TAUX D'ERREUR  ET VALIDATION ======================
