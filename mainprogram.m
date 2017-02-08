%% Semi-supvised learning

%  Instructions
%  ------------
% 
%

%% Initialisation
clear ; close all; clc

%% =========== Part 1: CHARGEMENT ET VISUALISATION DES DONNEES =============
%  Dans ce partie, on charge la donn¨¦e MNIST de http://yann.lecun.com/exdb/mnist/
%  gr?ce ¨¤ les deux fonctions: loadMNISTImages et loadMNISTLabels 
%  pour simplifier et aplliquer dans notre algorithme, on note X et Y correspondant ¨¤ image et label 
%  afin de les visualiser, on choisit 100 ¨¦chantillons al¨¦atoires
%


% Read training images

images = loadMNISTImages('train-images.idx3-ubyte');
X = images';
m = size(X, 1);

% Load training labels

fprintf('chargement ...\n')

labels = loadMNISTLabels('train-labels.idx1-ubyte');
Y = labels;


% On choisit 100 ¨¦chantillon al¨¦atoire
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 2: CHARGEMENT DES PARAMETRES ET CALCUL DES POIDS ================
% In this part of the exercise, we load some pre-initialized 
% neural network parameters.

% fprintf('\nLoading Saved Neural Network Parameters ...\n')
% 
% % Load the weights into variables Theta1 and Theta2
% load('ex4weights.mat');
% 
% % Unroll parameters 
% nn_params = [Theta1(:) ; Theta2(:)];

%% ================= Part 3: CALCUL ET TRI VECTEURS PROPRES ====================



%% ================= Part 4: CONSTRUCTION DE LA SOLUTION====================


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
