
data =load('机器学习\实验\实验二\orl.mat');
whos
images = data.D;
features = data.X;
labels = data.y;


% 每行显示 10 张图像
num_rows = 20;
num_cols = size(images, 4) / num_rows;

% 设置图形窗口大小
fig = figure;
set(fig, 'Position', [300, 300, 1200, 1200]); % 调整窗口的位置和大小
% 重新组织图像数据，将10张图像堆叠在一起
% reshaped_images = reshape(images, size(images, 1), size(images, 2), 1, []);

% 显示整个布局
% montage(reshaped_images, 'Size', [num_rows, num_cols]);
% 

% 计算平均脸
mean_face = mean(features, 2); 
% 将平均脸的特征向量转换为图像矩阵
% average_face_matrix = reshape(mean_face, [32, 32]);
% % 可视化平均脸
% imshow(uint8(average_face_matrix), []);
% title('平均脸');

% 将每张图像减去平均脸X`
num_images = size(features, 2);
centered_features = features - repmat(mean_face, 1, num_images);
% 显示每行20张图片
num_rows = 20;
num_cols = num_images / num_rows;
% reshaped_images = reshape(centered_features, [32, 32, 1, num_images]);
% montage(reshaped_images, 'Size', [num_rows, num_cols]);
% title('减去平均脸之后的图像');

% % 检查维度是否大于样本数
% if size(centered_features, 1) > size(centered_features, 2)
%     % 维度大于样本数，使用SVD
%     [U, S, V] = svd(centered_features');
%     
%     % 选择前 K 个较大的奇异值对应的奇异向量
%     K = 100;
%     selected_singular_vectors = U(:, 1:K);
%     
%     % 使用SVD降维后的奇异向量将原始数据投影到降维后的空间中
%     new_data = centered_features * selected_singular_vectors;
%     
%     % 重构数据
%     pca_face = new_data * selected_singular_vectors';
%     % 将重构后的数据重新整形为图像矩阵
%     reshaped_images = reshape(pca_face, 32, 32,[]);
%     montage(reshaped_images,'Size',[num_rows, num_cols]);
%     title('PCA重构后的图像(SVD)');
% else
%     % 维度不大于样本数，使用特征值分解
%     K = 100;
% 
%     % 2. 计算协方差矩阵
%     cov_matrix = cov(centered_features');
%     % 3. 计算协方差矩阵的特征值和特征向量
%     [eigenvalues, eigenvectors] = eig(cov_matrix);
%     % 4. 将特征值排序
%     eigenvalues_diag = diag(eigenvalues);
%     % 对特征值进行降序排序，同时记录排序索引
%     [sorted_eigenvalues, sorted_indices] = sort(eigenvalues_diag, 'descend');
%     % 5. 保留前 N 个较大特征值对应的特征向量
%     K = 100; 
% 
%     % 获取前 K 个较大特征值对应的特征向量
%     selected_eigenvectors = eigenvectors(:, sorted_indices(1:K));
% 
%     % 使用PCA降维后的特征向量将原始数据投影到降维后的空间中
%     new_data = centered_features' * selected_eigenvectors;
% 
%     pca_face = new_data * selected_eigenvectors';
% 
%     % 将重构后的数据重新整形为图像矩阵
%     reshaped_images = reshape(pca_face, 32, 32,[]);
%     % 使用montage函数显示重构的图像
%     montage(reshaped_images,'Size',[num_rows, num_cols]);
%     title('PCA重构后的图像(EVD)');
% end

centered_features = centered_features';
% 计算特征之间的协方差矩阵
cov_matrix = (centered_features' * centered_features) ./ 400;
% 计算协方差矩阵的特征值和特征向量
[eigenvectors, eigenvalues] = eig(cov_matrix);

% 将特征值排序
eigenvalues_diag = diag(eigenvalues);

% 对特征值进行降序排序，同时记录排序索引
[sorted_eigenvalues, sorted_indices] = sort(eigenvalues_diag, 'descend');
% 选择前N个最大特征值
N = 100;
% 获取前N个较大特征值对应的特征向量
selected_eigenvectors = eigenvectors(:, sorted_indices(1:N));

% 使用PCA降维后的特征向量将原始数据投影到降维后的空间中
new_data = centered_features * selected_eigenvectors;
reconstructed_data = new_data * selected_eigenvectors';
reconstructed_images = reshape(reconstructed_data', 32, 32, []);
montage(reconstructed_images, 'Size', [num_rows, num_cols]);
title('PCA重构后的图像(EVD)');