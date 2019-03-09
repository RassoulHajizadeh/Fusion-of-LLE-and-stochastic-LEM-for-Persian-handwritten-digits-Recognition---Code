function [All_MN_neighboors, num_neighboors] = Mutual_Neighboors_func(sorted_index,initial_K,N)
% Mutual neighborhood conception has been proposed by us in the article by
% title "Mutual neighbors and diagonal loading-based sparse locally linear
% embedding", that has been published in Applied Artificial Intelligence (link: https://www.tandfonline.com/doi/abs/10.1080/08839514.2018.1486129). 

for c1=1:N
    MN_neighboors = [];
    pre_niegboors = sorted_index(2:(1+initial_K),c1);
    count_1 = 1;
    for c2=1:initial_K
        temp = find(sorted_index(:,pre_niegboors(c2))==c1);
        if temp < initial_K+2
            MN_neighboors(count_1) = pre_niegboors(c2);
            count_1 = count_1 + 1;
        end
    end
    if count_1 ==1
        MN_neighboors = pre_niegboors(1);
        count_1 = count_1 +1;
    end
    All_MN_neighboors{c1} = MN_neighboors; % index of mutual neighbors for all data points.
    num_neighboors(c1) = count_1-1; % number of mutual neighbors for all data points.
end

