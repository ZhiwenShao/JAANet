clc,clear;

part_ind = [1,2,1,3,2,3];

for i=1:2:length(part_ind)
    curr_AUoccur = importdata(['BP4D_combine_',num2str(part_ind(1,i)),'_',num2str(part_ind(1,i+1)),'_AUoccur.txt']);   
    AU_num = size(curr_AUoccur,2);   
    total_num = size(curr_AUoccur,1);
    
    [occur_row, occur_col] = find (curr_AUoccur>0);

    for j=1:AU_num
        t_ind = find(occur_col==j);
        AUoccur_id{j,1} = occur_row(t_ind,:);    
        AUoccur_rate(j,1) = size(AUoccur_id{j,1},1)/total_num;
    end
    
    AU_weight = 1./AUoccur_rate';
    AU_weight = AU_weight/sum(AU_weight(:))*AU_num;
    dlmwrite(['BP4D_combine_',num2str(part_ind(1,i)),'_',num2str(part_ind(1,i+1)),'_weight.txt'], AU_weight, 'delimiter',' ', 'newline', 'pc');
    % for dice loss
    dice_weight=AU_weight/AU_num;
    
end

test_weight = ones(size(AU_weight));
dlmwrite('BP4D_test_weight.txt', test_weight, 'delimiter',' ', 'newline', 'pc');
