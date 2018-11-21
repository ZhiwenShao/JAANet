clc,clear

part_ind = [1,2,1,3,2,3];

for i=1:2:size(part_ind,2)
    part1_path=importdata(['BP4D_part',num2str(part_ind(1,i)),'_path.txt']);
    part1_land=importdata(['BP4D_part',num2str(part_ind(1,i)),'_land.txt']);
    part1_AUoccur=importdata(['BP4D_part',num2str(part_ind(1,i)),'_AUoccur.txt']);
    part1_biocular=importdata(['BP4D_part',num2str(part_ind(1,i)),'_biocular.txt']);
    
    part2_path=importdata(['BP4D_part',num2str(part_ind(1,i+1)),'_path.txt']);
    part2_land=importdata(['BP4D_part',num2str(part_ind(1,i+1)),'_land.txt']);
    part2_AUoccur=importdata(['BP4D_part',num2str(part_ind(1,i+1)),'_AUoccur.txt']);
    part2_biocular=importdata(['BP4D_part',num2str(part_ind(1,i+1)),'_biocular.txt']);
    
    combine_path =[part1_path;part2_path];
    combine_land =[part1_land;part2_land];
    combine_AUoccur =[part1_AUoccur;part2_AUoccur];
    combine_biocular =[part1_biocular;part2_biocular];
    
    combine_randAUind = randperm(size(combine_path,1));
    save(['combine_randAUind_',num2str(part_ind(1,i)),'_',num2str(part_ind(1,i+1)),'.mat'],'combine_randAUind');
%     load (['combine_randAUind_',num2str(part_ind(1,i)),'_',num2str(part_ind(1,i+1))]);
    
    final_curr_path=combine_path(combine_randAUind,:);
    final_curr_land=combine_land(combine_randAUind,:);
    final_curr_AUoccur=combine_AUoccur(combine_randAUind,:);
    final_curr_biocular=combine_biocular(combine_randAUind,:);
    
    dlmwrite(['BP4D_combine_',num2str(part_ind(1,i)),'_',num2str(part_ind(1,i+1)),'_path.txt'], final_curr_path, 'delimiter','', 'newline', 'pc');
    dlmwrite(['BP4D_combine_',num2str(part_ind(1,i)),'_',num2str(part_ind(1,i+1)),'_land.txt'], final_curr_land, 'delimiter',' ', 'newline', 'pc');
    dlmwrite(['BP4D_combine_',num2str(part_ind(1,i)),'_',num2str(part_ind(1,i+1)),'_AUoccur.txt'], final_curr_AUoccur, 'delimiter',' ', 'newline', 'pc');
    dlmwrite(['BP4D_combine_',num2str(part_ind(1,i)),'_',num2str(part_ind(1,i+1)),'_biocular.txt'], final_curr_biocular, 'delimiter',' ', 'newline', 'pc');
end
