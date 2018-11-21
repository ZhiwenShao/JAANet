clc,clear

part_num = 3;

for p=1:part_num
    input_land=importdata(['BP4D_part',num2str(p),'_land.txt']);
    num = size(input_land,1);
    
    biocular=[];
    for i=1:num
        l_ocular(1,1)=sum(input_land(i,2*(20:1:25)-1));
        l_ocular(1,2)=sum(input_land(i,2*(20:1:25)));
        l_ocular(1,:)=l_ocular(1,:)/6;
        r_ocular(1,1)=sum(input_land(i,2*(26:1:31)-1));
        r_ocular(1,2)=sum(input_land(i,2*(26:1:31)));  
        r_ocular(1,:)=r_ocular(1,:)/6;

        biocular(i,1)=(l_ocular(1,1)-r_ocular(1,1))*(l_ocular(1,1)-r_ocular(1,1))+(l_ocular(1,2)-r_ocular(1,2))*(l_ocular(1,2)-r_ocular(1,2)); 
    end
    dlmwrite(['BP4D_part',num2str(p),'_biocular.txt'],biocular,'delimiter',' ', 'newline', 'pc');
end