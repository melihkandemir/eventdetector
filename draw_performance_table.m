function draw_performance_table()
     addpath ../../util
     
    %vidList=[2 7 9:14];
    vidList=[2 7:14];
    %vidList=3:5;
     
     perf=zeros(length(vidList),3);
     
     prec_all1=[];
     prec_all2=[];
     prec_all3=[];
     
     recall_all1=[];
     recall_all2=[];
     recall_all3=[];        
     
     for ii=1:length(vidList)
         load(['/home/mkandemi/Dropbox/results/zeiss/video' num2str(vidList(ii)) '_performance_model1']);
         auc1=auc;
         prbep1=prbep;
         recalls1{ii}=recall;
         precs1{ii}=prec;
         
         load(['/home/mkandemi/Dropbox/results/zeiss/video' num2str(vidList(ii)) '_performance_model3']);
         auc2=auc;
         prbep2=prbep;     
         recalls2{ii}=recall;
         precs2{ii}=prec;         
         
         load(['/home/mkandemi/Dropbox/results/zeiss/video' num2str(vidList(ii)) '_performance_model5']);
         auc3=auc;
         prbep3=prbep;            
         recalls3{ii}=recall;
         precs3{ii}=prec;   
         
         %fprintf('Video %d (%d) & %.2f & %.2f & %.2f \\\\\n',ii,vidList(ii),auc1,auc2,auc3);
         perf(ii,:)=[auc1 auc2 auc3];
                
     end
     
     [recall_all1,prec_all1]=draw_average_pr_curve(recalls1,precs1);
     [recall_all2,prec_all2]=draw_average_pr_curve(recalls2,precs2);
     [recall_all3,prec_all3]=draw_average_pr_curve(recalls3,precs3);
     
     auc1=trapz(recall_all1,prec_all1);
     auc2=trapz(recall_all2,prec_all2);
     auc3=trapz(recall_all3,prec_all3);
            
    perf
     fprintf('\\hline\nAverage & %.3f & %.3f & %.3f\\\\\n\\hline\n',mean(perf));     
     
    close all;    
     figure(1);
     set(gca,'FontSize',25,'FontWeight','Bold');               
     plot(recall_all3,prec_all3,'Color',[0 0 255]/255,'LineWidth',5); hold on; 
     plot(recall_all2,prec_all2,'Color',[0 255 0]/255,'LineWidth',5); hold on;      
     plot(recall_all1,prec_all1,'Color',[255 0 0]/255,'LineWidth',5); hold on;     
     xlabel('Recall'); ylabel('Precision');
     axis([0 1 0 1]);         
     legend(sprintf('NMF^{ } (AUC: %.2f)^{      }',auc3), ...
            sprintf('OC-SVM^{ } (AUC: %.2f)^{      }',auc2), ...
            sprintf('MOGP^{      } (AUC: %.2f)^{      } ',auc1), ...
             'Location','SouthWest');
     saveas(gcf,'/home/mkandemi/Desktop/prcurve.eps','psc2');      
     
end
