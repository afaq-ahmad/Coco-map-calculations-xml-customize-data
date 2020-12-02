
import pandas as pd
import os
import shutil
import numpy as np
import glob
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

XML_folder='GroundTruth_XML/'
Folders=os.listdir()[1:2]
print(Folders)
Results_saver=[]
for folder in Folders:
    Pre_Files_dir=folder+'/'

    File_name_change_Dir=XML_folder.replace('/','').replace('\\','')+'_INT_Fortmat'
    Ground_truth_json=XML_folder.replace('/','').replace('\\','')+'.json'
    Files_names_data_path=Pre_Files_dir.replace('/','').replace('\\','')+'.csv'

    Files=os.listdir(XML_folder)
    if not os.path.exists(File_name_change_Dir):
        os.mkdir(File_name_change_Dir)

    Files_names_data=[]
    for file in Files:
        shutil.copy(XML_folder+'/'+file,File_name_change_Dir+'/'+str(Files.index(file))+'.xml')
        Files_names_data.append([file,str(Files.index(file))+'.xml'])

    Files_names_data=pd.DataFrame(Files_names_data,columns=['Old File Name','New File Name'])
    # Files_names_data.to_csv(Files_names_data_path,index=False)

    from voc2coco import convert
    xml_files = glob.glob(os.path.join(File_name_change_Dir, "*.xml"))
    print("Number of xml files: {}".format(len(xml_files)))
    convert(xml_files, Ground_truth_json)

    with open(Ground_truth_json,'r') as f:
        Data_gt=json.load(f)
    # Files_names_data=pd.read_csv(Files_names_data)

    Actual_Files_names=pd.DataFrame([[g['file_name'].split('\\')[-1],g['id']] for g in Data_gt['images']],columns=['Imagename','id'])
    Pre_Files=os.listdir(Pre_Files_dir)
    Data_predicted_json=[]
    for ii in range(len(Pre_Files)):
    #     print(ii)
        with open(Pre_Files_dir+'/'+Pre_Files[ii],'r') as f:
            prefile_data=f.read().split('\n')
            prefile_data=[g.split() for g in prefile_data if g!='']
            if len(prefile_data)==0:
                continue
            prefile_data=np.array(prefile_data)
            prefile_data=prefile_data.astype(object)
            prefile_data[:,1]=prefile_data[:,1].astype(float)
            prefile_data[:,2:]=prefile_data[:,2:].astype(int)
            
        File_new_name=Files_names_data[Files_names_data['Old File Name']==Pre_Files[ii].replace('.txt','.xml')]['New File Name'].values
        if len(File_new_name)==0:
            File_new_name=Files_names_data[Files_names_data['Old File Name']==Pre_Files[ii].replace('.txt','.pdf.xml')]['New File Name'].values
            if len(File_new_name)==0:
                continue
            File_new_name=File_new_name[0]
        else:
            
            File_new_name=File_new_name[0]
            
        Image_id=Actual_Files_names[Actual_Files_names['Imagename']==File_new_name.replace('.xml','.png')]['id'].values
        if len(Image_id)==0:
            Image_id=Actual_Files_names[Actual_Files_names['Imagename']==File_new_name.replace('.xml','.jpg')]['id'].values

        for g in prefile_data:
            _,score,x1,y1,x2,y2=g
            Data_predicted_json.append({"image_id":Image_id[0],"category_id":0,"bbox":[x1,y1,x2-x1,y2-y1],"score":score})

    cocoGt=COCO(Ground_truth_json)
    cocoDt=cocoGt.loadRes(Data_predicted_json)

    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    print(folder)
    cocoEval.summarize()
    
    for g in [0.5,0.75,0.85,0.95]:
        cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
        cocoEval.params.iouThrs=[g]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        Av_p,Av_r=cocoEval.stats[0],cocoEval.stats[-4]
        Results_saver.append([folder,g,np.round(Av_p,3),np.round(Av_r,3)])
pd.DataFrame(Results_saver,columns=['Folder Name','IOU','Average Precision  (AP)','Average Recall     (AR)']).to_csv('Results.csv',index=False)