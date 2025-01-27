import cv2
import matplotlib.pyplot as plt
from googletrans import Translator
import pytesseract
import numpy as np
import pandas as pd

def find_text(a):
    #translator = Translator()
    text = pytesseract.image_to_string(a,config='--psm 3')
    a = text.split("\n")
    new_list = []
    x = [new_list.append(i.strip()) for i in a if len(i.strip())>0]
    dict1 = {}
    count=0
    last_e = ''
    s_l1 = new_list[count].split(" ")
    dict1['serial number'] = s_l1[0]
    dict1['id'] = "".join(s_l1[1:])
    count+=1
    dict1['name'] = new_list[count][new_list[count].lower().find('name')+6:]
    last_e = 'name'
    count+=1
    if 'father' in new_list[count].lower():
        dict1['father name'] = new_list[count][new_list[count].lower().find('father')+15:]
        last_e = 'father name'
    elif 'husband' in new_list[count].lower():
        dict1['husband name'] = new_list[count][new_list[count].lower().find('husband')+16:]
        last_e = 'husband name'
    elif 'mother' in new_list[count].lower():
        dict1['mother name'] = new_list[count][new_list[count].lower().find('mother')+15:]
        last_e = 'mother name'
    elif 'other' in new_list[count].lower():
        dict1['other name'] = new_list[count][new_list[count].lower().find('other')+14:]
        last_e = 'other name'
    elif 'wife' in new_list[count].lower():
        dict1['wife name'] = new_list[count][new_list[count].lower().find('wife')+13:]
        last_e = 'wife name'
    else:
        dict1[last_e]+=f' {new_list[count]}'
        count+=1
        if 'father' in new_list[count].lower():
            dict1['father name'] = new_list[count][new_list[count].lower().find('father')+15:]
            last_e = 'father name'
        elif 'husband' in new_list[count].lower():
            dict1['husband name'] = new_list[count][new_list[count].lower().find('husband')+16:]
            last_e = 'husband name'
        elif 'mother' in new_list[count].lower():
            dict1['mother name'] = new_list[count][new_list[count].lower().find('mother')+15:]
            last_e = 'mother name'
        elif 'other' in new_list[count].lower():
            dict1['other name'] = new_list[count][new_list[count].lower().find('other')+14:]
            last_e = 'other name'
        elif 'wife' in new_list[count].lower():
            dict1['wife name'] = new_list[count][new_list[count].lower().find('wife')+13:]
            last_e = 'wife name'
    count+=1
    if 'house' in new_list[count].lower():
        dict1['house number'] = new_list[count][new_list[count].lower().find('house')+14:]
        last_e = 'house number'
    else:
        dict1[last_e]+=f' {new_list[count]}'
        count+=1
        if 'house' in new_list[count].lower():
            dict1['house number'] = new_list[count][new_list[count].lower().find('house')+14:]
            last_e = 'house number'
    count+=1
    if 'age' in new_list[count].lower() and 'gender' in new_list[count].lower():
        dict1['age'] = new_list[count][new_list[count].lower().find('age')+5:new_list[count].lower().find('gender')-1].strip()
        dict1['gender'] = new_list[count][new_list[count].lower().find('gender')+8:].strip()
    else:
        dict1[last_e]+=f' {new_list[count]}'
        count+=1       
        if 'age' in new_list[count].lower() and 'gender' in new_list[count].lower():
            dict1['age'] = new_list[count][new_list[count].lower().find('age')+5:new_list[count].lower().find('gender')-1].strip()
            dict1['gender'] = new_list[count][new_list[count].lower().find('gender')+8:].strip()
    count+=1
    if len(dict1['id'])<3:
        for i in new_list[count:]:
            dict1['id'] = "".join(i.split(" "))
    #if 'name' in dict1.keys():
    #    dict1['name kannada kannada'] = translator.translate(dict1['name'], dest='kn').text
    #if 'father name' in dict1.keys():
    #    dict1['father name kannada'] = translator.translate(dict1['father name'], dest='kn').text
    #if 'mother name' in dict1.keys():
    #    dict1['mother name kannada'] = translator.translate(dict1['mother name'], dest='kn').text
    #if 'husband name' in dict1.keys():
    #    dict1['husband name kannada'] = translator.translate(dict1['husband name'], dest='kn').text        
    #if 'other name' in dict1.keys():
    #    dict1['other name kannada'] = translator.translate(dict1['other name'], dest='kn').text        
    return dict1

def get_data(filename):
    img = cv2.imread(filename,0)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
    img_bin = 255-img_bin
    img_bin1 = 255-img
    thresh1,img_bin1_otsu = cv2.threshold(img_bin1,128,255,cv2.THRESH_OTSU)    
    img_bin2 = 255-img
    thresh1,img_bin_otsu = cv2.threshold(img_bin2,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1]//200))
    eroded_image = cv2.erode(img_bin_otsu, vertical_kernel, iterations=3)
    vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1]//200, 1))
    horizontal_lines = cv2.erode(img_bin_otsu, hor_kernel, iterations=5)
    horizontal_lines = cv2.dilate(horizontal_lines, hor_kernel, iterations=5)
    vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)
    thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bitxor = cv2.bitwise_xor(img,vertical_horizontal_lines)
    bitnot = cv2.bitwise_not(bitxor)
    vertical_horizontal_lines_not = cv2.bitwise_not(vertical_horizontal_lines)
    contours, hierarchy = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda x:x[1][1]))
    img_new = img.copy()
    img_safe = img.copy()
    boxes = []
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      if (w*h>5000 and w*h<10000):
        image = cv2.rectangle(img_new,(x-10,y-10),(x+w+10,y+h+10),(0,255,0),2)
        boxes.append([x-5,y-5,w+10,h+10])
    for i in boxes:
        img_safe[i[1]:i[1]+i[3],i[0]:i[0]+i[2]] = 255
    boxes = []
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      if (w*h>10000 and w<1000 and h<1000):
        image = cv2.rectangle(img_new,(x-2,y-2),(x+w+2,y+h+2),(0,255,0),2)
        boxes.append([x-2,y-2,w+4,h+4])
    df_final = pd.DataFrame(columns = ['serial number','id','name','father name','husband name','mother name','other name', 'name kannada', 
                                       'father name kannada', 'husband name kannada','mother name kannada','other name kannada', 
                                       'house number', 'age', 'gender'])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    vertical_lines = cv2.dilate(vertical_lines, kernel, iterations=2)
    img_safe = cv2.bitwise_or(img_safe,vertical_lines)
    horizontal_lines = cv2.dilate(horizontal_lines, kernel, iterations=2)
    img_safe = cv2.bitwise_or(img_safe,horizontal_lines)
    count = 0
    box_list = []
    for i in boxes:
        x,y,w,h = i
        roi = img_safe[y:y+h,x:x+w,]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        border = cv2.copyMakeBorder(roi,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
        try:
            dict1 = find_text(border)
            box_list.append([i,dict1])
        except:
            box_list.append([i,None])
            pass
    box_list_new = sorted(box_list,key=lambda x:x[0][0])
    box_list_new = sorted(box_list_new,key=lambda x:x[0][1])
    dict_count = {}
    for i,j in enumerate(box_list_new):
        if j[1]!=None:
            if j[1]['serial number'].isnumeric():
                a = int(j[1]['serial number'])-i
                if a not in dict_count:
                    dict_count[a] = 1
                else:
                    dict_count[a] += 1
    serial_start = 0
    max_count = -1
    for i in dict_count:
        if dict_count[i]>max_count:
            max_count = dict_count[i]
            serial_start = i
    for i,j in enumerate(box_list_new):
        if j[1]!=None:
            j[1]['serial number'] = serial_start+i
            df_final = df_final._append(j[1],ignore_index=True)   
    return df_final
    
    
                                       
if __name__ == '__main__':
    df_final = pd.DataFrame(columns = ['serial number','id','name','father name','husband name','mother name','other name','wife name','name kannada', 
                                       'father name kannada', 'husband name kannada','mother name kannada','other name kannada','wife name kannada','house number', 'age', 'gender'])
    for i in range(3,33):
        filename = f"C:\\Users\\shiva\\Downloads\\shivam_anaconda\\pdf_extraction\\data_folder\\_{i}_.png"
        df_temp = get_data(filename)
        df_final = df_final._append(df_temp,ignore_index=True)
        print(f"Finished extracting page {i}. Data shape : {df_temp.shape}")
    df_final.to_csv('final_file.csv')
