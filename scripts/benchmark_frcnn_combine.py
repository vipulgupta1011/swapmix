import json
import argparse
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--obj')
parser.add_argument('--attr')
args = parser.parse_args()

result_obj = json.load(open(args.obj))
result_attr = json.load(open(args.attr))

baseline_overall = 0
baseline_new = 0
total_ques = 0
total_ques_only_correct = 0
changed_obj = 0
changed_attr = 0
changed_ques = {}

for img in result_obj :
    ques_res = result_obj[img]
    for ques in ques_res :
        found = False
        ans = int(ques_res[ques]["ans"])
        original_pred = int(ques_res[ques]["pred"][0][0])

        preds = ques_res[ques]["pred"]
        changes = ques_res[ques]["changes"]

        total_ques += 1

        if ans == original_pred :
            baseline_overall +=1

        if not ans == original_pred :
            continue

        total_ques_only_correct += 1
        for j in range(len(preds)) :
            pred = preds[j]
            for i in range(len(pred)) :
                change = changes[j*32+i]
                if change == "original" :
                    continue

                if int(pred[i]) != original_pred :
                    found = True
        
        if found :
            changed_obj += 1
            changed_ques[ques] = {}



for img in result_attr :
    ques_res = result_attr[img]
    for ques in ques_res :
        found = False
        ans = int(ques_res[ques]["ans"])
        original_pred = int(ques_res[ques]["pred"][0][0])

        preds = ques_res[ques]["pred"]
        changes = ques_res[ques]["changes"]

        if not ans == original_pred :
            continue

        for j in range(len(preds)) :
            pred = preds[j]
            for i in range(len(pred)) :
                change = changes[j*64+i]
                if change == "original" :
                    continue

                if int(pred[i]) != original_pred :
                    found = True
                    
        if found :
            changed_attr += 1
            changed_ques[ques] = {}

print ('model accuracy without SwapMix : ' , float(baseline_overall)/total_ques)
print ('object reliance: ', float(changed_obj)/total_ques_only_correct)
print ('attribute reliance: ', float(changed_attr)/total_ques_only_correct)
print ('context reliance : ', float(len(changed_ques))/total_ques_only_correct)
print ('effective accuracy : ', (float(baseline_overall - float(len(changed_ques)))/total_ques)) 
