import json
import argparse
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--obj')
parser.add_argument('--attr')
args = parser.parse_args()

result_obj = json.load(open(args.obj))
result_attr = json.load(open(args.attr))

baseline_correct = 0
total_ques = 0
overall_ques = 0
changed_obj = 0
changed_attr = 0
changed = {}
changed_proportion = 0
changed_proportion_obj = 0


for img in result_obj :
    result_obj_img = result_obj[img]
    answers = result_obj_img['answers']
    original_prediction = {}
    perturbed_predictions = {}
    #original_prediction = dict.fromkeys(result_obj_img['predictions'].keys(),{})

    for obj in result_obj_img :
        if obj == 'answers' :
            continue
        relevant_ques = result_obj_img[obj]['relevant_questions']
        if relevant_ques == [] :
            continue
        if len(result_obj_img[obj]['prediction']) == 0:
            continue
        pred_original = result_obj_img[obj]['prediction']['0']['pred']
        for j in range(len(pred_original)) :
            preds = pred_original[j]
            for i in range(len(preds)) :
                rele_ques = relevant_ques[j*32+i]
                orig_pred = preds[i]

                if rele_ques not in original_prediction :
                    original_prediction[rele_ques] = orig_pred
                    perturbed_predictions[rele_ques] = {}

    for obj in result_obj_img :
        if obj == 'answers' :
            continue
        res_obj = result_obj_img[obj]
        relevant_questions = res_obj['relevant_questions']
        if relevant_questions == [] :
            continue
        if len(res_obj['prediction']) == 0:
            continue
        for j in range(1,len(res_obj['changes'])) :
            preds_obj = res_obj['prediction'][str(j)]['pred']
            for i in range(len(preds_obj)) :
                preds = preds_obj[i]
                for k in range(len(preds)) :
                    rele_ques = relevant_questions[i*32+k]
                    pred = preds[k]

                    if obj not in perturbed_predictions[rele_ques] :
                        perturbed_predictions[rele_ques][obj] = []

                    perturbed_predictions[rele_ques][obj].append(pred)

    for ques in perturbed_predictions :

        overall_ques += 1
        if original_prediction[ques] == answers[ques] :
            baseline_correct += 1

        if original_prediction[ques] != answers[ques] :
            continue

        found = False
        change_output = {}
        change_output_obj = {}
        no_of_changes = 0
        total_ques += 1
        for obj_interest in perturbed_predictions[ques] :
            pred_obj = perturbed_predictions[ques][obj_interest]
            change_output[obj_interest] = []
            change_output_obj[obj_interest] = 0

            no_of_changes += len(pred_obj)
            for i in range(len(pred_obj)) :
                if pred_obj[i] != original_prediction[ques] :
                    found = True
                    change_output[obj_interest].append(1)
                    change_output_obj[obj_interest] = 1
                else :
                    change_output[obj_interest].append(0)

        sum_ques = 0
        sum_ques_obj = 0
        for obj_interest in change_output :
            sum_obj = sum(change_output[obj_interest])
            sum_ques += sum_obj
            if sum_obj > 0 :
                sum_ques_obj += 1

        if len(change_output) > 0 :
            prop_obj = float(sum_ques_obj)/len(change_output)
        else :
            prop_obj = 0

        if no_of_changes > 1 :
            prop = float(sum_ques)/(no_of_changes)
        else :
            prop = 0

        changed_proportion += prop
        changed_proportion_obj += prop_obj

        if found :
            changed_obj += 1
            changed[ques] = {}

for img in result_attr :
    result_attr_img = result_attr[img]
    answers = result_attr_img['answers']
    original_prediction = {}
    perturbed_predictions = {}

    for obj in result_attr_img :
        if obj == 'answers' :
            continue
        relevant_ques = result_attr_img[obj]['relevant_questions']
        if relevant_ques == [] :
            continue
        if len(result_attr_img[obj]['prediction']) == 0:
            continue
        pred_original = result_attr_img[obj]['prediction']['0']['pred']
        for j in range(len(pred_original)) :
            preds = pred_original[j]
            for i in range(len(preds)) :
                rele_ques = relevant_ques[j*32+i]
                orig_pred = preds[i]

                if rele_ques not in original_prediction :
                    original_prediction[rele_ques] = orig_pred
                    perturbed_predictions[rele_ques] = {}

    for obj in result_attr_img :
        if obj == 'answers' :
            continue
        res_obj = result_attr_img[obj]
        relevant_questions = res_obj['relevant_questions']
        if relevant_questions == [] :
            continue
        if len(res_obj['prediction']) == 0:
            continue
        for j in range(1,len(res_obj['changes'])) :
            preds_obj = res_obj['prediction'][str(j)]['pred']
            for i in range(len(preds_obj)) :
                preds = preds_obj[i]
                for k in range(len(preds)) :
                    rele_ques = relevant_questions[i*32+k]
                    pred = preds[k]

                    if obj not in perturbed_predictions[rele_ques] :
                        perturbed_predictions[rele_ques][obj] = []

                    perturbed_predictions[rele_ques][obj].append(pred)

    for ques in perturbed_predictions :

        if original_prediction[ques] != answers[ques] :
            continue
        found = False
        change_output = {}
        change_output_obj = {}
        no_of_changes = 0
        for obj_interest in perturbed_predictions[ques] :
            pred_obj = perturbed_predictions[ques][obj_interest]
            change_output[obj_interest] = []
            change_output_obj[obj_interest] = 0

            no_of_changes += len(pred_obj)
            for i in range(len(pred_obj)) :
                if pred_obj[i] != original_prediction[ques] :
                    found = True
                    change_output[obj_interest].append(1)
                    change_output_obj[obj_interest] = 1
                else :
                    change_output[obj_interest].append(0)

        sum_ques = 0
        sum_ques_obj = 0
        for obj_interest in change_output :
            sum_obj = sum(change_output[obj_interest])
            sum_ques += sum_obj
            if sum_obj > 0 :
                sum_ques_obj += 1

        if len(change_output) > 0 :
            prop_obj = float(sum_ques_obj)/len(change_output)
        else :
            prop_obj = 0

        if no_of_changes > 1 :
            prop = float(sum_ques)/(no_of_changes)
        else :
            prop = 0

        changed_proportion += prop
        changed_proportion_obj += prop_obj

        if found :
            changed_attr += 1
            changed[ques] = {}


print ('model accuracy without SwapMix : ', float(baseline_correct)/overall_ques)
print ('object reliance : ', float(changed_obj)/total_ques)
print ('attribute reliance : ', float(changed_attr)/total_ques)
print ('context reliance : ', float(len(changed))/total_ques)
print ('effective accuracy : ', (float(baseline_correct) - float(len(changed)))/overall_ques)
#print ('changed_proportion : ', float(changed_proportion)/total_ques)
#print ('changed_proportion_obj : ', float(changed_proportion_obj)/total_ques)

