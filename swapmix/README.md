# Implementation steps for a new model

Steps to measure visual bias for a new model

1. Initialize FRCNN() function in <code>frcnn_modify.py</code> and GQADataset() function inside <code>gqa.py</code>.
2. Modify data loading section 
    - Generate iid_to_img_feat mapping by using <code>GQADataset.extractembeddings</code>
    - Iterate over each image to extract {questions,unrelevant_objects_for_each_question} using iid_to_img_feat.
3. For testing, Iterate individually over each question for the image.
4. Use <code>frcnn_modify.get_features_including_random</code> function to extract set of visual context perturbation features for each question by passing img_id and unrelevant objects info.
5. Pass all perturbation features to the model and save results to a dictionary.


For further details please refer to SwapMix implementation on mcan or lxmert.
