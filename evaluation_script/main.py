import random
import json
import pickle as pkl
import copy
import math
import numpy as np
import os

def load_jsonl(input_file) -> list:
    """Load jsonl-format ground-truth and result."""
    result_data_dict = {}
    with open(input_file, "r") as fin:
        data_lines = fin.readlines()
        for line in data_lines:
            # Remove empty row.
            line = line.strip()
            if len(line) < 1:
                continue
            # Decode.
            data_dict = json.loads(line)
            # Push the data into a list.
            result_data_dict[data_dict["index"]] = data_dict
    return result_data_dict

def compare(gt_candidates: dict, data_candidates: dict) -> float:
    """Compare one query gt and prediction and get ap.

    Args:
        gt_candidates(Dict): gt dict for each query,
        data_candidates(Dict): prediction dict for each query
    
    Return average_precision, the label for the first query
    """
    # A dict to store the score.
    data_candidates_dict = {}
    # A dict to store the gt.
    gt_candidates_dict = {}

    # A list 
    score_list = []
    for candidate in data_candidates:
        data_candidates_dict[candidate["candidate_asin"]] = candidate["score"]

    for candidate in gt_candidates:
        gt_candidates_dict[candidate["candidate_asin"]] = (candidate["annotation"] == "fulfill")

    for key in gt_candidates_dict:
        if key not in data_candidates_dict:
            score = 0
        else:
            score = data_candidates_dict[key]
        score_list.append((score, gt_candidates_dict[key]))
    score_list.sort(reverse = True, key = lambda x: x[0])

    positive_count = 0
    total_acc = 0

    for idx in range(len(score_list)):
        if score_list[idx][1]:
            positive_count+=1
            total_acc += positive_count/(idx+1)
    
    ap = total_acc/positive_count
    return ap, score_list[0][1]

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    print("Submission related metadata:")
    """
    Evaluates the submission for a particular challenge phase adn returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            "status": u"running",
            "when_made_public": None,
            "participant_team": 5,
            "input_file": "https://abc.xyz/path/to/submission/file.json",
            "execution_time": u"123",
            "publication_url": u"ABC",
            "challenge_phase": 1,
            "created_by": u"ABC",
            "stdout_file": "https://abc.xyz/path/to/stdout/file.json",
            "method_name": u"Test",
            "stderr_file": "https://abc.xyz/path/to/stderr/file.json",
            "participant_team_name": u"Test Team",
            "project_url": u"http://foo.bar",
            "method_description": u"ABC",
            "is_public": False,
            "submission_result_file": "https://abc.xyz/path/result/file.json",
            "id": 123,
            "submitted_at": u"2017-03-20T19:22:03.880652Z",
        }
    """
    print(kwargs["submission_metadata"])
    output = {}
    gt_dict = load_jsonl(test_annotation_file)
    if phase_codename == "dev":
        data_dict = load_jsonl(user_submission_file)
        print("Evaluating for Dev Phase")
        # get AP for each query
        ap_list = []
        first_one_correct_count = 0
        for idx in gt_dict:
            if idx not in data_dict:
                ap_list.append(0)
                continue
            ap, label_at_1 = compare(gt_dict[idx]["candidates"], data_dict[idx]["candidates"])
            first_one_correct_count += label_at_1
            ap_list.append(ap)

        total_ap = 0
        for ap in ap_list:
            total_ap+=ap
        mAP = sum(ap_list)/len(ap_list)

        output["result"] = [
            {
                "val_split": {
                    "mAP": mAP,
                    "P@1": first_one_correct_count/len(ap_list),
                    "max_AP": max(ap_list),
                    "min_AP": min(ap_list),
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["val_split"]
        print("Completed evaluation for Dev Phase")

    elif phase_codename == "veri" or phase_codename == "eval":
        # Set phase name and split name.
        feature_dim = 128
        if phase_codename == "eval":
            phase_name = "Evaluation"
            split_name = "eval_split"
        elif phase_codename == "veri":
            phase_name = "Verification"
            split_name = "veri_split"

        print(f"Evaluating for {phase_name} Phase")

        file_stats = os.stat(user_submission_file)    
        # Load features.
        assert file_stats.st_size/(1024 * 1024) < 500, "Submission file size shouldn't larger than 500 MB."
    
        with open(user_submission_file, 'rb') as f:
            submitted_result = pkl.load(f)
        query_feature = submitted_result["query_feature"]
        gallery_feature = submitted_result["gallery_feature"]

        assert query_feature.shape[1] == feature_dim, f"query_feature dimension != {feature_dim}"
        assert gallery_feature.shape[1] == feature_dim, f"gallery_feature dimension != {feature_dim}"

        ap_list = []
        first_one_correct_count = 0
        
        # Get Metric
        for idx in gt_dict:
            data_dict = copy.deepcopy(gt_dict[idx])
            local_query_feature = query_feature[idx-1, :]

            for candidate_idx, candidate in enumerate(gt_dict[idx]["candidates"]):
                max_score = 0
                for local_idx in candidate["feature_index_list"]:
                    local_gallery_feature = gallery_feature[local_idx - 1, :]
                    l2_norm = math.sqrt(np.dot(local_gallery_feature, local_gallery_feature))
                    if l2_norm<1e-5:
                        continue
                    else:
                        score = (np.dot(local_query_feature, local_gallery_feature)+1)/2
                        max_score = max(max_score, score)
                data_dict["candidates"][candidate_idx]["score"] = max_score
            ap, label_at_1 = compare(gt_dict[idx]["candidates"], data_dict["candidates"])
            first_one_correct_count += label_at_1
            ap_list.append(ap)

        mAP = sum(ap_list)/len(ap_list)
        output["result"] = [
            {
                split_name: {
                    "mAP": mAP,
                    "P@1": first_one_correct_count/len(ap_list),
                    "max_AP": max(ap_list),
                    "min_AP": min(ap_list),
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0][split_name]
        print(f"Completed evaluation for {phase_name} Phase")
    return output
