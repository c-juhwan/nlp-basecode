import os
import sys
import argparse
from utils.utils import check_path

def submission(args: argparse.Namespace) -> None:
    # Submission for SuperGLUE Benchmark

    task_dict = {
        'multi_classification': ['boolq', 'cb', 'multirc', 'rte', 'axb', 'axg'],
        'reading_comprehension': ['record'],
        'multiple_choice': ['copa'],
        'span_classification': ['wic', 'wsc']
    }

    # Find each task's submission file
    submission_file_list = []
    for task in task_dict.keys():
        for dataset in task_dict[task]:
            submission_file_list.append(os.path.join(args.result_path, task, dataset, args.model_type, 'test_results.jsonl'))

    print(submission_file_list)
    # Check if all submission files exist
    assert len(submission_file_list) == 10, f'Number of submission files is not 10, but {len(submission_file_list)}'
    for submission_file in submission_file_list:
        if not os.path.isfile(submission_file):
            print(f'{submission_file} does not exist.')
            sys.exit()

    # Copy file and rename them to each dataset's name
    check_path(os.path.join(args.result_path, 'submission', args.model_type))
    for submission_file in submission_file_list:
        os.system(f'cp {submission_file} {os.path.join(args.result_path, "submission", args.model_type)}')
        print(f'cp {submission_file} {os.path.join(args.result_path, "submission", args.model_type)}')

        # e.g. 'COPA.jsonl' - need capitalize
        dataset_name = submission_file.split('/')[-3].upper() + '.jsonl'

        # Rename the file
        os.system(f'mv {os.path.join(args.result_path, "submission", args.model_type, "test_results.jsonl")} {os.path.join(args.result_path, "submission", args.model_type, dataset_name)}')
        print(f'mv {os.path.join(args.result_path, "submission", args.model_type, "test_results.jsonl")} {os.path.join(args.result_path, "submission", args.model_type, dataset_name)}')

    # Rename - some datasets have different names in submission
    os.system(f'mv {os.path.join(args.result_path, "submission", args.model_type, "AXB.jsonl")} {os.path.join(args.result_path, "submission", args.model_type, "AX-b.jsonl")}')
    os.system(f'mv {os.path.join(args.result_path, "submission", args.model_type, "AXG.jsonl")} {os.path.join(args.result_path, "submission", args.model_type, "AX-g.jsonl")}')
    os.system(f'mv {os.path.join(args.result_path, "submission", args.model_type, "BOOLQ.jsonl")} {os.path.join(args.result_path, "submission", args.model_type, "BoolQ.jsonl")}')
    os.system(f'mv {os.path.join(args.result_path, "submission", args.model_type, "MULTIRC.jsonl")} {os.path.join(args.result_path, "submission", args.model_type, "MultiRC.jsonl")}')
    os.system(f'mv {os.path.join(args.result_path, "submission", args.model_type, "RECORD.jsonl")} {os.path.join(args.result_path, "submission", args.model_type, "ReCoRD.jsonl")}')
    os.system(f'mv {os.path.join(args.result_path, "submission", args.model_type, "WIC.jsonl")} {os.path.join(args.result_path, "submission", args.model_type, "WiC.jsonl")}')

    # Zip the submission folder
    os.system(f'cd {os.path.join(args.result_path, "submission")} && zip -r {args.model_type}.zip {args.model_type}')
