import os
import sys
import argparse
from utils.utils import check_path

def submission(args: argparse.Namespace) -> None:
    # Submission for GLUE Benchmark

    task_dict = {
        'single_classification': ['cola', 'sst2'],
        'multi_classification': ['mrpc', 'qqp', 'mnli_m', 'mnli_mm', 'qnli', 'rte', 'wnli', 'ax'],
        'regression': ['sts_b']
    }

    # Find each task's submission file
    submission_file_list = []
    for task in task_dict.keys():
        for dataset in task_dict[task]:
            submission_file_list.append(os.path.join(args.result_path, task, dataset, args.model_type, 'test_result.tsv'))

    print(submission_file_list)
    # Check if all submission files exist
    assert len(submission_file_list) == 11, f'Number of submission files is not 11, but {len(submission_file_list)}'
    for submission_file in submission_file_list:
        if not os.path.isfile(submission_file):
            print(f'{submission_file} does not exist.')
            sys.exit()

    # Copy file and rename them to each dataset's name
    check_path(os.path.join(args.result_path, 'submission', args.model_type))
    for submission_file in submission_file_list:
        os.system(f'cp {submission_file} {os.path.join(args.result_path, "submission", args.model_type)}')
        print(f'cp {submission_file} {os.path.join(args.result_path, "submission", args.model_type)}')

        # e.g. 'COLA.tsv' - need capitalize
        dataset_name = submission_file.split('/')[-3].upper() + '.tsv'

        # Rename the file
        os.system(f'mv {os.path.join(args.result_path, "submission", args.model_type, "test_result.tsv")} {os.path.join(args.result_path, "submission", args.model_type, dataset_name)}')
        print(f'mv {os.path.join(args.result_path, "submission", args.model_type, "test_result.tsv")} {os.path.join(args.result_path, "submission", args.model_type, dataset_name)}')

    # Rename - some datasets have different names in submission
    os.system(f'mv {os.path.join(args.result_path, "submission", args.model_type, "MNLI_M.tsv")} {os.path.join(args.result_path, "submission", args.model_type, "MNLI-m.tsv")}')
    os.system(f'mv {os.path.join(args.result_path, "submission", args.model_type, "MNLI_MM.tsv")} {os.path.join(args.result_path, "submission", args.model_type, "MNLI-mm.tsv")}')
    os.system(f'mv {os.path.join(args.result_path, "submission", args.model_type, "SST2.tsv")} {os.path.join(args.result_path, "submission", args.model_type, "SST-2.tsv")}')
    os.system(f'mv {os.path.join(args.result_path, "submission", args.model_type, "STS_B.tsv")} {os.path.join(args.result_path, "submission", args.model_type, "STS-B.tsv")}')
    os.system(f'mv {os.path.join(args.result_path, "submission", args.model_type, "COLA.tsv")} {os.path.join(args.result_path, "submission", args.model_type, "CoLA.tsv")}')

    # Zip the submission folder
    os.system(f'cd {os.path.join(args.result_path, "submission")} && zip -r {args.model_type}.zip {args.model_type}')
