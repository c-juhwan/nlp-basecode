# Standard Library Modules
import time
import argparse
# Custom Modules
from utils.arguments import ArgParser
from utils.utils import check_path, set_random_seed

def main(args: argparse.Namespace) -> None:
    # Set random seed
    if args.seed is not None:
        set_random_seed(args.seed)

    start_time = time.time()

    # Check if the path exists
    for path in []:
        check_path(path)

    # Get the job to do
    if args.job == None:
        raise ValueError('Please specify the job to do.')
    else:
        if args.task == 'single_classification':
            if args.job == 'preprocessing':
                from task.single_classification.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.single_classification.train import training as job
            elif args.job == 'testing':
                from task.single_classification.test import testing as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        elif args.task == 'multi_classification':
            if args.job == 'preprocessing':
                from task.multi_classification.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.multi_classification.train import training as job
            elif args.job == 'testing':
                from task.multi_classification.test import testing as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        elif args.task == 'regression':
            if args.job == 'preprocessing':
                from task.regression.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.regression.train import training as job
            elif args.job == 'testing':
                from task.regression.test import testing as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        elif args.task == 'submission':
            if args.job == 'submission':
                from task.submission.glue_submission import submission as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        else:
            raise ValueError(f'Invalid task: {args.task}')

    # Do the job
    job(args)

    elapsed_time = time.time() - start_time
    print(f'Completed {args.job}; Time elapsed: {elapsed_time / 60:.2f} minutes')

if __name__ == '__main__':
    # Parse arguments
    parser = ArgParser()
    args = parser.get_args()

    # Run the main function
    main(args)