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
        if args.task == 'multi_classification':
            if args.job == 'preprocessing':
                from task.multi_classification.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.multi_classification.train import training as job
            elif args.job == 'testing':
                from task.multi_classification.test import testing as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        elif args.task == 'multiple_choice':
            if args.job == 'preprocessing':
                from task.multiple_choice.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.multiple_choice.train import training as job
            elif args.job == 'testing':
                from task.multiple_choice.test import testing as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        elif args.task == 'reading_comprehension':
            if args.job == 'preprocessing':
                from task.reading_comprehension.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                # As this is same as True-false classification, use the same code
                from task.multi_classification.train import training as job
            elif args.job == 'testing':
                from task.reading_comprehension.test import testing as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        elif args.task == 'span_classification':
            if args.job == 'preprocessing':
                from task.span_classification.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.span_classification.train import training as job
            elif args.job == 'testing':
                from task.span_classification.test import testing as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        elif args.task == 'submission':
            if args.job == 'submission':
                from task.submission.superglue_submission import submission as job
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